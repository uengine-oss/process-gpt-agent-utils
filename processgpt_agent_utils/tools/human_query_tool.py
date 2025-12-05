from __future__ import annotations

import json
import logging
import hashlib
import time
import uuid
from typing import Optional, List, Type, Dict, Any, Literal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..utils.context_manager import get_context_snapshot
from ..utils.database import (
    fetch_human_response_sync,
    save_notification_sync,
    save_event_sync,
    fetch_events_by_todo_id,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------
# 스키마
# ---------------------------------------------------------------------
class HumanQuerySchema(BaseModel):
    """사용자 확인/추가정보 요청 스키마 (간결 버전)"""
    role: str = Field(..., description="질의 대상(예: user, manager)")
    text: str = Field(..., description="질의 내용")
    type: Literal["text", "select", "confirm"] = Field(default="text", description="질의 유형")
    options: Optional[List[str]] = Field(default=None, description="type이 select일 때 선택지")


# ---------------------------------------------------------------------
# 본체
# ---------------------------------------------------------------------
class HumanQueryTool(BaseTool):
    """사람에게 질문을 보내고, DB(events)에서 응답을 감지하는 도구."""

    name: str = "human_asked"
    description: str = (
        "🚨 중요: 각 질문마다 한 번씩만 호출하세요. 같은 질문이나 비슷한 질문을 반복하면 안 됩니다.\n\n"
        "[1] 언제 사용해야 하나 (매우 제한적 사용)\n"
        "이 도구는 다음 조건을 모두 만족하는 경우에만 사용하세요:\n"
        "- 컨텍스트나 지침이 근본적으로 모호하여 주제 및 핵심 방향이 정해지지 않은 경우\n"
        "- 작업의 목적, 범위, 방향성 자체가 불명확하여 추측으로 진행할 수 없는 경우\n"
        "- 보안에 민감한 정보를 다루거나 데이터베이스 저장/수정/삭제 작업을 수행해야 하는 경우\n"
        "⛔ 단순히 세부 정보가 부족한 경우는 이 도구를 사용하지 말고, 기존 컨텍스트와 지침을 바탕으로 추론하여 진행하세요.\n\n"
        "[2] 응답 타입과 작성 방식 (항상 JSON으로 질의 전송)\n"
        "- 공통 형식: { role: <누구에게>, text: <질의>, type: <text|select|confirm>, options?: [선택지...] }\n"
        "- 질의는 한 번에 모든 필요한 정보를 묻도록 작성하세요\n\n"
        "// 1) type='text' — 근본적인 방향성/주제가 불명확할 때만 사용\n"
        "{\n"
        '  "role": "user",\n'
        '  "text": "이 작업의 핵심 목적과 방향성을 명확히 해주세요. 어떤 결과물을 만들어야 하나요?",\n'
        '  "type": "text"\n'
        "}\n\n"
        "// 2) type='select' — 여러 옵션 중 선택(옵션은 상호배타적, 명확/완전하게 제시)\n"
        "{\n"
        '  "role": "system",\n'
        '  "text": "배포 환경을 선택하세요. 선택 근거(위험/롤백/감사 로그)를 함께 알려주세요.",\n'
        '  "type": "select",\n'
        '  "options": ["dev", "staging", "prod"]\n'
        "}\n\n"  
        "// 3) type='confirm' — 보안/DB 변경 등 민감 작업 승인(필수)\n"
        "{\n"
        '  "role": "user",\n'
        '  "text": "DB에서 주문 상태를 shipped로 업데이트합니다. 대상: order_id=..., 영향 범위: ...건, 롤백: ..., 진행 승인하시겠습니까?",\n'
        '  "type": "confirm"\n'
        "}\n\n"
        "[3] 주의사항 (반드시 준수)\n"
        "- ⚠️ 같은 질문을 여러 번 반복하면 안 됩니다. 똑같은 질문 금지.\n"
        "- ⚠️ 비슷한 질문도 반복하면 안 됩니다. 각 호출마다 완전히 다른 질문이어야 합니다.\n"
        "- ⚠️ 주제 및 핵심 방향이 정해지지 않은 경우에만 사용하세요.\n"
        "- ⚠️ 단순히 세부 정보가 부족한 경우는 사용하지 말고 기존 정보로 추론하세요.\n"
        "- select 타입은 반드시 'options'를 포함하세요.\n"
        "- confirm 응답에 따라: ✅ 승인 → 즉시 수행 / ❌ 거절 → 즉시 중단(건너뛰기).\n"
        "- 타임아웃/미응답 시 '사용자 미응답 거절'을 반환하며, 후속 변경 작업을 중단하세요.\n"
        "- 한국어 존댓말 사용, 간결하되 상세하게 작성하세요.")

    args_schema: Type[HumanQuerySchema] = HumanQuerySchema

    def __init__(
        self,
        *,
        proc_inst_id: str,
        task_id: str,
        tenant_id: Optional[str] = None,
        agent_name: Optional[str] = None,
        user_ids_csv: Optional[str] = None,  # 알림 대상 (CSV)
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._proc_inst_id = proc_inst_id
        self._task_id = task_id
        self._tenant_id = tenant_id
        self._agent_name = agent_name
        self._user_ids_csv = user_ids_csv

        logger.info("\n\n✅ HumanQueryTool 초기화 완료 | proc_inst_id=%s task_id=%s tenant_id=%s agent_name=%s user_ids_csv=%s", proc_inst_id, task_id, tenant_id, agent_name, user_ids_csv)

    @staticmethod
    def _make_signature(role: str, text: str, type: str, options: List[str]) -> str:
        """role/text/type/options 조합을 해시로 정규화."""
        payload = {
            "role": role,
            "text": text,
            "type": type,
            "options": options,
        }
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    # CrewAI Tool 규약: 동기 실행 (내부 비동기 작업은 sync 래퍼 사용)
    def _run(self, role: str, text: str, type: str = "text", options: Optional[List[str]] = None) -> str:
        logger.info("\n\n👤 사용자 확인 요청 시작 | role=%s type=%s", role, type)
        
        # 1) 컨텍스트 정보 가져오기
        ctx = get_context_snapshot()
        crew_type = ctx.get("crew_type")

        # 2) 중복 질의 확인 및 기존 응답 재사용
        normalized_options = options or []
        signature = self._make_signature(role, text, type, normalized_options)
        existing_events: List[Dict[str, Any]] = []
        try:
            if self._task_id:
                existing_events = fetch_events_by_todo_id(self._task_id)
        except Exception as e:
            logger.warning("⚠️ 기존 이벤트 조회 실패(무시) | task_id=%s err=%s", self._task_id, str(e), exc_info=True)

        if existing_events:
            try:
                # 가장 최근 이벤트부터 역순으로 스캔하며 동일 질의 여부를 판단
                for event in reversed(existing_events):
                    if event.get("event_type") != "human_asked":
                        continue
                    data = event.get("data") or {}
                    if data.get("signature") == signature:
                        prev_job_id = event.get("job_id")
                        if not prev_job_id:
                            continue

                        # 동일 job_id에 대한 응답을 먼저 탐색
                        for resp in reversed(existing_events):
                            if resp.get("event_type") != "human_response":
                                continue
                            if resp.get("job_id") != prev_job_id:
                                continue

                            resp_data = resp.get("data") or {}
                            answer = resp_data.get("answer")
                            if isinstance(answer, str):
                                logger.info("♻️ 기존 사용자 응답 재사용 | job_id=%s", prev_job_id)
                                return answer
                            logger.info("♻️ 기존 사용자 응답 재사용(JSON) | job_id=%s", prev_job_id)
                            return json.dumps(resp_data, ensure_ascii=False)

                        # 응답이 아직 없는 동일 질의가 이미 등록된 경우, 새로 묻지 않고 기존 job으로 대기
                        logger.info("⏳ 기존 사용자 응답 대기 재사용 | job_id=%s", prev_job_id)
                        return self._wait_for_response(prev_job_id)
            except Exception as e:
                logger.warning("⚠️ 기존 질의 중복 확인 실패(무시) | err=%s", str(e), exc_info=True)

        # 3) 메시지 페이로드 구성
        payload: Dict[str, Any] = {
            "role": role,
            "text": text,
            "type": type,
            "options": normalized_options,
            "signature": signature,
        }

        # 3) job_id 발급
        job_id = f"human_asked_{uuid.uuid4()}"

        # 4) 이벤트를 DB에 직접 저장
        try:
            save_event_sync(
                job_id=job_id,
                todo_id=self._task_id,
                proc_inst_id=self._proc_inst_id,
                crew_type=crew_type,
                data=payload,
                event_type="human_asked",
            )
            logger.info("✅ 사용자 확인 이벤트 DB 저장 완료 | proc=%s task=%s job_id=%s", self._proc_inst_id, self._task_id, job_id)
        except Exception as e:
            logger.error("❌ 사용자 확인 이벤트 DB 저장 실패 | proc=%s task=%s job_id=%s err=%s", self._proc_inst_id, self._task_id, job_id, str(e), exc_info=True)
            raise

        # 5) 알림 저장 (있으면)
        try:
            if self._user_ids_csv and self._user_ids_csv.strip():
                save_notification_sync(
                    title=text,
                    notif_type="workitem_bpm",
                    description=self._agent_name,
                    user_ids_csv=self._user_ids_csv,
                    tenant_id=self._tenant_id,
                    url=f"/todolist/{self._task_id}" if self._task_id else None,
                    from_user_id=self._agent_name,
                )
                logger.info("✅ 사용자 알림 저장 완료 | user_ids_csv=%s", self._user_ids_csv)
            else:
                logger.info("⏭️ 사용자 알림 저장 생략: user_ids_csv 비어있음")
        except Exception as e:
            logger.error("❌ 사용자 알림 저장 실패 | user_ids_csv=%s err=%s", self._user_ids_csv, str(e), exc_info=True)
            raise

        # 6) DB에서 사람 응답 폴링
        logger.info("\n\n⏳ 사용자 응답 대기 시작 | job_id=%s", job_id)
        answer = self._wait_for_response(job_id)
        logger.info("✅ 사용자 응답 수신 완료 | job_id=%s answer_length=%d", job_id, len(answer) if answer else 0)
        return answer

    # -----------------------------------------------------------------
    # 응답 폴링 (DB events 테이블)
    # -----------------------------------------------------------------
    def _wait_for_response(self, job_id: str, timeout_sec: int = 180, poll_interval_sec: int = 5) -> str:
        deadline = time.time() + timeout_sec
        error_count = 0

        while time.time() < deadline:
            try:
                event = fetch_human_response_sync(job_id=job_id)
                if event:
                    data = (event.get("data") or {})
                    answer = data.get("answer")
                    if isinstance(answer, str):
                        logger.info("✅ 사용자 응답 수신 성공 | job_id=%s", job_id)
                        return answer
                    return json.dumps(data, ensure_ascii=False)
                error_count = 0  # 성공 시 에러 카운트 리셋
            except Exception as e:
                logger.error("❌ 사용자 응답 폴링 오류 | job_id=%s err=%s", job_id, str(e), exc_info=True)
                error_count += 1
                if error_count >= 3:
                    logger.error("💥 사용자 응답 폴링 중단 | job_id=%s 연속 오류 3회", job_id)
                    raise RuntimeError("human_asked polling aborted after 3 consecutive errors") from e
                logger.warning("⚠️ 사용자 응답 폴링 재시도 | job_id=%s error_count=%d", job_id, error_count)
            
            time.sleep(poll_interval_sec)

        logger.warning("⏰ 사용자 응답 타임아웃 | job_id=%s timeout=%ds", job_id, timeout_sec)
        return "사용자 미응답 거절"

