from __future__ import annotations

import json
import re
import logging
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

# --- minimal guardrail helpers (작게 추가: 가독성 목적) ---
_JSON_BLOCK = re.compile(r"(\{[\s\S]*\}|\[[\s\S]*\])", re.DOTALL)

def _looks_like_json(text: str) -> bool:
    t = text.strip()
    return t.startswith("{") or t.startswith("[")

def _strip_code_fence(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        # ```json ... ``` 또는 ``` ... ``` 제거
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.DOTALL).strip()
    return s

def _extract_first_json_block(s: str) -> str:
    if _looks_like_json(s):
        return s.strip()
    m = _JSON_BLOCK.search(s)
    return m.group(1).strip() if m else s

# CrewAI 이벤트 임포트 (신/구 버전 호환)
try:
    from crewai.events import CrewAIEventsBus
    from crewai.events import (
        TaskStartedEvent,
        TaskCompletedEvent,
        ToolUsageStartedEvent,
        ToolUsageFinishedEvent,
    )
except ImportError:  # 구버전
    from crewai.utilities.events import CrewAIEventsBus
    from crewai.utilities.events.task_events import TaskStartedEvent, TaskCompletedEvent
    from crewai.utilities.events import ToolUsageStartedEvent, ToolUsageFinishedEvent

# context_var import
from .context_manager import task_id_var, proc_inst_id_var, crew_type_var

# database 저장 함수 import
from .database import save_event_sync, initialize_db


class CrewAIEventLogger:
    """CrewAI 이벤트 → events 테이블 저장 (단순/가독성 우선)"""

    # --------- 공개 엔트리포인트 ---------
    def on_event(self, event: Any, source: Any = None) -> None:
        """
        이벤트 수신 → (job_id, event_type, data) 추출 → DB 저장
        - 동기 메서드이며 내부에서는 save_event_sync로 저장 수행
        - 모든 예외는 상위로 전파
        """
        logger.info("📨 CrewAI 이벤트 수신 시작 | event_class=%s", event.__class__.__name__ if event else "None")
        
        try:
            # DB 초기화 확인 및 실행
            try:
                initialize_db()
            except Exception as db_e:
                logger.error("❌ DB 초기화 실패, 이벤트 저장 건너뜀 | err=%s", str(db_e), exc_info=True)
                raise

            event_type = self._extract_event_type(event)
            # 지원 타입만 처리
            if event_type not in ("task_started", "task_completed", "tool_usage_started", "tool_usage_finished"):
                return

            job_id = self._extract_job_id(event, source)
            data = self._extract_data(event, event_type)

            # context_var
            todo_id = task_id_var.get()
            proc_inst_id = proc_inst_id_var.get()
            crew_type = crew_type_var.get()

            # DB 저장
            event_id = save_event_sync(
                job_id=job_id,
                todo_id=todo_id,
                proc_inst_id=proc_inst_id,
                crew_type=crew_type,
                data=data,
                event_type=event_type,
                status=None,
            )
            logger.info("✅ 이벤트 DB 저장 완료 | event_id=%s job_id=%s type=%s crew_type=%s todo_id=%s proc_inst_id=%s",
                event_id, job_id, event_type, str(crew_type), str(todo_id), str(proc_inst_id))

        except Exception as e:
            logger.error("❌ CrewAI 이벤트 처리 실패 | event_class=%s err=%s", event.__class__.__name__ if event else "None", str(e), exc_info=True)
            raise

    # --------- 헬퍼(가독성 유지용 최소) ---------
    def _extract_job_id(self, event: Any, source: Any = None) -> str:
        try:
            if hasattr(event, "task") and hasattr(event.task, "id"):
                return str(event.task.id)
            if source and hasattr(source, "task") and hasattr(source.task, "id"):
                return str(source.task.id)
            if hasattr(event, "job_id"):
                return str(getattr(event, "job_id"))
        except Exception as e:
            logger.warning("⚠️ job_id 추출 중 예외 발생 | err=%s", str(e), exc_info=True)
        logger.warning("⚠️ job_id 추출 실패 - 기본값 사용 | job_id=unknown")
        return "unknown"

    def _extract_event_type(self, event: Any) -> str:
        try:
            if hasattr(event, "type") and isinstance(event.type, str):
                return event.type
        except Exception as e:
            logger.debug("⚠️ 이벤트 타입 속성 접근 실패 | err=%s", str(e))
            pass
        name = event.__class__.__name__.lower()
        if "taskstarted" in name:
            return "task_started"
        if "taskcompleted" in name:
            return "task_completed"
        if "toolusagestarted" in name:
            return "tool_usage_started"
        if "toolusagefinished" in name:
            return "tool_usage_finished"
        
        logger.warning("⚠️ 알 수 없는 이벤트 타입 | class_name=%s event_type=unknown", name)
        return "unknown"

    def _extract_data(self, event: Any, event_type: str) -> Dict[str, Any]:
        try:
            if event_type == "task_started":
                task = getattr(event, "task", None)
                agent = getattr(task, "agent", None) if task else None
                return {
                    "role": getattr(agent, "role", None) or "Unknown",
                    "goal": getattr(agent, "goal", None) or "Unknown",
                    "agent_profile": getattr(agent, "profile", None) or "/images/chat-icon.png",
                    "name": getattr(agent, "name", None) or "Unknown",
                    "task_description": getattr(task, "description", None),
                }

            if event_type == "task_completed":
                # output 우선순위: event.output.raw -> event.output(str) -> event.result
                output = getattr(event, "output", None)
                text = getattr(output, "raw", None)
                if text is None:
                    text = output if isinstance(output, str) else getattr(event, "result", None)
                parsed = self._safe_json(text)

                # ✅ planning 포맷(list_of_plans_per_task) → Markdown 축약
                #    이 시점에 planning 단계가 완료되었다고 보고 crew_type을 action으로 전환
                # ! 좀 더 범용적으로 수정이 필요 임시 조취
                if isinstance(parsed, dict) and "list_of_plans_per_task" in parsed:
                    md = self._format_plans_md(parsed["list_of_plans_per_task"])
                    # crew_type을 planning → action으로 전환 (향후 이벤트는 action으로 기록)
                    try:
                        from .context_manager import crew_type_var
                        crew_type_var.set("action")
                        logger.info("🔁 crew_type 전환: planning → action (list_of_plans_per_task 감지)")
                    except Exception as e:
                        logger.warning("⚠️ crew_type 전환(planning→action) 중 예외 발생: %s", str(e), exc_info=True)
                    return {"plans": md}

                return {"result": parsed}

            if event_type in ("tool_usage_started", "tool_usage_finished"):
                tool_name = getattr(event, "tool_name", None)
                tool_args = getattr(event, "tool_args", None)
                args = self._safe_json(tool_args)
                query = args.get("query") if isinstance(args, dict) else None
                return {"tool_name": tool_name, "query": query, "args": args}

            logger.warning("⚠️ 처리되지 않는 이벤트 타입 | event_type=%s", event_type)
            return {"info": f"Unhandled event type: {event_type}"}

        except Exception as e:
            logger.error("❌ 이벤트 데이터 추출 실패 | event_type=%s err=%s", event_type, str(e), exc_info=True)
            raise

    # --------- 단순 유틸 ---------
    def _safe_json(self, value: Any) -> Any:
        """문자열 결과를 견고하게 JSON으로 파싱(최대 2회 디코딩).
        - 1차: 그대로 json.loads
        - 실패 시: 코드펜스 제거 + 첫 JSON 블록 추출 후 재시도
        - 각 단계에서 결과가 'JSON 문자열'이면 추가로 1회만 더 파싱
        - 여전히 실패면 원문 반환(보수적)
        """
        if value is None or isinstance(value, (dict, list)):
            return value
        if not isinstance(value, str):
            return value

        def _loads_once(s: str):
            try:
                return True, json.loads(s)
            except Exception:
                return False, None

        def _maybe_decode_nested(obj: Any) -> Any:
            # 결과가 "JSON을 담은 문자열"이면 거기까지만 1회 더 파싱
            if isinstance(obj, str):
                s2 = obj.strip()
                if _looks_like_json(s2):
                    ok2, obj2 = _loads_once(s2)
                    if ok2:
                        return obj2
            return obj

        # 1) 있는 그대로 1차 시도
        ok, obj = _loads_once(value)
        if ok:
            return _maybe_decode_nested(obj)

        # 2) 정리 후 재시도(코드펜스 제거 + 첫 JSON 블록 추출)
        s = _strip_code_fence(value)
        s = _extract_first_json_block(s)
        ok, obj = _loads_once(s)
        if ok:
            return _maybe_decode_nested(obj)

        # 3) 모두 실패 → 원문 반환
        logger.debug("⚠️ JSON 파싱 실패 (원문 반환) | snippet=%s", value[:120])
        return value

    def _format_plans_md(self, plans: List[Dict[str, Any]]) -> str:
        """list_of_plans_per_task → Markdown 문자열로 축약"""
        lines: List[str] = []
        for idx, item in enumerate(plans, 1):
            task = item.get("task", "")
            plan = item.get("plan", "")
            lines.append(f"## {idx}. {task}")
            lines.append("")
            if isinstance(plan, list):
                for line in plan:
                    lines.append(str(line))
            elif isinstance(plan, str):
                lines.extend(plan.splitlines())
            else:
                lines.append(str(plan))
            lines.append("")
        return "\n".join(lines).strip()


class CrewConfigManager:
    """글로벌 CrewAI 이벤트 리스너 등록 매니저"""
    _registered = False

    def __init__(self):
        self.logger = CrewAIEventLogger()
        logger.info("✅ CrewConfigManager 초기화 완료")
        
        # 한번만 리스너 등록
        if not CrewConfigManager._registered:
            try:
                bus = CrewAIEventsBus()
                for evt in (TaskStartedEvent, TaskCompletedEvent, ToolUsageStartedEvent, ToolUsageFinishedEvent):
                    bus.on(evt)(lambda source, event, logger=self.logger: logger.on_event(event, source))
                CrewConfigManager._registered = True
                logger.info("✅ CrewAI 이벤트 리스너 등록 완료 | registered_events=4")
            except Exception as e:
                logger.error("❌ CrewAI 이벤트 리스너 등록 실패 | err=%s", str(e), exc_info=True)
                raise
        else:
            logger.info("⏭️ CrewAI 이벤트 리스너 이미 등록됨 - 생략")
