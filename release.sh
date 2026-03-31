#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"
TARGET="${2:-pypi}" # pypi | testpypi

if [[ -z "$VERSION" ]]; then
  echo "Usage: ./release.sh <version> [pypi|testpypi]" >&2
  exit 1
fi

# 0) .env 자동 로드(있으면)
if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

# 1) 버전 반영
if sed --version >/dev/null 2>&1; then
  sed -i -E "s/^version\s*=\s*\"[^\"]+\"/version = \"$VERSION\"/" pyproject.toml
else
  sed -i '' -E "s/^version\s*=\s*\"[^\"]+\"/version = \"$VERSION\"/" pyproject.toml
fi

# 2) 빌드 정리 및 생성
rm -rf dist build *.egg-info || true
python -m pip install --upgrade build twine >/dev/null
python -m build

# 3) 메타 검증
python -m twine check dist/*

# 4) 업로드 (UTF-8 강제 + 진행바 비활성화)
export PYTHONIOENCODING="utf-8"
if [[ "$TARGET" == "testpypi" ]]; then
  : "${TEST_PYPI_TOKEN:?TEST_PYPI_TOKEN env required}"
  python -m twine upload --disable-progress-bar --repository-url https://test.pypi.org/legacy/ -u __token__ -p "$TEST_PYPI_TOKEN" dist/*
else
  : "${PYPI_TOKEN:?PYPI_TOKEN env required}"
  python -m twine upload --disable-progress-bar -u __token__ -p "$PYPI_TOKEN" dist/*
fi

echo "🎉 Released process-gpt-agent-utils $VERSION to $TARGET"


