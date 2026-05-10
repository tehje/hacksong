#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="/home/zhangj/miniconda3/envs/py310/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "py310 interpreter not found: $PYTHON_BIN" >&2
  exit 1
fi

missing_modules="$("$PYTHON_BIN" - <<'PY'
import importlib.util

checks = [
    ("uvicorn", "uvicorn"),
    ("fastapi", "fastapi"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("transformers", "transformers"),
    ("sentence-transformers", "sentence_transformers"),
    ("chromadb", "chromadb"),
    ("faster-whisper", "faster_whisper"),
    ("Pillow", "PIL"),
    ("python-multipart", "multipart"),
]

missing = [package for package, module_name in checks if importlib.util.find_spec(module_name) is None]
if missing:
    print(",".join(missing))
PY
)"

if [[ -n "$missing_modules" ]]; then
  echo "py310 is missing required packages: $missing_modules" >&2
  echo "Install them with:" >&2
  echo "  $PYTHON_BIN -m pip install -r $ROOT_DIR/long_video_pipeline/requirements.txt" >&2
  exit 1
fi

exec "$PYTHON_BIN" -m uvicorn long_video_pipeline.service:app --host 0.0.0.0 --port 8008
