#!/usr/bin/env bash
set -euo pipefail

# =========================
# Qwen3-VL Instruct + Thinking downloader (CN-friendly)
# Prefer ModelScope; optionally use HuggingFace mirror.
# =========================

INSTRUCT_MODEL="Qwen/Qwen3-VL-8B-Instruct"
THINKING_MODEL="Qwen/Qwen3-VL-8B-Thinking"

# Where to store models
ROOT_DIR="${1:-$PWD/models}"
mkdir -p "$ROOT_DIR"

echo "[INFO] Target dir: $ROOT_DIR"
echo "[INFO] Models:"
echo "  - $INSTRUCT_MODEL"
echo "  - $THINKING_MODEL"
echo

# -------- helpers --------
have_cmd() { command -v "$1" >/dev/null 2>&1; }

install_pip_pkg_hint() {
  local pkg="$1"
  echo "[HINT] You can install it via: python3 -m pip install -U $pkg"
}

# -------- option A: ModelScope (recommended in China) --------
download_via_modelscope() {
  echo "[INFO] Trying ModelScope download..."
  if ! have_cmd python3; then
    echo "[ERROR] python3 not found."
    exit 1
  fi

  # Ensure modelscope is installed
  python3 - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("modelscope") else 1)
PY
  if [[ $? -ne 0 ]]; then
    echo "[WARN] modelscope not installed."
    install_pip_pkg_hint "modelscope"
    echo "[INFO] Installing modelscope now..."
    python3 -m pip install -U modelscope
  fi

  # Use modelscope CLI if present; otherwise use python one-liner.
  if have_cmd modelscope; then
    echo "[INFO] Using modelscope CLI..."
    modelscope download --model "$INSTRUCT_MODEL" --local_dir "$ROOT_DIR/qwen3-vl-8b-instruct"
    modelscope download --model "$THINKING_MODEL" --local_dir "$ROOT_DIR/qwen3-vl-8b-thinking"
  else
    echo "[INFO] modelscope CLI not found; using Python API..."
    python3 - <<PY
from modelscope.hub.snapshot_download import snapshot_download
snapshot_download(model_id="${INSTRUCT_MODEL}", local_dir=r"${ROOT_DIR}/qwen3-vl-8b-instruct")
snapshot_download(model_id="${THINKING_MODEL}", local_dir=r"${ROOT_DIR}/qwen3-vl-8b-thinking")
print("Done.")
PY
  fi

  echo "[OK] ModelScope download finished."
}

# -------- option B: HuggingFace mirror (optional) --------
# Requires: git-lfs + huggingface_hub (hf CLI) OR git clone
# You can set HF_ENDPOINT to a mirror endpoint.
download_via_hf_mirror() {
  echo "[INFO] Trying Hugging Face mirror download..."

  if ! have_cmd git; then
    echo "[ERROR] git not found."
    exit 1
  fi
  if ! have_cmd git-lfs; then
    echo "[WARN] git-lfs not found. Large files may not download correctly."
    echo "[HINT] Ubuntu/Debian: sudo apt-get update && sudo apt-get install -y git-lfs && git lfs install"
    echo "[HINT] CentOS/RHEL: sudo yum install -y git-lfs && git lfs install"
    echo "[HINT] Then rerun this script."
    exit 1
  fi

  git lfs install >/dev/null 2>&1 || true

  # Option 1: use HF_ENDPOINT + huggingface-cli snapshot-download (recommended)
  if have_cmd python3; then
    python3 - <<'PY'
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("huggingface_hub") else 1)
PY
    if [[ $? -ne 0 ]]; then
      echo "[WARN] huggingface_hub not installed."
      install_pip_pkg_hint "huggingface_hub"
      echo "[INFO] Installing huggingface_hub now..."
      python3 -m pip install -U huggingface_hub
    fi

    # Set HF_ENDPOINT to your reachable mirror if needed:
    # export HF_ENDPOINT=https://hf-mirror.com
    # export HF_HUB_ENABLE_HF_TRANSFER=1  (optional)
    echo "[INFO] Using huggingface-cli snapshot-download..."
    python3 -m huggingface_hub.cli snapshot_download \
      "$INSTRUCT_MODEL" \
      --local-dir "$ROOT_DIR/qwen3-vl-8b-instruct" \
      --local-dir-use-symlinks False

    python3 -m huggingface_hub.cli snapshot_download \
      "$THINKING_MODEL" \
      --local-dir "$ROOT_DIR/qwen3-vl-8b-thinking" \
      --local-dir-use-symlinks False

    echo "[OK] HuggingFace mirror snapshot-download finished."
    return 0
  fi

  # Option 2: fallback to git clone (uses whatever git remote is reachable)
  # You can replace the base URL with a mirror like:
  # https://hf-mirror.com/Qwen/Qwen3-VL-8B-Instruct
  # or a gitcode mirror (if you prefer).
  echo "[WARN] python3 not available, falling back to git clone."
  echo "[HINT] If huggingface.co is slow, set HF mirror URL in the repo below."

  ( cd "$ROOT_DIR" && \
    git clone "https://huggingface.co/${INSTRUCT_MODEL}" "qwen3-vl-8b-instruct" && \
    git clone "https://huggingface.co/${THINKING_MODEL}" "qwen3-vl-8b-thinking" )

  echo "[OK] git clone finished."
}

# -------- main flow --------
# 1) ModelScope first (best in CN)
# 2) If it fails, try HF mirror
set +e
download_via_modelscope
MS_OK=$?
set -e

if [[ $MS_OK -ne 0 ]]; then
  echo "[WARN] ModelScope download failed, trying HF mirror..."
  download_via_hf_mirror
fi

echo
echo "[DONE] Models saved under:"
echo "  - $ROOT_DIR/qwen3-vl-8b-instruct"
echo "  - $ROOT_DIR/qwen3-vl-8b-thinking"
echo
echo "[TIP] If you want HF mirror acceleration, before running you can do:"
echo "  export HF_ENDPOINT=https://hf-mirror.com"
echo "  export HF_HUB_ENABLE_HF_TRANSFER=1"