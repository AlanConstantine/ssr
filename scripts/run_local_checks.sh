#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON:-python}"
if [[ -n "${CONDA_PREFIX:-}" && -x "${CONDA_PREFIX}/bin/python" ]]; then
  PYTHON_BIN="${CONDA_PREFIX}/bin/python"
fi

"${PYTHON_BIN}" -m py_compile *.py scripts/*.py
"${PYTHON_BIN}" -m pytest tests/test_smoke.py
