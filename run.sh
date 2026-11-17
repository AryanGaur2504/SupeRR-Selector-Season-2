#!/usr/bin/env bash
cd "$(dirname "$0")"
PY=$(command -v python3 || command -v python) || { echo "Python 3.10+ not found"; exit 1; }
[ -d .venv ] || $PY -m venv .venv || exit 1
.venv/bin/python -m pip install --disable-pip-version-check -q -r requirements.txt || exit 1
(sleep 4; (open http://localhost:8501 || xdg-open http://localhost:8501) >/dev/null 2>&1) &
.venv/bin/python -m streamlit run app.py --server.port 8501
