#!/bin/bash
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo "First follow the setup steps in README.md to create .venv and install requirements."
  read -r -p "Press Enter to close."
  exit 1
fi
exec .venv/bin/python -m streamlit run application/app.py --server.address 127.0.0.1 --theme.base light --browser.gatherUsageStats false
