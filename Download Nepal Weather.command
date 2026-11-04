#!/bin/zsh
cd -- "${0:A:h}" || exit 1
if [[ ! -x .weather-venv/bin/python ]]; then
    python3 -m venv .weather-venv || exit 1
fi
if ! .weather-venv/bin/python -c 'import cdsapi' >/dev/null 2>&1; then
    .weather-venv/bin/python -m pip install 'cdsapi>=0.7.7' || exit 1
fi
.weather-venv/bin/python download_weather.py
read '?Press Enter to close this window.'
