"""Download daily ERA5 weather from Open-Meteo for the prepared forest grid."""
import argparse
import csv
import datetime as dt
import json
import math
from pathlib import Path
import time
import requests

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'open_meteo_daily'
VARIABLES = ['temperature_2m_mean', 'temperature_2m_max', 'precipitation_sum',
             'relative_humidity_2m_mean', 'relative_humidity_2m_min',
             'wind_speed_10m_mean', 'wind_speed_10m_max']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=0, help='Maximum new locations; 0 means all')
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    with (ROOT / 'data/study_grid/nepal_forest_weather_locations.csv').open() as f:
        locations = list(csv.DictReader(f))
    expected = [(dt.date(2012, 1, 1) + dt.timedelta(days=i)).isoformat() for i in range(1827)]
    completed = 0
    for i, loc in enumerate(locations, 1):
        target = OUT / (loc['grid_id'] + '.csv')
        metadata = OUT / (loc['grid_id'] + '.json')
        if target.exists() and metadata.exists():
            continue
        params = dict(latitude=loc['latitude'], longitude=loc['longitude'],
                      start_date='2012-01-01', end_date='2016-12-31',
                      daily=','.join(VARIABLES), models='era5', timezone='Asia/Kathmandu',
                      temperature_unit='celsius', wind_speed_unit='ms', precipitation_unit='mm',
                      cell_selection='nearest', elevation='nan')
        print(f'[{i}/{len(locations)}] Requesting {loc["grid_id"]}', flush=True)
        while True:
            try:
                response = requests.get('https://archive-api.open-meteo.com/v1/archive', params=params, timeout=180)
                if response.status_code >= 500:
                    print('Temporary server error; retrying in 60 seconds.', flush=True)
                    time.sleep(60)
                    continue
                if response.ok:
                    payload = response.json()
            except (requests.RequestException, ValueError):
                print('Temporary connection or invalid response; retrying in 60 seconds.', flush=True)
                time.sleep(60)
                continue
            if response.status_code != 429:
                break
            try:
                reason = response.json().get('reason', 'Usage limit reached')
            except ValueError:
                reason = 'Usage limit reached'
            wait_seconds = 86400 if 'daily' in reason.lower() else 3600
            if 'minutely' in reason.lower() or 'minute limit' in reason.lower():
                wait_seconds = 60
            retry_after = response.headers.get('Retry-After', '')
            if retry_after.isdigit():
                wait_seconds = max(wait_seconds, int(retry_after))
            print(f'{reason} Retrying in {wait_seconds // 60} minutes; completed files retained.', flush=True)
            time.sleep(wait_seconds)
        if not response.ok:
            raise SystemExit(f'Open-Meteo HTTP {response.status_code}: {response.text[:500]}')
        daily = payload['daily']
        if daily['time'] != expected:
            raise SystemExit('Unexpected dates; response not saved.')
        for variable in VARIABLES:
            values = daily[variable]
            if len(values) != len(expected) or any(v is None or not math.isfinite(v) for v in values):
                raise SystemExit(f'Incomplete values for {variable}; response not saved.')
        fields = ['grid_id', 'requested_latitude', 'requested_longitude', 'date'] + VARIABLES
        temp = target.with_suffix('.part')
        with temp.open('w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(fields)
            for j, date in enumerate(expected):
                writer.writerow([loc['grid_id'], loc['latitude'], loc['longitude'], date] +
                                [daily[v][j] for v in VARIABLES])
        temp.replace(target)
        payload.pop('daily')
        metadata.write_text(json.dumps({'request': params, 'response_metadata': payload,
                                       'source': 'https://open-meteo.com/en/docs/historical-weather-api',
                                       'note': 'ERA5 nearest cell; elevation adjustment disabled'}, indent=2))
        completed += 1
        print(f'Saved {loc["grid_id"]}: 1,827 complete days, seven variables.', flush=True)
        if args.limit and completed >= args.limit:
            break
        time.sleep(12)  # Pace long requests; stop, rather than bypass, usage limits.
    print(f'Complete location files: {len(list(OUT.glob("*.csv")))}/{len(locations)}', flush=True)


if __name__ == '__main__':
    main()
