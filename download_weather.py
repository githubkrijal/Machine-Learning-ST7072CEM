#!/usr/bin/env python3
"""Download monthly Nepal ERA5-Land weather; credentials stay out of files."""
import argparse
import calendar
import getpass
import json
import os
from pathlib import Path
import zipfile

VARIABLES = [
    '2m_temperature', '2m_dewpoint_temperature', 'total_precipitation',
    '10m_u_component_of_wind', '10m_v_component_of_wind',
]
ROOT = Path(__file__).resolve().parent


def request_for(year, month):
    return {
        'variable': VARIABLES,
        'year': str(year), 'month': f'{month:02d}',
        'day': [f'{d:02d}' for d in range(1, calendar.monthrange(year, month)[1] + 1)],
        'time': [f'{h:02d}:00' for h in range(24)],
        'area': [31, 80, 26, 89],  # North, West, South, East
        'data_format': 'netcdf', 'download_format': 'unarchived',
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = [(y, m) for y in range(2012, 2017) for m in range(1, 13)]
    if args.dry_run:
        print(json.dumps({'months': len(jobs), 'first': request_for(*jobs[0]),
                          'last': request_for(*jobs[-1])}, indent=2))
        return
    import cdsapi
    print('Nepal weather: 2012–2016, all hours, five variables.')
    print('First accept the dataset licence at:')
    print('https://cds.climate.copernicus.eu/datasets/reanalysis-era5-land?tab=download')
    print('Find your personal access token at https://cds.climate.copernicus.eu/how-to-api')
    token = os.environ.get('CDSAPI_KEY') or getpass.getpass('Paste token (hidden; not saved): ').strip()
    if not token:
        raise SystemExit('No token supplied. Nothing downloaded.')
    client = cdsapi.Client(url='https://cds.climate.copernicus.eu/api', key=token,
                           quiet=True, debug=False)
    out = ROOT / 'data' / 'era5_land'
    out.mkdir(parents=True, exist_ok=True)
    failures = []
    for index, (year, month) in enumerate(jobs, 1):
        stem = f'nepal_era5_land_{year}_{month:02d}'
        marker = out / f'{stem}.json'
        if marker.exists():
            saved = json.loads(marker.read_text())
            if all((out / f['name']).is_file() and (out / f['name']).stat().st_size == f['bytes']
                   for f in saved.get('files', [])) and saved.get('files'):
                print(f'[{index}/60] {year}-{month:02d}: already downloaded', flush=True)
                continue
        temp = out / f'{stem}.part'
        print(f'[{index}/60] Downloading {year}-{month:02d}; waiting for CDS if queued…', flush=True)
        try:
            request = request_for(year, month)
            client.retrieve('reanalysis-era5-land', request).download(str(temp))
            # CDS may package separate accumulated/instantaneous NetCDFs in a ZIP.
            with temp.open('rb') as stream:
                signature = stream.read(8)
            extension = '.zip' if zipfile.is_zipfile(temp) else '.nc'
            if extension == '.zip':
                with zipfile.ZipFile(temp) as archive:
                    bad = archive.testzip()
                    if bad:
                        raise ValueError('Downloaded ZIP failed integrity check')
            elif not (signature.startswith(b'CDF') or signature == b'\x89HDF\r\n\x1a\n'):
                raise ValueError('Response is not a NetCDF or ZIP file')
            if temp.stat().st_size == 0:
                raise ValueError('Empty download')
            target = out / (stem + extension)
            temp.replace(target)
            marker.write_text(json.dumps({'dataset': 'reanalysis-era5-land', 'request': request,
                'files': [{'name': target.name, 'bytes': target.stat().st_size}]}, indent=2))
            print(f'Saved {target.name}', flush=True)
        except KeyboardInterrupt:
            print('\nStopped. Run again to resume completed months.')
            return
        except Exception as error:
            # Do not print exception messages that might contain credentials.
            print(f'Failed {year}-{month:02d} ({type(error).__name__}). Check account licence, token and connection.', flush=True)
            failures.append(f'{year}-{month:02d}')
            if index == 1:
                raise SystemExit('First request failed; stopping so you can check account setup.')
    if failures:
        raise SystemExit('Rerun to retry failed months: ' + ', '.join(failures))
    print(f'All 60 months downloaded to {out}')


if __name__ == '__main__':
    main()
