"""Prepare reproducible forest-grid/day examples without altering source files."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
from rasterio.windows import Window
from rasterio.features import geometry_mask
from shapely.geometry import shape
from shapely import intersects_xy

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data/prepared'
OUT.mkdir(parents=True, exist_ok=True)
BOUNDARY = ROOT / 'data/sources/geoBoundaries-NPL-ADM0.geojson'
LC = ROOT / 'data/study_grid/lc2011.tif'
DEM = ROOT / 'data/terrain/nepal_SRTMGL1.tif'


def grid_code(lat, lon):
    return np.floor((lat - 26) / .25 + .5).astype(int) * 37 + np.floor((lon - 80) / .25 + .5).astype(int)


def terrain_features(geometry):
    target = OUT / 'forest_grid_terrain.csv'
    if target.exists():
        return pd.read_csv(target)
    size = 21 * 37
    sums = {k: np.zeros(size, dtype=float) for k in ['forest_n', 'classified_n', 'elevation_sum',
                                                   'slope_sum', 'northness_sum', 'eastness_sum']}
    with rasterio.open(LC) as lc, rasterio.open(DEM) as dem:
        with WarpedVRT(dem, crs=lc.crs, transform=lc.transform, width=lc.width,
                       height=lc.height, resampling=Resampling.bilinear, dtype='float32') as aligned:
            lon = lc.transform.c + (np.arange(lc.width) + .5) * lc.transform.a
            for start in range(0, lc.height, 128):
                n = min(128, lc.height - start)
                window = Window(0, start, lc.width, n)
                classes = lc.read(1, window=window)
                inside = geometry_mask([geometry], (n, lc.width), lc.window_transform(window), invert=True)
                lat = lc.transform.f + (np.arange(start, start + n) + .5) * lc.transform.e
                codes = grid_code(lat[:, None], lon[None, :])
                classified = inside & (classes >= 1) & (classes <= 11)
                forest = inside & (classes == 4)
                sums['classified_n'] += np.bincount(codes[classified], minlength=size)
                if not forest.any():
                    continue
                # One-row halos prevent artificial edges between processing blocks.
                a = max(0, start - 1)
                b = min(lc.height, start + n + 1)
                z = aligned.read(1, window=Window(0, a, lc.width, b-a), masked=True).filled(np.nan)
                row_derivative, col_derivative = np.gradient(z)
                sl = slice(start-a, start-a+n)
                elevation = z[sl]
                # Local spherical metric approximation; never differentiate degrees as metres.
                dy = abs(lc.transform.e) * np.pi / 180 * 6371008.8
                dx = abs(lc.transform.a) * np.pi / 180 * 6371008.8 * np.cos(np.radians(lat[:, None]))
                north_gradient = -row_derivative[sl] / dy
                east_gradient = col_derivative[sl] / dx
                magnitude = np.hypot(north_gradient, east_gradient)
                slope = np.degrees(np.arctan(magnitude))
                northness = np.divide(-north_gradient, magnitude, out=np.zeros_like(magnitude), where=magnitude > 1e-9)
                eastness = np.divide(-east_gradient, magnitude, out=np.zeros_like(magnitude), where=magnitude > 1e-9)
                if not np.isfinite(elevation[forest]).all() or not np.isfinite(slope[forest]).all():
                    raise ValueError('Missing terrain at forest pixels')
                sums['forest_n'] += np.bincount(codes[forest], minlength=size)
                for name, values in [('elevation_sum', elevation), ('slope_sum', slope),
                                     ('northness_sum', northness), ('eastness_sum', eastness)]:
                    sums[name] += np.bincount(codes[forest], weights=values[forest], minlength=size)
                if start % 2048 == 0:
                    print(f'Terrain processing: row {start}/{lc.height}', flush=True)
    rows = []
    for code in np.flatnonzero(sums['forest_n']):
        iy, ix = divmod(int(code), 37)
        count = sums['forest_n'][code]
        rows.append(dict(grid_id=f'NP_{iy:02d}_{ix:02d}',
                         forest_fraction_2011=count/sums['classified_n'][code],
                         forest_pixels_2011=int(count),
                         elevation_mean_m=sums['elevation_sum'][code]/count,
                         slope_mean_degrees=sums['slope_sum'][code]/count,
                         aspect_northness_mean=sums['northness_sum'][code]/count,
                         aspect_eastness_mean=sums['eastness_sum'][code]/count))
    frame = pd.DataFrame(rows)
    frame.to_csv(target, index=False)
    return frame


def main():
    geometry = json.loads(BOUNDARY.read_text())['features'][0]['geometry']
    polygon = shape(geometry)
    if not polygon.is_valid:
        raise ValueError('Invalid boundary polygon')
    terrain = terrain_features(geometry)
    fire = pd.read_csv(ROOT / 'data/sources/Active_Fire_Incidents_in_Nepal.csv')
    audit = {'source_fire_rows': len(fire), 'weather_target_locations': 238,
             'study_definition': 'Forecast next-day recorded fire detection in fixed 2011 forest mask',
             'negative_definition': 'No recorded detection; absence and satellite observation coverage unverified'}
    # The source follows MODIS acquisition fields. UTC interpretation is an assumption
    # pending source metadata, and is recorded in the report.
    date = pd.to_datetime(fire['ACQ_DATE'], errors='coerce')
    hhmm = pd.to_numeric(fire['ACQ_TIME'], errors='coerce')
    hour = hhmm // 100
    minute = hhmm % 100
    valid_time = date.notna() & hhmm.notna() & (hhmm == np.floor(hhmm)) & hour.between(0,23) & minute.between(0,59)
    audit['invalid_acquisition_times'] = int((~valid_time).sum())
    fire = fire.loc[valid_time].copy()
    fire['timestamp_utc'] = (date.loc[valid_time] + pd.to_timedelta(hour.loc[valid_time], unit='h') +
                             pd.to_timedelta(minute.loc[valid_time], unit='m')).dt.tz_localize('UTC')
    fire['date'] = fire['timestamp_utc'].dt.tz_convert('Asia/Kathmandu').dt.tz_localize(None).dt.normalize()
    fire['lat_key'] = fire['LATITUDE'].round(5)
    fire['lon_key'] = fire['LONGITUDE'].round(5)
    keys = ['lat_key', 'lon_key', 'timestamp_utc', 'SATELLITE']
    fire = fire.sort_values('CONFIDENCE', ascending=False).drop_duplicates(keys).copy()
    audit['unique_detection_rows'] = len(fire)
    audit['duplicate_detection_rows_removed'] = audit['source_fire_rows'] - audit['invalid_acquisition_times'] - len(fire)
    inside = intersects_xy(polygon, fire['LONGITUDE'].to_numpy(), fire['LATITUDE'].to_numpy())
    audit['outside_nepal_rows'] = int((~inside).sum())
    fire = fire.loc[inside].copy()
    nationwide_dates = set(fire['date'])
    with rasterio.open(LC) as lc:
        classes = np.array([a[0] for a in lc.sample(zip(fire['LONGITUDE'], fire['LATITUDE']))])
    fire['landcover_2011'] = classes
    audit['within_nepal_nonforest_rows'] = int((classes != 4).sum())
    fire = fire.loc[classes == 4].copy()
    codes = grid_code(fire['LATITUDE'].to_numpy(), fire['LONGITUDE'].to_numpy())
    fire['grid_id'] = [f'NP_{c//37:02d}_{c%37:02d}' for c in codes]
    fire.to_csv(OUT / 'cleaned_forest_detections.csv', index=False)
    events = fire.groupby(['grid_id','date']).size().rename('detection_count').reset_index()
    audit['forest_detection_rows'] = len(fire)
    audit['source_rows_per_year'] = pd.read_csv(ROOT / 'data/sources/Active_Fire_Incidents_in_Nepal.csv')['ACQ_DATE'].str[:4].value_counts().sort_index().to_dict()
    frames = []
    paths = sorted((ROOT / 'data/open_meteo_daily').glob('*.csv'))
    for path in paths:
        weather = pd.read_csv(path, parse_dates=['date']).sort_values('date')
        if len(weather) != 1827 or weather['date'].duplicated().any():
            raise ValueError(f'Invalid dates: {path.name}')
        features = weather[['grid_id','date']].copy()
        for name in ['temperature_2m_mean','temperature_2m_max','relative_humidity_2m_mean',
                     'relative_humidity_2m_min','wind_speed_10m_mean','wind_speed_10m_max']:
            features[name+'_previous_day'] = weather[name].shift(1)
        for days in [1, 7, 30]:
            features[f'precipitation_previous_{days}d_mm'] = weather['precipitation_sum'].shift(1).rolling(days, min_periods=days).sum()
        features['temperature_mean_previous_7d'] = weather['temperature_2m_mean'].shift(1).rolling(7).mean()
        features['humidity_mean_previous_7d'] = weather['relative_humidity_2m_mean'].shift(1).rolling(7).mean()
        features['latitude'] = weather['requested_latitude']
        features['longitude'] = weather['requested_longitude']
        # Complete local days: exclude first/last acquisition days because the CSV
        # does not establish full-day monitoring at either endpoint.
        features = features[features['date'].between('2012-02-02','2016-06-18')].copy()
        features = features.merge(events, on=['grid_id','date'], how='left', validate='one_to_one')
        features['detection_count'] = features['detection_count'].fillna(0).astype(int)
        features['fire_detected'] = (features['detection_count'] > 0).astype(int)
        features['national_detection_recorded'] = features['date'].isin(nationwide_dates).astype(int)
        features['day_of_year_sin'] = np.sin(2*np.pi*features['date'].dt.dayofyear/365.25)
        features['day_of_year_cos'] = np.cos(2*np.pi*features['date'].dt.dayofyear/365.25)
        frames.append(features)
    data = pd.concat(frames, ignore_index=True).merge(terrain, on='grid_id', how='inner', validate='many_to_one')
    data['split'] = np.select([data['date'].dt.year <= 2014, data['date'].dt.year == 2015], ['train','validation'], default='test')
    predictor_columns = [c for c in data if c not in ['grid_id','date','detection_count','fire_detected','national_detection_recorded','split','forest_pixels_2011']]
    if data.duplicated(['grid_id','date']).any():
        raise ValueError('Duplicate grid/day rows')
    if data[predictor_columns].isna().any().any():
        raise ValueError('Missing predictors')
    data.to_csv(OUT / 'forest_fire_grid_daily.csv', index=False)
    (OUT/'feature_columns.json').write_text(json.dumps(predictor_columns, indent=2))
    audit.update(weather_complete_locations=len(paths), modelling_locations=data['grid_id'].nunique(),
                 daily_examples=len(data), positive_grid_days=int(data['fire_detected'].sum()),
                 provisional=len(paths) < 238,
                 date_range=[str(data['date'].min().date()),str(data['date'].max().date())],
                 utc_acquisition_time_assumed=True,
                 terrain_method='Bilinear elevation aligned to land-cover raster; central differences with local spherical metre spacing and row halos. Forest-pixel means within boundary.',
                 landcover_baseline='2011 forest class 4; fixed across all years',
                 predictor_count=len(predictor_columns),
                 splits=data.groupby('split').agg(rows=('fire_detected','size'),positive_days=('fire_detected','sum')).to_dict('index'))
    (OUT/'preparation_audit.json').write_text(json.dumps(audit,indent=2))
    print(json.dumps(audit,indent=2),flush=True)


if __name__ == '__main__':
    main()
