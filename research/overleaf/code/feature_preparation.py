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
