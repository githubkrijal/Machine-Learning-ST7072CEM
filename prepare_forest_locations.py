"""Select ERA5-sized weather cells containing forest in the 2011 Nepal map."""
import csv
import json
import zipfile
from pathlib import Path
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'data' / 'study_grid'
OUT.mkdir(parents=True, exist_ok=True)
source = Path('/Users/krijalshrestha/Downloads/Land cover of Nepal 2011.zip')
local = OUT / 'lc2011.tif'
if not local.exists():
    with zipfile.ZipFile(source) as archive:
        local.write_bytes(archive.read('data/lc2011.tif'))
forest = np.zeros(21 * 37, dtype=np.int64)
valid = forest.copy()
with rasterio.open(local) as src:
    assert src.crs.to_epsg() == 4326
    longitudes = src.transform.c + (np.arange(src.width) + .5) * src.transform.a
    col_cells = np.floor((longitudes - 80) / .25 + .5).astype(np.int32)
    for start in range(0, src.height, 256):
        count = min(256, src.height - start)
        a = src.read(1, window=Window(0, start, src.width, count))
        latitudes = src.transform.f + (np.arange(start, start + count) + .5) * src.transform.e
        row_cells = np.floor((latitudes - 26) / .25 + .5).astype(np.int32)
        codes = row_cells[:, None] * 37 + col_cells[None, :]
        forest += np.bincount(codes[a == 4], minlength=len(forest))
        valid += np.bincount(codes[(a >= 1) & (a <= 11)], minlength=len(valid))
rows = []
for code in np.flatnonzero(forest):
    iy, ix = divmod(int(code), 37)
    rows.append({'grid_id': f'NP_{iy:02d}_{ix:02d}', 'latitude': 26 + iy * .25,
                 'longitude': 80 + ix * .25, 'forest_pixels_2011': int(forest[code]),
                 'classified_pixels_2011': int(valid[code]),
                 'forest_fraction_of_classified_pixels': round(float(forest[code]/valid[code]),6)})
with (OUT / 'nepal_forest_weather_locations.csv').open('w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
(OUT / 'method.json').write_text(json.dumps({
    'source': str(source), 'baseline_year': 2011, 'forest_class': 4,
    'grid_spacing_degrees': .25, 'selection': 'Any forest pixel in the cell',
    'coordinates': 'Nominal ERA5 weather cell centres, not necessarily forest points or inside Nepal',
    'purpose': 'Weather lookup; restrict training examples to actual forest pixels',
    'forest_fraction': 'Pixel count fraction among classified source pixels, not exact geodesic area',
    'limitations': 'Fixed baseline omits later forest gains and retains later forest losses; 2012 land cover unavailable',
    'location_count': len(rows),
}, indent=2))
print(f'Prepared {len(rows)} weather locations, covering {int(forest.sum()):,} forest pixels.')
print(OUT / 'nepal_forest_weather_locations.csv')
