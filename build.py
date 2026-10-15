"""
Builds the global plant hardiness zone tile layer.

Hardiness zones are defined by the average annual extreme minimum temperature
(the coldest night of each year, averaged over 1991-2020), split into 10 degF
zones and 5 degF half zones, the same definition the USDA uses.

Sources:
  - NOAA CPC Global Unified Daily Temperature (0.5 deg, daily Tmin), for the
    yearly extreme minimum:
    https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.YYYY.nc
  - WorldClim 2.1 BIO6 (min temperature of coldest month, 5 arc-minutes), to
    add terrain detail the 0.5 deg grid can't resolve:
    https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_5m_bio.zip

Downscaling: the coarse offset (extreme min - BIO6) varies smoothly in space,
so it's computed at 0.5 deg, interpolated to 5', and added back onto BIO6.

Usage:
  pip install numpy pillow netCDF4 tifffile imagecodecs
  python download.py <data_dir>
  python build.py <data_dir> <out_dir> [zone_grid.npy]

  data_dir must contain cpc/tmin.1991.nc ... cpc/tmin.2020.nc and
  wc/wc2.1_5m_bio.zip, which download.py fetches. Tiles are written to out_dir/{z}/{x}/{y}.png; the
  optional third argument also saves the 5' zone grid for spot checks.
"""

import io
import math
import sys
import warnings
import zipfile
from pathlib import Path

import netCDF4
import numpy as np
import tifffile
from PIL import Image

YEARS = range(1991, 2021)  # keep in sync with YEARS in download.py
MIN_DAYS_PER_YEAR = 300  # skip station-gap years rather than trust a partial winter
MIN_YEARS = 20
SPIKE_C = 12
MAX_ZOOM = 6
TILE = 256

# Ocean cells are all-NaN, which numpy's nan reductions warn about on every call
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Half zones 1a..13b; keep in sync with HARDINESS_ZONE_COLORS in src/constants/hardiness.ts
COLORS = [
    '#d6d6ff', '#c4c4f2', '#eba6eb', '#dd8fe8', '#a66bff', '#5a75ed', '#73a1ff', '#5ec9e0',
    '#1eb51e', '#61c94a', '#8cc63f', '#b5d559', '#d4e36b', '#edda85', '#ead26c', '#e1b937',
    '#f5b04c', '#e8913a', '#e66e2a', '#e55a1e', '#e63d1b', '#d62a2a', '#b5182b', '#9c1036',
    '#7e0b3e', '#640a3d'
]


def extreme_min(cpc_dir: Path) -> np.ndarray:
    """Mean annual extreme minimum (degC) on the CPC grid, rows 90N->90S, cols 180W->180E."""
    yearly = []
    for year in YEARS:
        with netCDF4.Dataset(cpc_dir / f'tmin.{year}.nc') as ds:
            tmin = np.ma.filled(ds['tmin'][:].astype(np.float32), np.nan)
            lat = ds['lat'][:]
        days = np.sum(~np.isnan(tmin), axis=0)
        # CPC has occasional one-day station glitches (e.g. -25C over London on 1991-12-27);
        # a real cold snap lasts more than one night, so drop days far colder than both neighbours
        spike = np.fmin(tmin[:-2], tmin[2:]) - tmin[1:-1] > SPIKE_C
        tmin[1:-1][spike] = np.nan
        annual = np.nanmin(tmin, axis=0)
        annual[days < MIN_DAYS_PER_YEAR] = np.nan
        if lat[0] < lat[-1]:
            annual = annual[::-1]
        yearly.append(annual)
        print(f'  {year}: {np.sum(~np.isnan(annual))} cells')

    stack = np.stack(yearly)
    mean = np.nanmean(stack, axis=0)
    mean[np.sum(~np.isnan(stack), axis=0) < MIN_YEARS] = np.nan
    # CPC longitudes run 0..360; shift to -180..180
    return np.roll(mean, mean.shape[1] // 2, axis=1)


def worldclim_bio6(zip_path: Path) -> np.ndarray:
    """BIO6 (degC) at 5', rows 90N->90S, cols 180W->180E, NaN over ocean."""
    with zipfile.ZipFile(zip_path) as zf:
        name = next(n for n in zf.namelist() if n.endswith('_bio_6.tif'))
        data = tifffile.imread(io.BytesIO(zf.read(name))).astype(np.float32)
    data[data < -1000] = np.nan
    return data


def fill_nan(grid: np.ndarray, iterations: int = 200) -> np.ndarray:
    """Spread valid values into NaN cells by repeated neighbour averaging (wraps east-west)."""
    out = grid.copy()
    for _ in range(iterations):
        missing = np.isnan(out)
        if not missing.any():
            break
        padded = np.pad(out, ((1, 1), (0, 0)), constant_values=np.nan)
        neighbours = np.stack([
            padded[:-2], padded[2:],
            np.roll(out, 1, axis=1), np.roll(out, -1, axis=1)
        ])
        avg = np.nanmean(neighbours, axis=0)
        out[missing] = avg[missing]
    return out


def upsample(grid: np.ndarray, factor: int) -> np.ndarray:
    """Bilinear interpolation between cell centres, wrapping east-west."""
    rows, cols = grid.shape
    y = (np.arange(rows * factor) + 0.5) / factor - 0.5
    x = (np.arange(cols * factor) + 0.5) / factor - 0.5
    y = np.clip(y, 0, rows - 1)
    y0 = np.floor(y).astype(int)
    y1 = np.minimum(y0 + 1, rows - 1)
    fy = (y - y0)[:, None]
    x0 = np.floor(x).astype(int)
    fx = (x - x0)[None, :]
    x0 %= cols
    x1 = (x0 + 1) % cols
    top = grid[y0][:, x0] * (1 - fx) + grid[y0][:, x1] * fx
    bottom = grid[y1][:, x0] * (1 - fx) + grid[y1][:, x1] * fx
    return top * (1 - fy) + bottom * fy


def classify(temp_c: np.ndarray) -> np.ndarray:
    """Half-zone index: 0 = no data, 1 = 1a (-60..-55 degF) ... 26 = 13b (65..70 degF)."""
    temp_f = temp_c * 9 / 5 + 32
    with np.errstate(invalid='ignore'):
        zones = np.floor((temp_f + 60) / 5).astype(np.int16) + 1
    zones = np.clip(zones, 1, len(COLORS))
    zones[np.isnan(temp_c)] = 0
    return zones.astype(np.uint8)


def palette() -> list[int]:
    flat = [0, 0, 0]
    for hex_color in COLORS:
        flat += [int(hex_color[i:i + 2], 16) for i in (1, 3, 5)]
    return flat + [0] * (768 - len(flat))


def write_tiles(zones: np.ndarray, out_dir: Path) -> None:
    rows, cols = zones.shape
    pal = palette()
    written = 0
    for z in range(MAX_ZOOM + 1):
        n = 2 ** z
        world = TILE * n
        for ty in range(n):
            py = ty * TILE + np.arange(TILE) + 0.5
            lat = np.degrees(np.arctan(np.sinh(math.pi * (1 - 2 * py / world))))
            r = np.clip(((90 - lat) / 180 * rows).astype(int), 0, rows - 1)
            for tx in range(n):
                px = tx * TILE + np.arange(TILE) + 0.5
                lon = px / world * 360 - 180
                c = np.clip(((lon + 180) / 360 * cols).astype(int), 0, cols - 1)
                tile = zones[r][:, c]
                if not tile.any():
                    continue
                img = Image.fromarray(tile, mode='P')
                img.putpalette(pal)
                path = out_dir / str(z) / str(tx) / f'{ty}.png'
                path.parent.mkdir(parents=True, exist_ok=True)
                img.save(path, optimize=True, transparency=0)
                written += 1
        print(f'  z{z} done ({written} tiles so far)')


def main() -> None:
    data_dir, out_dir = Path(sys.argv[1]), Path(sys.argv[2])

    print('Extreme minimum from CPC...')
    ext = extreme_min(data_dir / 'cpc')

    print('WorldClim BIO6...')
    bio6 = worldclim_bio6(data_dir / 'wc' / 'wc2.1_5m_bio.zip')
    factor = bio6.shape[0] // ext.shape[0]

    print('Downscaling...')
    blocks = bio6.reshape(ext.shape[0], factor, ext.shape[1], factor)
    bio6_coarse = np.nanmean(blocks, axis=(1, 3))
    offset = fill_nan(ext - bio6_coarse)
    fine = bio6 + upsample(offset, factor)

    zones = classify(fine)
    if len(sys.argv) > 3:
        np.save(sys.argv[3], zones)

    print('Writing tiles...')
    write_tiles(zones, out_dir)


if __name__ == '__main__':
    main()
