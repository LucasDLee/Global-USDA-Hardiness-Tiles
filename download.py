"""
Downloads the source data build.py needs

Usage:
  python download.py <data_dir>

Creates data_dir/cpc/tmin.1991.nc ... tmin.2020.nc and data_dir/wc/wc2.1_5m_bio.zip.
Safe to re-run: files that are already complete are skipped, and interrupted
downloads start over from a .part file instead of leaving a truncated file behind.
"""

import sys
import time
import urllib.request
from pathlib import Path

CPC_URL = 'https://downloads.psl.noaa.gov/Datasets/cpc_global_temp/tmin.{year}.nc'
WORLDCLIM_URL = 'https://geodata.ucdavis.edu/climate/worldclim/2_1/base/wc2.1_5m_bio.zip'
YEARS = range(1991, 2021)  # keep in sync with YEARS in build.py
RETRIES = 3
CHUNK = 1024 * 1024


def remote_size(url: str) -> int | None:
    request = urllib.request.Request(url, method='HEAD')
    with urllib.request.urlopen(request, timeout=60) as res:
        length = res.headers.get('Content-Length')
    return int(length) if length else None


def download(url: str, dest: Path) -> None:
    size = remote_size(url)
    if dest.exists() and size is not None and dest.stat().st_size == size:
        print(f'  {dest.name}: already downloaded')
        return

    part = dest.with_name(dest.name + '.part')
    for attempt in range(1, RETRIES + 1):
        try:
            with urllib.request.urlopen(url, timeout=120) as res, open(part, 'wb') as out:
                done = 0
                while chunk := res.read(CHUNK):
                    out.write(chunk)
                    done += len(chunk)
                    if size:
                        print(f'\r  {dest.name}: {done * 100 // size}%', end='', flush=True)
            if size is not None and part.stat().st_size != size:
                raise OSError(f'expected {size} bytes, got {part.stat().st_size}')
            part.replace(dest)
            print(f'\r  {dest.name}: done ')
            return
        except OSError as err:
            print(f'\r  {dest.name}: attempt {attempt} failed ({err})')
            if attempt == RETRIES:
                raise
            time.sleep(5 * attempt)


def main() -> None:
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    data_dir = Path(sys.argv[1])
    (data_dir / 'cpc').mkdir(parents=True, exist_ok=True)
    (data_dir / 'wc').mkdir(parents=True, exist_ok=True)

    print('WorldClim BIO variables...')
    download(WORLDCLIM_URL, data_dir / 'wc' / 'wc2.1_5m_bio.zip')

    print('NOAA CPC daily minimum temperature...')
    for year in YEARS:
        download(CPC_URL.format(year=year), data_dir / 'cpc' / f'tmin.{year}.nc')

    print(f'All source data is in {data_dir}')


if __name__ == '__main__':
    main()
