"""
gfs_fetcher.py
==============
Smart GFS fetcher — auto-discovers latest available cycle on NOMADS.
Fetches CAPE, T, q, u, v at 500/700/850hPa for Bengaluru Airport.
Saves to data/gfs_realtime_43295.csv (single-row latest values).

Usage:
  python gfs_fetcher.py
  python gfs_fetcher.py --slot 2
"""

import requests
import tempfile
import os
import re
import argparse
import pandas as pd
import numpy as np
import cfgrib
import warnings
from datetime import datetime, timedelta, timezone
from pathlib import Path

warnings.filterwarnings('ignore')

LAT = 12.97
LON = 77.58
OUT_DIR = Path('data')
OUT_DIR.mkdir(exist_ok=True)
OUT_FILE      = OUT_DIR / 'gfs_realtime_43295.csv'
OUT_HIST_FILE = OUT_DIR / 'gfs_history_43295.json'
KEEP_CYCLES   = 6

NOMADS_BASE = 'https://nomads.ncep.noaa.gov'


def get_latest_available_cycle():
    """Find the most recent GFS cycle available on NOMADS."""
    try:
        r = requests.get(f'{NOMADS_BASE}/pub/data/nccf/com/gfs/prod/', timeout=15)
        dates = sorted(set(re.findall(r'gfs\.(\d{8})', r.text)))
        if not dates:
            return None, None
        latest_date = dates[-1]

        # Try cycles in reverse order: 18Z, 12Z, 06Z, 00Z
        for cycle_hour in [18, 12, 6, 0]:
            url = (
                f'{NOMADS_BASE}/cgi-bin/filter_gfs_0p25.pl'
                f'?dir=%2Fgfs.{latest_date}%2F{cycle_hour:02d}%2Fatmos'
                f'&file=gfs.t{cycle_hour:02d}z.pgrb2.0p25.f012'
                f'&var_TMP=on&lev_850_mb=on'
                f'&subregion=&toplat=13.25&leftlon=77.25&rightlon=77.75&bottomlat=12.75'
            )
            r2 = requests.get(url, timeout=15)
            if r2.status_code == 200 and len(r2.content) > 100:
                print(f'  Latest available: {latest_date} {cycle_hour:02d}Z')
                return latest_date, cycle_hour

        # Try previous date if today's cycles aren't ready
        if len(dates) >= 2:
            prev_date = dates[-2]
            for cycle_hour in [18, 12, 6, 0]:
                url = (
                    f'{NOMADS_BASE}/cgi-bin/filter_gfs_0p25.pl'
                    f'?dir=%2Fgfs.{prev_date}%2F{cycle_hour:02d}%2Fatmos'
                    f'&file=gfs.t{cycle_hour:02d}z.pgrb2.0p25.f012'
                    f'&var_TMP=on&lev_850_mb=on'
                    f'&subregion=&toplat=13.25&leftlon=77.25&rightlon=77.75&bottomlat=12.75'
                )
                r2 = requests.get(url, timeout=15)
                if r2.status_code == 200 and len(r2.content) > 100:
                    print(f'  Latest available: {prev_date} {cycle_hour:02d}Z')
                    return prev_date, cycle_hour
        return None, None
    except Exception as e:
        print(f'  Error finding latest cycle: {e}')
        return None, None


def fetch_grib_data(date_str, cycle_hour):
    """Fetch all required variables in one GRIB request."""
    url = (
        f'{NOMADS_BASE}/cgi-bin/filter_gfs_0p25.pl'
        f'?dir=%2Fgfs.{date_str}%2F{cycle_hour:02d}%2Fatmos'
        f'&file=gfs.t{cycle_hour:02d}z.pgrb2.0p25.f012'
        f'&var_TMP=on&var_RH=on&var_UGRD=on&var_VGRD=on'
        f'&var_CAPE=on&var_CIN=on&var_SPFH=on&var_PRES=on'
        f'&lev_2_m_above_ground=on'
        f'&lev_10_m_above_ground=on'
        f'&lev_surface=on'
        f'&lev_500_mb=on&lev_700_mb=on&lev_850_mb=on'
        f'&lev_255-0_mb_above_ground=on'
        f'&lev_90-0_mb_above_ground=on'
        f'&subregion=&toplat=13.25&leftlon=77.25&rightlon=77.75&bottomlat=12.75'
    )

    print(f'  Fetching from NOMADS...')
    r = requests.get(url, timeout=120, stream=True)
    if r.status_code != 200:
        raise RuntimeError(f'HTTP {r.status_code}')

    with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as f:
        for chunk in r.iter_content(chunk_size=1024*1024):
            f.write(chunk)
        tmp = f.name

    size_kb = os.path.getsize(tmp) / 1024
    print(f'  Downloaded: {size_kb:.1f} KB')
    return tmp


def parse_grib(grib_path):
    """Extract Bengaluru point values from GRIB file."""
    row = {}
    try:
        datasets = cfgrib.open_datasets(grib_path, errors='ignore')
    except Exception as e:
        raise RuntimeError(f'cfgrib error: {e}')

    def nearest(ds, var, level=None, level_dim=None):
        try:
            if level is not None and level_dim is not None:
                ds = ds.sel({level_dim: level})
            pt = ds.sel(latitude=LAT, longitude=LON, method='nearest')
            v = float(pt[var].values)
            return v if np.isfinite(v) else np.nan
        except Exception:
            return np.nan

    for ds in datasets:
        dims = list(ds.dims)
        dvars = list(ds.data_vars)

        # Pressure level data
        if 'isobaricInhPa' in dims:
            for lev in [500, 700, 850]:
                for v in dvars:
                    vl = v.lower()
                    if vl in ('t', 'tmp') and f'ERA5_t_{lev}hPa' not in row:
                        val = nearest(ds, v, lev, 'isobaricInhPa')
                        if not np.isnan(val):
                            row[f'ERA5_t_{lev}hPa'] = val
                    elif vl in ('q', 'spfh', 'r', 'rh') and f'ERA5_q_{lev}hPa' not in row:
                        val = nearest(ds, v, lev, 'isobaricInhPa')
                        if not np.isnan(val):
                            # Convert RH to specific humidity if needed
                            if vl in ('r', 'rh') and val > 2:
                                val = val / 100.0 * 0.02  # rough conversion
                            row[f'ERA5_q_{lev}hPa'] = val
                    elif vl in ('u', 'ugrd') and f'ERA5_u_{lev}hPa' not in row:
                        val = nearest(ds, v, lev, 'isobaricInhPa')
                        if not np.isnan(val):
                            row[f'ERA5_u_{lev}hPa'] = val
                    elif vl in ('v', 'vgrd') and f'ERA5_v_{lev}hPa' not in row:
                        val = nearest(ds, v, lev, 'isobaricInhPa')
                        if not np.isnan(val):
                            row[f'ERA5_v_{lev}hPa'] = val

        # Surface/near-surface
        elif 'pressureFromGroundLayer' not in dims:
            for v in dvars:
                vl = v.lower()
                if vl in ('t2m', 'tmp') and 'ERA5_T2M' not in row:
                    val = nearest(ds, v)
                    if not np.isnan(val):
                        row['ERA5_T2M'] = val
                elif vl in ('d2m',) and 'ERA5_D2M' not in row:
                    val = nearest(ds, v)
                    if not np.isnan(val):
                        row['ERA5_D2M'] = val
                elif vl in ('u10', 'ugrd') and 'ERA5_U10' not in row:
                    val = nearest(ds, v)
                    if not np.isnan(val):
                        row['ERA5_U10'] = val
                elif vl in ('v10', 'vgrd') and 'ERA5_V10' not in row:
                    val = nearest(ds, v)
                    if not np.isnan(val):
                        row['ERA5_V10'] = val
                elif vl in ('cape',) and 'ERA5_CAPE' not in row:
                    val = nearest(ds, v)
                    if not np.isnan(val):
                        row['ERA5_CAPE'] = val
                elif vl in ('sp', 'pres') and 'ERA5_SP' not in row:
                    val = nearest(ds, v)
                    if not np.isnan(val):
                        row['ERA5_SP'] = val

        # CAPE from pressureFromGroundLayer
        elif 'pressureFromGroundLayer' in dims:
            if 'cape' in dvars and 'ERA5_CAPE' not in row:
                try:
                    pt = ds.sel(latitude=LAT, longitude=LON, method='nearest')
                    cape_vals = pt['cape'].values
                    max_cape = float(np.nanmax(cape_vals))
                    if max_cape > 0:
                        row['ERA5_CAPE'] = max_cape
                except Exception:
                    pass

    return row


def compute_stability_indices(row):
    """Compute K-Index, Lifted Index, Totals-Totals from GFS T/q profiles."""
    try:
        T850 = row.get('ERA5_t_850hPa', np.nan) - 273.15  # K to C
        T700 = row.get('ERA5_t_700hPa', np.nan) - 273.15
        T500 = row.get('ERA5_t_500hPa', np.nan) - 273.15
        q850 = row.get('ERA5_q_850hPa', np.nan)
        q700 = row.get('ERA5_q_700hPa', np.nan)

        # Dewpoint from specific humidity (approximate)
        def Td(q, p_hPa):
            if np.isnan(q) or q <= 0:
                return np.nan
            e = q * p_hPa / (0.622 + q)
            return 243.5 * np.log(e / 6.112) / (17.67 - np.log(e / 6.112))

        Td850 = Td(q850, 850)
        Td700 = Td(q700, 700)

        if not any(np.isnan(x) for x in [T850, T700, T500, Td850, Td700]):
            row['K_INDEX'] = (T850 - T500) + Td850 - (T700 - Td700)
            row['TOTALS_TOTALS'] = (T850 + Td850) - (2 * T500)

            # Lifted Index using moist adiabatic approximation
            # LCL temperature from surface parcel (850hPa as proxy for surface)
            # Moist adiabatic lapse rate ~6 C/km, pressure scale ~8km/decade
            # Simple Bolton (1980) approximation
            try:
                import metpy.calc as mpcalc
                from metpy.units import units
                p   = [850, 700, 500] * units.hPa
                T_k = [(T850+273.15), (T700+273.15), (T500+273.15)] * units.kelvin
                Td_k= [(Td850+273.15), (Td700+273.15), (Td700+273.15)] * units.kelvin
                li  = mpcalc.lifted_index(p, T_k, Td_k)
                row['LIFTED_INDEX'] = float(li.magnitude)
            except Exception:
                # Simple empirical LI approximation (George 1960)
                # LI = T500 - Tp500 where Tp500 estimated from surface dewpoint
                # Empirical: LI ≈ 2*(T500+20) - (Td850+T850)  (simplified)
                # More reliable: use Showalter-style from 850hPa
                # Showalter Index = T500 - T_parcel lifted from 850hPa
                # Moist adiabatic lapse rate ~5.5 C/km, 850->500hPa ~ 3.5 km
                spread = T850 - Td850  # dewpoint depression
                # Parcel cools at dry adiabatic to LCL, then moist adiabatic
                # LCL height approx: (T850 - Td850) / 8 km per C (rule of thumb)
                lcl_C_above_850 = spread / 8.0  # km above 850hPa level
                # Moist adiabatic from LCL to 500hPa (~3.5km total, minus LCL height)
                moist_distance = max(0, 3.5 - lcl_C_above_850)
                dry_distance = min(3.5, lcl_C_above_850)
                T_parcel_500 = T850 - (dry_distance * 9.8) - (moist_distance * 5.5)
                row['LIFTED_INDEX'] = round(T500 - T_parcel_500, 2)

            row['CAPE'] = row.get('ERA5_CAPE', 0.0)
            row['PRECIP_WATER'] = float(
                q850 * 850 * 100 / 9.81 + q700 * 150 * 100 / 9.81
            ) if not np.isnan(q850) else np.nan

            print(f'  K-Index:       {row["K_INDEX"]:.2f}')
            print(f'  Totals-Totals: {row["TOTALS_TOTALS"]:.2f}')
            print(f'  Lifted Index:  {row["LIFTED_INDEX"]:.2f}')
            print(f'  CAPE:          {row["CAPE"]:.2f}')
        else:
            print('  Could not compute stability indices — missing T/q profiles')

    except Exception as e:
        print(f'  Stability index error: {e}')

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--slot', type=int, choices=[0, 1, 2, 3])
    parser.add_argument('--date', type=str)
    args = parser.parse_args()

    print('=' * 60)
    print('  gfs_fetcher.py — Smart GFS Real-Time Fetcher')
    print('=' * 60)

    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    date_str = args.date or now_ist.strftime('%Y-%m-%d')
    print(f'  IST date: {date_str}')

    # Auto-discover latest available cycle
    print('\n  Finding latest available GFS cycle...')
    avail_date, avail_cycle = get_latest_available_cycle()

    if avail_date is None:
        print('  ERROR: No GFS cycle available on NOMADS')
        return 1

    # Fetch and parse — today (f012)
    try:
        grib_path = fetch_grib_data(avail_date, avail_cycle)
        row = parse_grib(grib_path)
        os.unlink(grib_path)
    except Exception as e:
        print(f'  ERROR: {e}')
        return 1

    # Compute stability indices
    row = compute_stability_indices(row)

    # Add metadata
    row['date'] = date_str
    row['gfs_cycle'] = f'{avail_date} {avail_cycle:02d}Z'
    row['fetched_at'] = now_ist.strftime('%Y-%m-%d %H:%M IST')

    # Fetch tomorrow (f024) and day after (f036)
    multiday = []
    for fhour, day_offset, day_label in [(24, 1, 'tomorrow'), (48, 2, 'day_after')]:
        try:
            print(f'\n  Fetching {day_label} (f{fhour:03d})...')
            url = (
                f'https://nomads.ncep.noaa.gov/cgi-bin/filter_gfs_0p25.pl'
                f'?dir=%2Fgfs.{avail_date}%2F{avail_cycle:02d}%2Fatmos'
                f'&file=gfs.t{avail_cycle:02d}z.pgrb2.0p25.f{fhour:03d}'
                f'&var_TMP=on&var_RH=on&var_UGRD=on&var_VGRD=on'
                f'&var_CAPE=on&var_CIN=on&var_SPFH=on&var_PRES=on'
                f'&lev_2_m_above_ground=on&lev_10_m_above_ground=on'
                f'&lev_surface=on&lev_500_mb=on&lev_700_mb=on&lev_850_mb=on'
                f'&lev_255-0_mb_above_ground=on&lev_90-0_mb_above_ground=on'
                f'&subregion=&toplat=13.25&leftlon=77.25&rightlon=77.75&bottomlat=12.75'
            )
            r = requests.get(url, timeout=120, stream=True)
            if r.status_code == 200 and len(r.content) > 500:
                with tempfile.NamedTemporaryFile(suffix='.grib2', delete=False) as f:
                    f.write(r.content)
                    tmp = f.name
                day_row = parse_grib(tmp)
                os.unlink(tmp)
                day_row = compute_stability_indices(day_row)
                future_date = (now_ist + timedelta(days=day_offset)).strftime('%Y-%m-%d')
                day_row['date'] = future_date
                day_row['day_label'] = day_label
                day_row['fhour'] = fhour
                day_row['gfs_cycle'] = f'{avail_date} {avail_cycle:02d}Z f{fhour:03d}'
                multiday.append(day_row)
                print(f'  ✓ {day_label}: CAPE={day_row.get("CAPE",0):.0f} K={day_row.get("K_INDEX",0):.1f} LI={day_row.get("LIFTED_INDEX",0):.2f}')
            else:
                print(f'  ⚠ {day_label}: HTTP {r.status_code} or empty')
        except Exception as e:
            print(f'  ⚠ {day_label} failed: {e}')

    # Save multiday forecast
    multiday_path = OUT_DIR / 'gfs_multiday_43295.json'
    import json as _json2
    with open(multiday_path, 'w') as f:
        _json2.dump(multiday, f, indent=2, default=str)
    print(f'\n  Saved → {multiday_path} ({len(multiday)} days)')

    # Save latest
    df = pd.DataFrame([row])
    df.to_csv(OUT_FILE, index=False)
    print(f'\n  Saved → {OUT_FILE}')
    print(f'  Variables: {[c for c in df.columns if not c.startswith("date") and not c.startswith("gfs") and not c.startswith("fetched")]}')

    # Save to rolling history (last 6 cycles)
    import json as _json
    history = []
    if OUT_HIST_FILE.exists():
        try:
            with open(OUT_HIST_FILE) as f:
                history = _json.load(f)
        except Exception:
            history = []

    # Build history record with key met params only
    hist_record = {
        'fetched_at':    row.get('fetched_at'),
        'gfs_cycle':     row.get('gfs_cycle'),
        'date':          row.get('date'),
        'CAPE':          row.get('CAPE', 0),
        'K_INDEX':       row.get('K_INDEX'),
        'LIFTED_INDEX':  row.get('LIFTED_INDEX'),
        'TOTALS_TOTALS': row.get('TOTALS_TOTALS'),
        'PRECIP_WATER':  row.get('PRECIP_WATER'),
        'ERA5_T2M':      row.get('ERA5_T2M'),
    }
    history.append(hist_record)
    history = history[-KEEP_CYCLES:]

    with open(OUT_HIST_FILE, 'w') as f:
        _json.dump(history, f, indent=2)
    print(f'  Saved → {OUT_HIST_FILE} ({len(history)} cycles)')

    return 0


if __name__ == '__main__':
    import sys
    sys.exit(main())