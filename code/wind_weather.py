#!/usr/bin/env python3
"""Download NASA POWER Weather Data for PJM Wind Farms - 2022"""


import sys
import time
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

# ----------------------------
# Dependency handling (lightweight)
# ----------------------------
def _ensure_import(pkg_name: str, import_name: Optional[str] = None):
    """Try import; if missing, attempt pip install (best-effort)."""
    name = import_name or pkg_name
    try:
        return __import__(name)
    except ImportError:
        print(f"Installing {pkg_name}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", pkg_name])
        return __import__(name)

pd = _ensure_import("pandas", "pandas")
np = _ensure_import("numpy", "numpy")
requests = _ensure_import("requests", "requests")
openpyxl = _ensure_import("openpyxl", "openpyxl")  # noqa: F401

# ----------------------------
# NASA POWER downloader
# ----------------------------
NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/hourly/point"

# Parameters you requested (hourly)
POWER_PARAMETERS = [
    "PS",           # Surface pressure (kPa in POWER docs; convert to Pa if needed)
    "RH2M",         # Relative humidity at 2m (%)
    "PRECTOTCORR",  # Precipitation (mm/hr)
    "WS10M",        # Wind speed 10m (m/s)
    "WS50M",        # Wind speed 50m (m/s)
    "WD10M",        # Wind direction 10m (deg)
    "WD50M",        # Wind direction 50m (deg)
    "T2M",          # Air temperature 2m (C)
    "T10M",         # Air temperature 10m (C)
]

def _power_request_params(lat: float, lon: float, year: int) -> Dict[str, Any]:
    return {
        "parameters": ",".join(POWER_PARAMETERS),
        "community": "RE",
        "longitude": float(lon),
        "latitude": float(lat),
        "start": f"{year}0101",
        "end": f"{year}1231",
        "format": "JSON",
    }

def _safe_float(x) -> Optional[float]:
    try:
        if pd.isna(x):
            return None
        return float(x)
    except Exception:
        return None

def download_nasa_power_hourly(
    lat: float,
    lon: float,
    year: int = 2022,
    timeout_s: int = 300,
    max_retries: int = 3,
    backoff_s: float = 2.0,
) -> Optional["pd.DataFrame"]:
    """Download NASA POWER hourly data and return a clean numeric DataFrame."""
    params = _power_request_params(lat, lon, year)

    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = requests.get(NASA_POWER_URL, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()

            if "properties" not in data or "parameter" not in data["properties"]:
                raise ValueError("Unexpected NASA POWER response structure (missing properties.parameter).")

            p = data["properties"]["parameter"]

            # Ensure required parameter exists
            if "PS" not in p:
                raise ValueError("Missing PS in response; cannot build timestamps.")

            # Sort timestamps (don’t trust dict insertion)
            timestamps = sorted(p["PS"].keys())

            rows: List[Dict[str, Any]] = []

            for ts in timestamps:
                # ts format: YYYYMMDDHH
                y = int(ts[:4])
                m = int(ts[4:6])
                d = int(ts[6:8])
                h = int(ts[8:10])

                # read values; POWER uses -999 for missing
                def v(name: str) -> float:
                    return p.get(name, {}).get(ts, -999)

                ps = v("PS")
                rh2m = v("RH2M")
                prec = v("PRECTOTCORR")
                ws10m = v("WS10M")
                ws50m = v("WS50M")
                wd10m = v("WD10M")
                wd50m = v("WD50M")
                t2m = v("T2M")
                t10m = v("T10M")

                # Convert to numeric with NaN for missing
                def to_nan(x: float) -> float:
                    return np.nan if x == -999 else float(x)

                ps = to_nan(ps)
                rh2m = to_nan(rh2m)
                prec = to_nan(prec)
                ws10m = to_nan(ws10m)
                ws50m = to_nan(ws50m)
                wd10m = to_nan(wd10m)
                wd50m = to_nan(wd50m)
                t2m = to_nan(t2m)
                t10m = to_nan(t10m)

                # Pressure: POWER PS is typically kPa -> convert to Pa if you want Pa
                # If PS is already Pa in your workflow, remove the *1000 conversion.
                ps_pa = ps * 1000.0 if not np.isnan(ps) else np.nan

                row: Dict[str, Any] = {
                    "Year": y,
                    "Month": m,
                    "Day": d,
                    "Hour": h,
                    "Minute": 0,  # hourly at top of hour
                    "surface air pressure (Pa)": np.round(ps_pa, 2) if not np.isnan(ps_pa) else np.nan,
                    "relative humidity at 2m (%)": np.round(rh2m, 2) if not np.isnan(rh2m) else np.nan,
                    "surface precipitation rate (mm/h)": np.round(prec, 4) if not np.isnan(prec) else np.nan,
                    "wind speed at 10m (m/s)": np.round(ws10m, 2) if not np.isnan(ws10m) else np.nan,
                    "wind speed at 50m (m/s)": np.round(ws50m, 2) if not np.isnan(ws50m) else np.nan,
                    "wind direction at 10m (deg)": np.round(wd10m, 2) if not np.isnan(wd10m) else np.nan,
                    "wind direction at 50m (deg)": np.round(wd50m, 2) if not np.isnan(wd50m) else np.nan,
                    "air temperature at 2m (C)": np.round(t2m, 2) if not np.isnan(t2m) else np.nan,
                    "air temperature at 10m (C)": np.round(t10m, 2) if not np.isnan(t10m) else np.nan,
                }

                # Extrapolate wind speeds using power law from WS50M when available.
                if not np.isnan(ws50m):
                    alpha = 0.143
                    for height in [40, 60, 80, 100, 120, 140, 160, 200]:
                        row[f"wind speed at {height}m (m/s)"] = np.round(ws50m * (height / 50.0) ** alpha, 2)

                        # Use WD10M for <=40m; WD50M for >=60m (simple improvement)
                        if height <= 40:
                            row[f"wind direction at {height}m (deg)"] = np.round(wd10m, 2) if not np.isnan(wd10m) else np.nan
                        else:
                            row[f"wind direction at {height}m (deg)"] = np.round(wd50m, 2) if not np.isnan(wd50m) else (
                                np.round(wd10m, 2) if not np.isnan(wd10m) else np.nan
                            )

                rows.append(row)

            df = pd.DataFrame(rows)

            # Basic sanity check (2023 has 8760 hours; leap years 8784)
            expected = 8784 if (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)) else 8760
            if len(df) < expected * 0.98:
                print(f"  ⚠ Warning: only {len(df)} hourly records returned (expected ~{expected}).")

            return df

        except Exception as e:
            last_err = e
            if attempt < max_retries:
                sleep_s = backoff_s * (2 ** (attempt - 1))
                print(f"  ⚠ Attempt {attempt}/{max_retries} failed: {e}")
                print(f"  ↻ Retrying in {sleep_s:.1f}s...")
                time.sleep(sleep_s)
            else:
                print(f"  ✗ Failed after {max_retries} attempts: {e}")
                return None

    print(f"  ✗ Failed: {last_err}")
    return None

# ----------------------------
# Main script
# ----------------------------
def main():
    print("=" * 70)
    print("NASA POWER WEATHER DATA - PJM WIND FARMS (Hourly)")
    print("=" * 70)

    input_file = Path("data/wind_2022_PJMfilled.xlsx")
    if not input_file.exists():
        print(f"\n✗ Input file not found: {input_file.resolve()}")
        sys.exit(1)

    print(f"\nReading: {input_file.name}")
    df_farms = pd.read_excel(input_file)

    # Column names (adjust if your file differs)
    lat_col = "Latitude"
    lon_col = "Longitude"
    plant_id_col = "Plant Code"
    plant_name_col = "Plant Name"

    for col in [lat_col, lon_col, plant_id_col, plant_name_col]:
        if col not in df_farms.columns:
            print(f"\n✗ Missing required column '{col}' in Excel.")
            print(f"Available columns: {list(df_farms.columns)}")
            sys.exit(1)

    # Clean lat/lon
    df_farms[lat_col] = pd.to_numeric(df_farms[lat_col], errors="coerce")
    df_farms[lon_col] = pd.to_numeric(df_farms[lon_col], errors="coerce")

    df_farms = df_farms.dropna(subset=[lat_col, lon_col, plant_id_col, plant_name_col]).copy()
    df_farms[plant_id_col] = df_farms[plant_id_col].astype(str)

    print(f"✓ Loaded {len(df_farms)} wind farm rows")

    # Optional: dedupe by unique (lat, lon) to avoid duplicate downloads
    # Keep the first plant per coordinate pair for naming.
    df_sites = (
        df_farms
        .drop_duplicates(subset=[lat_col, lon_col])
        .reset_index(drop=True)
    )
    print(f"✓ Unique sites by lat/lon: {len(df_sites)}")

    output_dir = Path("wind_farms_nasa_weather")
    output_dir.mkdir(exist_ok=True)
    print(f"✓ Output directory: {output_dir.resolve()}")

    year = 2022
    print(f"\n{'='*70}")
    print(f"DOWNLOADING WEATHER DATA FOR {len(df_sites)} UNIQUE WIND FARM SITES ({year})")
    print(f"Rate limit: 2 seconds between sites (plus retries if needed)")
    print("=" * 70)

    success_count = 0
    fail_count = 0

    for i, site in df_sites.iterrows():
        lat = float(site[lat_col])
        lon = float(site[lon_col])
        plant_id = str(site[plant_id_col]).strip()
        plant_name = str(site[plant_name_col]).strip()

        print(f"\n[{i+1}/{len(df_sites)}] {plant_name} (Plant Code: {plant_id})")
        print(f"  Location: {lat:.5f}, {lon:.5f}")

        # Use Plant Code as SiteID (simple and stable)
        site_id = plant_id if plant_id else f"{lat:.5f}_{lon:.5f}"

        # Output filename
        safe_lat = f"{lat:.5f}"
        safe_lon = f"{lon:.5f}"
        output_file = output_dir / f"wind_farm_{site_id}_{safe_lat}_{safe_lon}_{year}.csv"

        if output_file.exists():
            print("  ✓ Already exists, skipping")
            success_count += 1
            continue

        print("  Downloading...")
        df_weather = download_nasa_power_hourly(lat, lon, year=year)

        if df_weather is None or df_weather.empty:
            print("  ✗ Failed to download / empty response")
            fail_count += 1
        else:
            # Header row (WIND-Toolkit-ish)
            # NOTE: timezones are tricky with DST. Keeping -5 like your original.
            timezone = -5
            header_row = (
                f"SiteID,{site_id},"
                f"Site Timezone,{timezone},"
                f"Data Timezone,{timezone},"
                f"Longitude,{lon:.10f},"
                f"Latitude,{lat:.10f}\n"
            )

            with open(output_file, "w", encoding="utf-8") as f:
                f.write(header_row)

            # Write weather data
            df_weather.to_csv(output_file, mode="a", index=False)

            print(f"  ✓ Saved: {output_file.name} ({len(df_weather)} rows)")
            success_count += 1

        # Rate limiting
        if i < len(df_sites) - 1:
            time.sleep(2)

    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE")
    print("=" * 70)
    print(f"✓ Success: {success_count}")
    print(f"✗ Failed: {fail_count}")
    print(f"\nFiles saved to: {output_dir.resolve()}\n")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted by user.")
        sys.exit(130)