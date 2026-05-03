"""
Run SAM Windpower simulation (via PySAM) using NASA POWER weather data for wind farms.

Turbine Matching Strategy (Windpower SAM / PySAM)
-------------------------------------------------
We use a two-step turbine selection heuristic against the SAM Wind Turbines Library:

1) Manufacturer + Model Name Match (priority)
   If EIA-860 provides manufacturer and model, we normalize strings (lowercase, trimmed),
   map manufacturer aliases (e.g., “GE”, “General Electric” → “ge”), and extract numeric
   model tokens via regex (e.g., “1.5”). Each library turbine is scored:
     +3: manufacturer alias match
     +2: model-number token match
     +1: full model string match
   We accept matches with score ≥ 3 (typically manufacturer + at least one model token).

2) Capacity Match (fallback)
   If name matching fails, we select the closest turbine by kW rating, preferring turbines
   in the 1.5–5 MW range when available.

3) Generic fallback
   If the library cannot be loaded/parsed, we revert to a generic IEC Class II-like power curve.

Weather inputs
--------------
- NASA POWER wind weather files already exist in: wind_farms_nasa_weather/
- File pattern: wind_farm_{ORISPL}_*.csv
- The weather CSVs contain a 1-line metadata header, then the CSV header row.

Modeling note
-------------
This script models each wind farm as a single "equivalent turbine":
- The turbine power curve is scaled to represent full farm output (num_turbines * per-turbine curve).
- system_capacity is set to the farm kW.

"""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import PySAM.Windpower as WP
from tqdm import tqdm


# -----------------------------
# Generic curve fallback
# -----------------------------
def create_generic_power_curve(rated_power_kw: float = 2500.0):
    """Generic wind turbine power curve (IEC Class II-ish), returns (windspeeds, power_kw)."""
    windspeeds = [0, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 30]
    power_fractions = [
        0.000, 0.000, 0.012, 0.060, 0.140, 0.260, 0.400, 0.560, 0.720, 0.860, 0.960, 1.000,
        1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000, 1.000,
        0.000, 0.000
    ]
    power_kw = [f * rated_power_kw for f in power_fractions]
    return windspeeds, power_kw


# -----------------------------
# Robust loading of SAM Wind Turbines Library.csv
# -----------------------------
def load_sam_turbine_library_csv(lib_path: Path) -> pd.DataFrame | None:
    """
    Load the SAM Wind Turbines Library CSV.

    The file from SAM typically has:
      Row 1: column headers
      Row 2: units
      Row 3: SAM variable mapping
      Row 4+: data

    So we keep the header and skip rows 2 and 3 using skiprows=[1,2].
    """
    if not lib_path.exists():
        return None

    df = pd.read_csv(lib_path, skiprows=[1, 2], on_bad_lines="skip")

    # Required columns
    required = ["Name", "kW Rating", "Rotor Diameter", "IEC Wind Speed Class", "Wind Speed Array", "Power Curve Array"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Turbine library missing columns {missing}. Found: {df.columns.tolist()}")

    df = df.copy()
    df["Name"] = df["Name"].astype(str).str.strip()
    df["kW Rating"] = pd.to_numeric(df["kW Rating"], errors="coerce")
    df["Rotor Diameter"] = pd.to_numeric(df["Rotor Diameter"], errors="coerce")
    df = df.dropna(subset=["Name", "kW Rating", "Rotor Diameter", "Wind Speed Array", "Power Curve Array"]).copy()

    return df


def parse_pipe_array(s: str) -> list[float]:
    """Parse '1|2|3|' safely (handles trailing '|')."""
    return [float(x) for x in str(s).split("|") if x != ""]


# -----------------------------
# Turbine selection logic
# -----------------------------
def select_turbine_from_library(
    df_lib: pd.DataFrame,
    target_turbine_kw: float,
    manufacturer: str | None = None,
    model: str | None = None,
):
    """
    Select a turbine row from library using:
    1) manufacturer+model scoring
    2) capacity closest match

    Returns:
      (hub_height_m, rotor_diameter_m, speeds, power_kw, matched_by_name, turbine_name)
    """
    if df_lib is None or df_lib.empty:
        return None

    df_filtered = df_lib[(df_lib["kW Rating"] >= 1500) & (df_lib["kW Rating"] <= 5000)].copy()
    if df_filtered.empty:
        df_filtered = df_lib.copy()

    manufacturer_aliases = {
        "ge": ["ge", "general electric", "ge energy"],
        "vestas": ["vestas"],
        "siemens": ["siemens", "siemens gamesa"],
        "gamesa": ["gamesa", "siemens gamesa"],
        "nordex": ["nordex"],
        "suzlon": ["suzlon"],
        "mitsubishi": ["mitsubishi"],
        "enercon": ["enercon"],
        "goldwind": ["goldwind"],
        "acciona": ["acciona"],
        "clipper": ["clipper"],
        "repower": ["repower", "senvion"],
        "senvion": ["senvion", "repower"],
    }

    best_row = None
    matched_by_name = False

    if manufacturer and model and str(manufacturer).strip() and str(model).strip():
        manufacturer_clean = str(manufacturer).lower().strip()
        model_clean = str(model).lower().strip()

        # Find manufacturer key
        mfr_key = None
        for key, aliases in manufacturer_aliases.items():
            if any(alias in manufacturer_clean for alias in aliases):
                mfr_key = key
                break

        model_numbers = re.findall(r"\d+\.?\d*", model_clean)

        best_score = 0
        for _, r in df_filtered.iterrows():
            tname = str(r["Name"]).lower()
            score = 0

            # Manufacturer
            if mfr_key:
                if any(alias in tname for alias in manufacturer_aliases.get(mfr_key, [])):
                    score += 3
            else:
                if manufacturer_clean in tname:
                    score += 2

            # Model number tokens
            for num in model_numbers:
                if num and num in tname:
                    score += 2

            # Full model substring
            if model_clean and model_clean in tname:
                score += 1

            if score > best_score and score >= 3:
                best_score = score
                best_row = r
                matched_by_name = True

    # Fallback: capacity match
    if best_row is None:
        df_filtered = df_filtered.copy()
        df_filtered["diff"] = (df_filtered["kW Rating"] - float(target_turbine_kw)).abs()
        best_row = df_filtered.loc[df_filtered["diff"].idxmin()]
        matched_by_name = False

    speeds = parse_pipe_array(best_row["Wind Speed Array"])
    power_kw = parse_pipe_array(best_row["Power Curve Array"])
    rotor_diameter = float(best_row["Rotor Diameter"])
    turbine_name = str(best_row["Name"]).strip()

    # Hub height not provided in this library export; default
    hub_height = 90.0

    return hub_height, rotor_diameter, speeds, power_kw, matched_by_name, turbine_name


# -----------------------------
# Weather file loading
# -----------------------------
def _orispl_to_str(x) -> str | None:
    if pd.isna(x):
        return None
    try:
        return str(int(float(x)))
    except Exception:
        s = str(x).strip()
        return s if s else None


def read_weather_with_fallback(weather_file: Path) -> pd.DataFrame:
    """
    Weather files generated previously typically have:
      line 1: metadata
      line 2: header
      data...
    So skiprows=1 is usually correct. Fall back to skiprows=0 if needed.
    """
    df = pd.read_csv(weather_file, skiprows=1)

    expected = {"wind speed at 80m (m/s)", "wind speed at 100m (m/s)"}
    if len(expected.intersection(df.columns)) > 0:
        return df

    df2 = pd.read_csv(weather_file)
    if len(expected.intersection(df2.columns)) > 0:
        return df2

    raise ValueError(f"Could not find wind speed columns in {weather_file.name}. Columns: {list(df.columns)[:30]}")


# -----------------------------
# Global tracking
# -----------------------------
name_matches = 0
capacity_matches = 0
matched_turbines: dict[str, str] = {}


# -----------------------------
# Core SAM Windpower run
# -----------------------------
def run_windpower_single(
    df_lib: pd.DataFrame | None,
    orispl: int,
    capacity_mw: float,
    weather_file: Path,
    num_turbines: int,
    hub_height: float = 90.0,
    rotor_diameter: float = 110.0,
    manufacturer: str | None = None,
    model: str | None = None,
    state: str | None = None,
    actual_gen_mwh: float | None = None,
    calibrate: bool = False,
):
    """
    Run SAM Windpower for one farm using an "equivalent turbine" representation.

    Returns:
      np.ndarray of hourly generation (MW), length typically 8760.
    """
    global name_matches, capacity_matches, matched_turbines

    if capacity_mw <= 0 or pd.isna(capacity_mw):
        raise ValueError(f"Invalid capacity_mw={capacity_mw}")

    wind = WP.default("WindPowerNone")

    # Determine per-turbine capacity (kW)
    if num_turbines is None or pd.isna(num_turbines) or int(num_turbines) <= 0:
        num_turbines = max(1, int(round(capacity_mw / 2.0)))  # ~2 MW turbines fallback
    num_turbines = int(num_turbines)

    turbine_capacity_kw = (capacity_mw * 1000.0) / float(num_turbines)

    matched_turbine_name = "Generic"

    # Select library turbine if available
    windspeeds = None
    power_kw_farm = None

    if df_lib is not None and not df_lib.empty:
        sel = select_turbine_from_library(df_lib, turbine_capacity_kw, manufacturer, model)
        if sel:
            lib_hub_height, lib_rotor_diameter, speeds, power_kw, matched_by_name, matched_turbine_name = sel
            if matched_by_name:
                name_matches += 1
            else:
                capacity_matches += 1

            # If rotor diameter not known, use library rotor diameter
            if rotor_diameter == 110.0:
                rotor_diameter = float(lib_rotor_diameter)

            # Scale selected turbine curve to match per-turbine rating, then multiply to farm
            max_power_kw = max(power_kw) if power_kw else turbine_capacity_kw
            scale_factor = turbine_capacity_kw / max_power_kw if max_power_kw > 0 else 1.0

            power_kw_per_turbine = [p * scale_factor for p in power_kw]
            power_kw_farm = [p * num_turbines for p in power_kw_per_turbine]
            windspeeds = speeds

    # Fallback: generic curve
    if windspeeds is None or power_kw_farm is None:
        windspeeds, power_kw_single = create_generic_power_curve(rated_power_kw=turbine_capacity_kw)
        power_kw_farm = [p * num_turbines for p in power_kw_single]

    # Weather
    df_weather = read_weather_with_fallback(weather_file)

    if "wind speed at 80m (m/s)" in df_weather.columns:
        wind_speed = pd.to_numeric(df_weather["wind speed at 80m (m/s)"], errors="coerce").to_numpy()
        actual_height = 80
    elif "wind speed at 100m (m/s)" in df_weather.columns:
        wind_speed = pd.to_numeric(df_weather["wind speed at 100m (m/s)"], errors="coerce").to_numpy()
        actual_height = 100
    else:
        raise ValueError(f"No suitable wind speed column found in {weather_file.name}")

    wind_speed = np.nan_to_num(wind_speed, nan=0.0)

    # Basic check
    if len(wind_speed) != 8760:
        print(f"  WARNING: ORISPL {orispl} has {len(wind_speed)} hours (expected 8760).")

    # Wind resource (PySAM-compatible SRW-style rows)
    if "air temperature at 2m (C)" in df_weather.columns:
        t2 = pd.to_numeric(df_weather["air temperature at 2m (C)"], errors="coerce").fillna(10.0)
    else:
        t2 = pd.Series(10.0, index=df_weather.index)

    if "surface air pressure (Pa)" in df_weather.columns:
        p_pa = pd.to_numeric(df_weather["surface air pressure (Pa)"], errors="coerce").fillna(101325.0)
    else:
        p_pa = pd.Series(101325.0, index=df_weather.index)

    if "wind direction at 80m (deg)" in df_weather.columns:
        wdir = pd.to_numeric(df_weather["wind direction at 80m (deg)"], errors="coerce").fillna(0.0)
    elif "wind direction at 100m (deg)" in df_weather.columns:
        wdir = pd.to_numeric(df_weather["wind direction at 100m (deg)"], errors="coerce").fillna(0.0)
    elif "wind direction at 50m (deg)" in df_weather.columns:
        wdir = pd.to_numeric(df_weather["wind direction at 50m (deg)"], errors="coerce").fillna(0.0)
    else:
        wdir = pd.to_numeric(df_weather["wind direction at 10m (deg)"], errors="coerce").fillna(0.0)

    rows = []
    for i in range(len(wind_speed)):
        # Pressure in atm for this resource-data path.
        rows.append([float(t2.iloc[i]), float(p_pa.iloc[i]) / 101325.0, float(wind_speed[i]), float(wdir.iloc[i])])

    wind.Resource.wind_resource_model_choice = 0
    wind.Resource.wind_resource_data = {
        "heights": [2.0, 2.0, float(actual_height), float(actual_height)],
        "fields": [1, 2, 3, 4],  # temperature, pressure, speed, direction
        "data": rows,
        "tz": -5,
        "lat": 0.0,
        "lon": 0.0,
        "elev": 0.0,
    }

    # Turbine + farm config
    wind.Turbine.wind_turbine_hub_ht = float(hub_height)
    wind.Turbine.wind_turbine_rotor_diameter = float(rotor_diameter)
    wind.Turbine.wind_turbine_powercurve_windspeeds = windspeeds
    wind.Turbine.wind_turbine_powercurve_powerout = power_kw_farm
    wind.Turbine.wind_resource_shear = 0.14
    wind.Turbine.wind_turbine_max_cp = 0.45

    wind.Farm.system_capacity = capacity_mw * 1000.0  # kW
    wind.Farm.wind_farm_wake_model = 0
    wind.Farm.wind_farm_xCoordinates = [0]
    wind.Farm.wind_farm_yCoordinates = [0]

    # Losses (heuristic)
    base_bop = 10.0
    base_grid = 10.0
    base_turb = 12.0

    terrain_factor = 1.0
    if state:
        terrain_adjustments = {
            "PA": 1.10,
            "WV": 1.15,
            "IL": 0.95,
            "IN": 0.95,
            "OH": 1.00,
            "MD": 1.00,
            "VA": 0.90,
            "NC": 0.90,
        }
        terrain_factor = terrain_adjustments.get(str(state).strip().upper(), 1.0)

    curtailment_factor = 0.0
    if capacity_mw > 150:
        curtailment_factor = min(5.0, (capacity_mw - 150.0) / 50.0)

    wind.Losses.avail_bop_loss = base_bop * terrain_factor
    wind.Losses.avail_grid_loss = base_grid
    wind.Losses.avail_turb_loss = base_turb + curtailment_factor

    # Optional calibration (heuristic)
    if calibrate and actual_gen_mwh is not None and pd.notna(actual_gen_mwh):
        wind.execute()
        initial_mwh = float(np.array(wind.Outputs.gen).sum() / 1000.0)  # kWh->MWh (matches your convention)

        if initial_mwh > 0:
            ratio = float(actual_gen_mwh) / initial_mwh
            if 0.5 <= ratio <= 2.0:
                current_total = 1 - (1 - wind.Losses.avail_bop_loss / 100.0) * \
                                    (1 - wind.Losses.avail_grid_loss / 100.0) * \
                                    (1 - wind.Losses.avail_turb_loss / 100.0)

                target_total = 1 - ratio * (1 - current_total)
                target_total = max(0.05, min(0.60, target_total))

                if target_total > current_total:
                    adj = (target_total - current_total) * 100.0 / 3.0
                    wind.Losses.avail_bop_loss = min(25, wind.Losses.avail_bop_loss + adj)
                    wind.Losses.avail_grid_loss = min(25, wind.Losses.avail_grid_loss + adj)
                    wind.Losses.avail_turb_loss = min(25, wind.Losses.avail_turb_loss + adj)
                else:
                    adj = (current_total - target_total) * 100.0 / 3.0
                    wind.Losses.avail_bop_loss = max(0, wind.Losses.avail_bop_loss - adj)
                    wind.Losses.avail_grid_loss = max(0, wind.Losses.avail_grid_loss - adj)
                    wind.Losses.avail_turb_loss = max(0, wind.Losses.avail_turb_loss - adj)

    wind.execute()

    matched_turbines[str(orispl)] = matched_turbine_name

    # Keep your convention: Outputs.gen treated as kW; /1000 => MW
    gen_output_mw = np.array(wind.Outputs.gen) / 1000.0
    return gen_output_mw


# -----------------------------
# Main
# -----------------------------
def main():
    print("=" * 70)
    print("RUNNING SAM WINDPOWER WITH NASA POWER WEATHER")
    print("=" * 70)
    print()

    # Load wind generation data
    df_wind = pd.read_excel("data/wind_2022_PJMfilled.xlsx", sheet_name=0)
    print(f"✓ Loaded {len(df_wind)} wind farms")

    # Load EIA-860 turbine data
    print("\nLoading EIA-860 turbine specifications...")
    eia860_path = Path("data/3_2_Wind_Y2022.xlsx")

    if eia860_path.exists():
        try:
            df_eia = pd.read_excel(eia860_path, sheet_name="Operable", skiprows=1)
            print(f"✓ Loaded {len(df_eia)} turbine records from EIA-860")

            agg_dict = {
                "Number of Turbines": "sum",
                "Turbine Hub Height (Feet)": "median",
            }
            if "Predominant Turbine Manufacturer" in df_eia.columns:
                agg_dict["Predominant Turbine Manufacturer"] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]
            if "Predominant Turbine Model Number" in df_eia.columns:
                agg_dict["Predominant Turbine Model Number"] = lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0]

            eia_agg = df_eia.groupby("Plant Code").agg(agg_dict).reset_index()

            eia_agg = eia_agg.rename(columns={
                "Number of Turbines": "num_turbines",
                "Turbine Hub Height (Feet)": "hub_height_ft",
                "Predominant Turbine Manufacturer": "turbine_manufacturer",
                "Predominant Turbine Model Number": "turbine_model",
            })

            eia_agg["hub_height_m"] = eia_agg["hub_height_ft"] * 0.3048
            eia_agg["rotor_diameter_m"] = 110.0  # default; updated from library when used

            df_wind = df_wind.merge(eia_agg, on="Plant Code", how="left")
            print(f"✓ Merged EIA-860 data for {int(df_wind['num_turbines'].notna().sum())} farms")

        except Exception as e:
            print(f"  Warning: Could not load EIA-860: {e}")
            df_wind["num_turbines"] = np.nan
            df_wind["hub_height_m"] = 90.0
            df_wind["rotor_diameter_m"] = 110.0
    else:
        print("  EIA-860 file not found")
        df_wind["num_turbines"] = np.nan
        df_wind["hub_height_m"] = 90.0
        df_wind["rotor_diameter_m"] = 110.0

    # Explicit columns
    df_wind["ORISPL"] = df_wind["Plant Code"]
    df_wind["Capacity_MW"] = pd.to_numeric(df_wind["Nameplate Capacity (MW)"], errors="coerce")

    # Map weather files
    nasa_dir = Path("wind_farms_nasa_weather")
    if not nasa_dir.exists():
        raise FileNotFoundError(f"Weather directory not found: {nasa_dir.resolve()}")

    file_mapping = {}
    for idx, row in df_wind.iterrows():
        orispl_str = _orispl_to_str(row["ORISPL"])
        if not orispl_str:
            continue
        matches = sorted(nasa_dir.glob(f"wind_farm_{orispl_str}_*.csv"))
        if matches:
            file_mapping[idx] = matches[0]

    df_wind = df_wind.loc[list(file_mapping.keys())].copy()
    df_wind["weather_file"] = df_wind.index.map(file_mapping)

    print(f"\n✓ Found weather for {len(df_wind)} farms")
    print()

    # Load turbine library once
    lib_path = Path("data/Wind_Turbines.csv")
    df_lib = load_sam_turbine_library_csv(lib_path)
    if df_lib is None:
        print("  WARNING: Could not load turbine library; using generic curve fallback for all farms.")

    # Run simulations
    results = {}
    errors = []

    print("Running simulations...")
    calibrate_losses = True  # set False for uncalibrated run

    for _, row in tqdm(df_wind.iterrows(), total=len(df_wind)):
        orispl = int(float(row["ORISPL"]))
        try:
            num_turbines = row.get("num_turbines", np.nan)
            hub_height = row.get("hub_height_m", 90.0)
            rotor_diameter = row.get("rotor_diameter_m", 110.0)
            manufacturer = row.get("turbine_manufacturer", None)
            model = row.get("turbine_model", None)
            state = row.get("State", None)
            actual_gen = row.get("Wind Annual Generation (MWh)", None)

            if pd.isna(num_turbines) or float(num_turbines) <= 0:
                num_turbines = max(1, int(round(float(row["Capacity_MW"]) / 2.0)))

            gen_mw = run_windpower_single(
                df_lib=df_lib,
                orispl=orispl,
                capacity_mw=float(row["Capacity_MW"]),
                weather_file=Path(row["weather_file"]),
                hub_height=float(hub_height) if pd.notna(hub_height) else 90.0,
                rotor_diameter=float(rotor_diameter) if pd.notna(rotor_diameter) else 110.0,
                num_turbines=int(num_turbines),
                manufacturer=manufacturer,
                model=model,
                state=state,
                actual_gen_mwh=actual_gen,
                calibrate=calibrate_losses,
            )
            results[str(orispl)] = gen_mw

        except Exception as e:
            errors.append(f"ORISPL {orispl}: {e}")
            if len(errors) <= 3:
                print(f"  Error on {orispl}: {e}")

    print(f"\n✓ Simulated {len(results)} farms")
    print(f"  Turbine matches: {name_matches} by manufacturer/model, {capacity_matches} by capacity")
    if errors:
        print(f"⚠ {len(errors)} errors")

    # Save hourly generation
    calibration_suffix = "_calibrated" if calibrate_losses else ""
    output_dir = Path(f"Processed/pjm/wind-farms/results_nasa_power{calibration_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    hourly_gen = pd.DataFrame(results)
    hourly_gen.to_csv(output_dir / "hourly_generation_mw.csv")
    print(f"\n✓ Saved hourly data: {output_dir / 'hourly_generation_mw.csv'}")

    # Summary
    annual_gen_mwh = hourly_gen.sum()
    total_mwh = float(annual_gen_mwh.sum())
    total_capacity = float(df_wind["Capacity_MW"].sum())
    avg_cf = (total_mwh / (total_capacity * 8760.0)) * 100.0 if total_capacity > 0 else np.nan

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total generation: {total_mwh:,.0f} MWh ({total_mwh/1e6:.3f} TWh)")
    print(f"Total capacity: {total_capacity:,.1f} MW")
    print(f"Average CF: {avg_cf:.1f}%")
    print(f"Results: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
