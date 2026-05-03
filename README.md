# 2022 Wind Workflow README

This project runs a 3-step wind workflow for PJM plants:

1. Download weather (`code/wind_weather.py`)
2. Simulate hourly generation with SAM (`code/run_sam_windpower_nasa_power.py`)
3. Compare modeled vs actual annual generation (`code/compare_generation.py`)

---

## Files You Are Given

- `code/wind_weather.py`
- `code/run_sam_windpower_nasa_power.py`  
  (if your copy is named `run_sam_windpower_nasa.py`, use that filename instead)
- `code/compare_generation.py`
- `data/3_2_Wind_Y2022.xlsx` (change if not testing for 2022)
- `data/Wind_Turbines.csv`
- `data/wind_farms_nasa_weather/` (update for wind outside of 2022)

---

## File You Must Input

Create `data/wind_2022_PJMfilled.xlsx` with one row per plant and at least these columns:

- `Plant Code`
- `Plant Name`
- `State`
- `Latitude`
- `Longitude`
- `Nameplate Capacity (MW)`
- `Wind Annual Generation (MWh)`

Brief guidance:
- Use consistent `Plant Code` values across all files (same IDs as EIA where possible).
- Keep latitude/longitude numeric decimal degrees.
- Keep capacity in MW and annual generation in MWh.
- Remove duplicate plant rows unless you intentionally model duplicates.

---

## Environment Setup

From project root (`2022Wind`):

```bash
python3 -m venv venv
source venv/bin/activate
pip install pandas numpy requests openpyxl pysam tqdm
```

---

## Run Order

Always run from the project root directory (`2022Wind`), not from `code/`.

### 1) Download weather files

```bash
python3 code/wind_weather.py
```

What this does:
- Reads `data/wind_2022_PJMfilled.xlsx`
- Downloads NASA POWER hourly weather (2022)
- Writes CSV files to `wind_farms_nasa_weather/`

### 2) Run SAM wind simulation

```bash
python3 code/run_sam_windpower_nasa_power.py
```

What this does:
- Reads:
  - `data/wind_2022_PJMfilled.xlsx`
  - `data/3_2_Wind_Y2022.xlsx`
  - `data/Wind_Turbines.csv` (or `data/Wind Turbines.csv`, depending on script/local filename)
  - `wind_farms_nasa_weather/*.csv`
- Writes hourly generation output to:
  - `Processed/pjm/wind-farms/results_nasa_power_calibrated/hourly_generation_mw.csv`
  - (plus summary CSV/txt files in the same output folder)

### 3) Compare modeled vs actual annual generation

```bash
python3 code/compare_generation.py
```

Output:
- `generation_comparison_2022.txt`

Note:
- `compare_generation.py` currently points to `data/Wind_Generation_PJM_2022.xlsx` for actuals.
- If you want it to compare against your custom file, update `actual_file` in `code/compare_generation.py` to `data/wind_2022_PJMfilled.xlsx`.
