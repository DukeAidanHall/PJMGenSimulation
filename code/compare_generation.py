import pandas as pd
from pathlib import Path

hourly_file = Path("Processed/pjm/wind-farms/results_nasa_power_calibrated/hourly_generation_mw.csv")
actual_file = Path("data/Wind_Generation_PJM_2022.xlsx")
output_file = Path("generation_comparison_2022.txt")

hourly = pd.read_csv(hourly_file, index_col=0)

modeled = hourly.sum(axis=0)
modeled.index = modeled.index.astype(str)

actual_df = pd.read_excel(actual_file)

actual_df["Plant Code"] = actual_df["Plant Code"].astype(str)
actual_df["Wind Annual Generation (MWh)"] = pd.to_numeric(
    actual_df["Wind Annual Generation (MWh)"],
    errors="coerce"
)

actual = actual_df.set_index("Plant Code")["Wind Annual Generation (MWh)"]

comparison = pd.DataFrame({
    "Modeled Generation (MWh)": modeled,
    "Actual Wind Annual Generation (MWh)": actual
})

comparison["Difference (MWh)"] = (
    comparison["Modeled Generation (MWh)"]
    - comparison["Actual Wind Annual Generation (MWh)"]
)

comparison["Percent Error (%)"] = (
    comparison["Difference (MWh)"]
    / comparison["Actual Wind Annual Generation (MWh)"]
    * 100
)

comparison = comparison.sort_index()

with open(output_file, "w") as f:
    f.write("PJM Wind Generation Comparison - 2022\n")
    f.write("=" * 80 + "\n\n")
    f.write(comparison.to_string(float_format=lambda x: f"{x:,.2f}"))
    f.write("\n\n")
    f.write("=" * 80 + "\n")
    f.write(f"Total Modeled Generation (MWh): {comparison['Modeled Generation (MWh)'].sum():,.2f}\n")
    f.write(f"Total Actual Generation (MWh): {comparison['Actual Wind Annual Generation (MWh)'].sum():,.2f}\n")
    f.write(f"Total Difference (MWh): {comparison['Difference (MWh)'].sum():,.2f}\n")

print(f"Saved comparison to {output_file}")