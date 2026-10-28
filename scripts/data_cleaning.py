from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "india_housing_prices.csv"
OUTPUT = ROOT / "data" / "cleaned_housing_data.csv"
REPORT = ROOT / "data" / "cleaning_report.json"

def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    original_rows = len(df)
    original_duplicates = int(df.duplicated().sum())
    missing_before = int(df.isna().sum().sum())
    out = df.copy()

    for col in out.select_dtypes(include=["object"]).columns:
        out[col] = out[col].astype("string").str.strip()
        out[col] = out[col].replace({"": pd.NA, "NA": pd.NA, "N/A": pd.NA, "null": pd.NA, "None": pd.NA})

    out = out.drop_duplicates()
    numeric_cols = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt",
                    "Year_Built", "Floor_No", "Total_Floors", "Age_of_Property",
                    "Nearby_Schools", "Nearby_Hospitals"]
    for col in numeric_cols:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    required = [c for c in ["City", "State", "Property_Type", "Size_in_SqFt", "Price_in_Lakhs"] if c in out.columns]
    out = out.dropna(subset=required)
    out = out[(out["Size_in_SqFt"] > 0) & (out["Price_in_Lakhs"] > 0)]
    if "BHK" in out:
        out = out[out["BHK"].between(1, 15)]
    if "Year_Built" in out:
        out = out[out["Year_Built"].between(1800, 2026)]
    if "Total_Floors" in out and "Floor_No" in out:
        out = out[(out["Floor_No"] >= 0) & (out["Total_Floors"] >= 1)]
        out["Floor_No"] = np.minimum(out["Floor_No"], out["Total_Floors"])
    if "Year_Built" in out:
        out["Age_of_Property"] = (2026 - out["Year_Built"]).clip(lower=0)
    if {"Price_in_Lakhs", "Size_in_SqFt"}.issubset(out.columns):
        out["Price_per_SqFt"] = (out["Price_in_Lakhs"] * 100000 / out["Size_in_SqFt"]).round(2)

    for col in numeric_cols:
        if col in out.columns and out[col].isna().any():
            out[col] = out[col].fillna(out[col].median())
    for col in out.select_dtypes(include=["object", "string"]).columns:
        out[col] = out[col].fillna("Unknown")

    for col in ["Parking_Space", "Security"]:
        if col in out.columns:
            out[col] = out[col].replace({"YES": "Yes", "NO": "No", "yes": "Yes", "no": "No"})

    if "Amenities" in out.columns:
        out["Amenities_Count"] = out["Amenities"].astype(str).apply(
            lambda x: 0 if x.strip().lower() in ["", "unknown", "nan"] else len([v for v in x.split(",") if v.strip()])
        )

    report = {
        "original_rows": int(original_rows),
        "cleaned_rows": int(len(out)),
        "rows_removed": int(original_rows - len(out)),
        "original_duplicate_rows": original_duplicates,
        "missing_cells_before": missing_before,
        "missing_cells_after": int(out.isna().sum().sum()),
        "notes": [
            "Price_per_SqFt was recalculated as Price_in_Lakhs * 100000 / Size_in_SqFt.",
            "Age_of_Property was recalculated as 2026 - Year_Built.",
            "The supplied data does not contain observed future prices or actual investment outcomes."
        ]
    }
    return out, report

if __name__ == "__main__":
    if not SOURCE.exists():
        raise FileNotFoundError(f"Dataset not found: {SOURCE}")
    data = pd.read_csv(SOURCE)
    cleaned, report = clean_data(data)
    cleaned.to_csv(OUTPUT, index=False)
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved cleaned dataset: {OUTPUT}")
    print(json.dumps(report, indent=2))
