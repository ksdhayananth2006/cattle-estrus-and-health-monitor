"""
BOVIGENIX - Data Processing & Validation (utils/data_processing.py)
-------------------------------------------------------------------
This module handles:
1. Column mapping and normalization for uploaded sensor datasets.
2. Data quality checks: missing values, duplicate detection, and range validation.
3. Generation of a realistic, clearly-flagged SYNTHETIC cattle dataset
   for student learning, system testing, and classroom demonstrations.

Biological Context:
- Healthy dairy cow resting body temperature: 38.0°C - 39.2°C.
- Normal daily rumination time: 400 - 550 minutes/day.
- Estrus behavior: Surge in walking activity, decreased rumination, decreased
  lying time, slight temporary dip in milk yield, slight body temp rise (0.3 - 0.5°C).
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random
from calculations import calculate_thi


# Standard internal column names expected by BOVIGENIX
STANDARD_COLUMNS = [
    "cow_id",
    "timestamp",
    "activity",
    "rumination_time",
    "lying_time",
    "eating_time",
    "milk_yield",
    "body_temperature",
    "ambient_temperature",
    "relative_humidity",
    "estrus_label",
    "health_label"
]

# Aliases mapping common CSV header variations to our standard names
COLUMN_ALIASES = {
    "cow_id": ["cow_id", "cowid", "cow id", "animal_id", "animal id", "tag_id", "tag", "id"],
    "timestamp": ["timestamp", "datetime", "date_time", "date", "time", "observation_time"],
    "activity": ["activity", "steps", "motion", "activity_index", "movement"],
    "rumination_time": ["rumination_time", "rumination", "rumination_min", "rumination_hours", "rum_time"],
    "lying_time": ["lying_time", "lying", "rest_time", "lying_hours", "lying_min"],
    "eating_time": ["eating_time", "feeding_time", "eating", "feed_intake_time"],
    "milk_yield": ["milk_yield", "milk", "yield", "milk_liters", "daily_milk"],
    "body_temperature": ["body_temperature", "body_temp", "cow_temp", "temp_rectal", "temperature"],
    "ambient_temperature": ["ambient_temperature", "ambient_temp", "env_temp", "air_temperature", "air_temp"],
    "relative_humidity": ["relative_humidity", "humidity", "rh", "ambient_humidity", "rel_humidity"],
    "estrus_label": ["estrus_label", "estrus", "heat", "heat_label", "in_estrus", "standing_heat"],
    "health_label": ["health_label", "health", "health_status", "disease", "clinical_condition"]
}


def normalize_column_names(df):
    """
    Scans dataframe headers and maps common column names to standard BOVIGENIX names.
    Returns: (renamed_df, mapping_dictionary)
    """
    df_clean = df.copy()
    col_mapping = {}

    for standard_col, aliases in COLUMN_ALIASES.items():
        for existing_col in df_clean.columns:
            clean_name = str(existing_col).strip().lower().replace("-", "_").replace(" ", "_")
            if clean_name in aliases:
                col_mapping[existing_col] = standard_col
                break

    df_clean = df_clean.rename(columns=col_mapping)
    return df_clean, col_mapping


def validate_dataset(df):
    """
    Performs data quality validation on an uploaded dataset.

    Returns a dictionary report containing:
    - total_rows: Number of observations
    - columns_found: List of recognized standard columns
    - missing_required: List of mandatory columns missing
    - duplicate_count: Number of duplicate (cow_id, timestamp) pairs
    - missing_value_summary: DataFrame with missing counts and %
    - biological_warnings: List of out-of-range sensor alerts
    """
    report = {
        "is_valid": True,
        "total_rows": len(df),
        "columns_found": [],
        "missing_required": [],
        "duplicate_count": 0,
        "missing_value_summary": None,
        "biological_warnings": []
    }

    # 1. Check mandatory columns (cow_id and timestamp are essential)
    required_cols = ["cow_id", "timestamp"]
    for col in required_cols:
        if col not in df.columns:
            report["missing_required"].append(col)
            report["is_valid"] = False

    report["columns_found"] = [c for c in STANDARD_COLUMNS if c in df.columns]

    # 2. Duplicate detection
    if "cow_id" in df.columns and "timestamp" in df.columns:
        dups = df.duplicated(subset=["cow_id", "timestamp"]).sum()
        report["duplicate_count"] = int(dups)

    # 3. Missing values summary
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df) * 100).round(1)
    report["missing_value_summary"] = pd.DataFrame({
        "Missing Values": missing_counts,
        "Percentage (%)": missing_pct
    })

    # 4. Biological sanity checks
    if "body_temperature" in df.columns:
        invalid_body_temps = df[(df["body_temperature"] < 35.0) | (df["body_temperature"] > 43.0)]
        if len(invalid_body_temps) > 0:
            report["biological_warnings"].append(
                f"Found {len(invalid_body_temps)} body temperature readings outside physiological limits (35.0°C - 43.0°C)."
            )

    if "relative_humidity" in df.columns:
        invalid_rh = df[(df["relative_humidity"] < 0) | (df["relative_humidity"] > 100)]
        if len(invalid_rh) > 0:
            report["biological_warnings"].append(
                f"Found {len(invalid_rh)} relative humidity values outside valid range (0% - 100%)."
            )

    return report


def generate_synthetic_cattle_dataset(num_cows=8, days=21, seed=42):
    """
    Generates a scientifically grounded, multi-day synthetic dataset for student learning.
    
    IMPORTANT DISCLAIMER:
    This dataset is strictly SYNTHETIC, explicitly labeled with is_synthetic = 1,
    and designed for interface and algorithm demonstration. It does not replace
    empirical biological sensor trials.
    """
    np.random.seed(seed)
    random.seed(seed)

    # Pre-defined cows with diverse breeds
    sample_cows = [
        {"cow_id": "COW-101", "name": "Bella", "breed": "Holstein Friesian", "dob": "2021-03-15", "parity": 2},
        {"cow_id": "COW-102", "name": "Daisy", "breed": "Holstein Friesian", "dob": "2022-05-10", "parity": 1},
        {"cow_id": "COW-103", "name": "Ganga", "breed": "Gir", "dob": "2020-08-20", "parity": 3},
        {"cow_id": "COW-104", "name": "Kalyani", "breed": "Sahiwal", "dob": "2021-11-04", "parity": 2},
        {"cow_id": "COW-105", "name": "Buttercup", "breed": "Jersey", "dob": "2023-01-12", "parity": 0},  # Heifer
        {"cow_id": "COW-106", "name": "Radha", "breed": "Crossbred", "dob": "2021-07-25", "parity": 2},
        {"cow_id": "COW-107", "name": "Lucy", "breed": "Holstein Friesian", "dob": "2020-02-18", "parity": 4},
        {"cow_id": "COW-108", "name": "Yamuna", "breed": "Gir", "dob": "2022-09-05", "parity": 1},
    ][:num_cows]

    # Specific estrus event days for cows (typical 21-day cycle peak)
    estrus_schedule = {
        "COW-101": [18],
        "COW-102": [14],
        "COW-105": [9],
        "COW-106": [19],
    }

    # Specific health anomaly days
    # COW-103 develops mild mastitis on day 11-13 (fever, rumination drop, milk drop)
    # COW-107 experiences heat stress & lethargy on days 15-16
    records = []
    start_date = datetime.now() - timedelta(days=days)

    for cow in sample_cows:
        cid = cow["cow_id"]
        breed = cow["breed"]

        # Breed-specific baselines
        if "Holstein" in breed:
            base_activity = 280.0
            base_rumination = 490.0
            base_milk = 28.0
            base_temp = 38.6
        elif breed in ["Gir", "Sahiwal"]:
            base_activity = 240.0
            base_rumination = 460.0
            base_milk = 15.0
            base_temp = 38.5
        elif breed == "Jersey":
            base_activity = 260.0
            base_rumination = 475.0
            base_milk = 20.0
            base_temp = 38.6
        else:  # Crossbred
            base_activity = 265.0
            base_rumination = 470.0
            base_milk = 22.0
            base_temp = 38.6

        # Zero milk for heifers (parity = 0)
        if cow["parity"] == 0:
            base_milk = 0.0

        for d in range(days):
            current_date = start_date + timedelta(days=d)
            timestamp_str = current_date.strftime("%Y-%m-%d 10:00:00")

            # Ambient weather pattern: Warm afternoons with some fluctuating humidity
            # Warmer days on days 14-17 (simulating seasonal heatwave)
            if 14 <= d <= 17:
                amb_temp = round(float(np.random.normal(32.5, 1.2)), 1)
                amb_rh = round(float(np.random.normal(75.0, 4.0)), 1)
            else:
                amb_temp = round(float(np.random.normal(25.0, 2.0)), 1)
                amb_rh = round(float(np.random.normal(58.0, 5.0)), 1)

            amb_rh = min(100.0, max(20.0, amb_rh))
            thi = calculate_thi(amb_temp, amb_rh)

            # Check for simulated Estrus
            is_estrus = 1 if d in estrus_schedule.get(cid, []) else 0

            # Default healthy status
            health_status = "Healthy"

            # Check for simulated Health Anomalies
            is_mastitis = (cid == "COW-103" and d in [11, 12, 13])
            is_lethargy = (cid == "COW-107" and d in [15, 16])

            if is_mastitis:
                health_status = "Suspected Mastitis"
            elif is_lethargy:
                health_status = "Lethargy / Heat Stress"

            # Behavioral adjustments
            if is_estrus:
                # Estrus pattern: Big activity surge (+80% to +120%), rumination drops, milk drops slightly, temp rises +0.4°C
                act = base_activity * np.random.uniform(1.8, 2.2)
                rum = base_rumination * np.random.uniform(0.65, 0.75)
                lying = np.random.uniform(6.0, 8.0)  # Lying decreases during restlessness
                eating = np.random.uniform(180, 240)
                milk = base_milk * np.random.uniform(0.85, 0.92) if base_milk > 0 else 0.0
                btemp = base_temp + np.random.uniform(0.3, 0.5)
            elif is_mastitis:
                # Mastitis: Rumination crashes, high fever, sharp milk drop, depressed activity
                act = base_activity * np.random.uniform(0.6, 0.75)
                rum = base_rumination * np.random.uniform(0.5, 0.65)
                lying = np.random.uniform(12.0, 15.0)
                eating = np.random.uniform(140, 180)
                milk = base_milk * np.random.uniform(0.55, 0.70) if base_milk > 0 else 0.0
                btemp = base_temp + np.random.uniform(1.0, 1.4)  # Fever (39.6 - 40.0°C)
            elif is_lethargy:
                # Lethargy / Severe heat stress: Low activity, high lying, high breathing
                act = base_activity * np.random.uniform(0.55, 0.70)
                rum = base_rumination * np.random.uniform(0.70, 0.85)
                lying = np.random.uniform(13.0, 16.0)
                eating = np.random.uniform(150, 200)
                milk = base_milk * np.random.uniform(0.75, 0.85) if base_milk > 0 else 0.0
                btemp = base_temp + np.random.uniform(0.4, 0.8)
            else:
                # Normal baseline variation
                act = base_activity * np.random.normal(1.0, 0.08)
                rum = base_rumination * np.random.normal(1.0, 0.06)
                lying = np.random.uniform(9.5, 12.5)
                eating = np.random.uniform(220, 310)
                milk = base_milk * np.random.normal(1.0, 0.05) if base_milk > 0 else 0.0
                btemp = base_temp + np.random.normal(0.0, 0.12)

            records.append({
                "cow_id": cid,
                "timestamp": timestamp_str,
                "activity": round(float(act), 1),
                "rumination_time": round(float(rum), 1),
                "lying_time": round(float(lying), 1),
                "eating_time": round(float(eating), 1),
                "milk_yield": round(float(milk), 1),
                "body_temperature": round(float(btemp), 2),
                "ambient_temperature": amb_temp,
                "relative_humidity": amb_rh,
                "thi": thi,
                "estrus_label": is_estrus,
                "health_label": health_status,
                "is_synthetic": 1
            })

    cows_df = pd.DataFrame(sample_cows)
    observations_df = pd.DataFrame(records)

    return cows_df, observations_df
