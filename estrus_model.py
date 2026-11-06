"""
BOVIGENIX - Estrus Prediction Models (models/estrus_model.py)
-------------------------------------------------------------
This module implements machine-learning models for estrus detection in dairy cows.

Educational Concepts for Students:
1. Supervised Learning: The algorithm learns from historical examples where
   estrus was confirmed (label = 1) or absent (label = 0).
2. Random Forest: An ensemble of multiple decision trees. Each tree votes on
   whether a cow is in estrus. It is robust, handles non-linear relationships,
   and resists overfitting.
3. Feature Engineering: Creating new informative variables from raw sensor data,
   such as ratio of current activity to the cow's 7-day normal baseline.
4. Cow-Wise Train/Test Split: Splitting data by cow ID rather than random rows
   prevents 'data leakage' (the model memorizing an individual cow's unique habits).
5. Two Approaches Implemented:
   - Approach A: Basic Behavioural Model (sensors only)
   - Approach B: Individualized & Context-Aware Model (sensors + baseline deviations + breed + age + THI)
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
from utils.calculations import calculate_age, calculate_thi, calculate_percentage_deviation


def engineer_individualized_features(df):
    """
    Computes individualized baseline features for each cow:
    - 7-day rolling average for activity, rumination, milk, and body temperature.
    - Ratio of current reading to the rolling average.
    - Age in years calculated from Date of Birth.
    """
    df_sorted = df.sort_values(by=["cow_id", "timestamp"]).copy()

    # Calculate cow age if DOB is available
    if "dob" in df_sorted.columns:
        df_sorted["age_years"] = df_sorted["dob"].apply(lambda d: calculate_age(d)[0])
    else:
        df_sorted["age_years"] = 3.0  # Default assumption if missing

    # Compute rolling 7-day baselines per cow
    # (For daily observations, window=7 represents the previous week)
    df_sorted["base_activity"] = df_sorted.groupby("cow_id")["activity"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_sorted["base_rumination"] = df_sorted.groupby("cow_id")["rumination_time"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_sorted["base_milk"] = df_sorted.groupby("cow_id")["milk_yield"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )
    df_sorted["base_body_temp"] = df_sorted.groupby("cow_id")["body_temperature"].transform(
        lambda s: s.rolling(window=7, min_periods=1).mean()
    )

    # Calculate Individualized Ratios (Deviation from personal baseline)
    # E.g. activity_ratio = 1.8 means activity is 80% higher than usual
    df_sorted["activity_ratio"] = (df_sorted["activity"] / (df_sorted["base_activity"] + 1e-5)).round(3)
    df_sorted["rumination_ratio"] = (df_sorted["rumination_time"] / (df_sorted["base_rumination"] + 1e-5)).round(3)
    df_sorted["milk_ratio"] = (df_sorted["milk_yield"] / (df_sorted["base_milk"] + 1e-5)).round(3)
    df_sorted["temp_deviation"] = (df_sorted["body_temperature"] - df_sorted["base_body_temp"]).round(2)

    # Encode Breed as numeric category
    breed_map = {
        "Holstein Friesian": 1,
        "Jersey": 2,
        "Gir": 3,
        "Sahiwal": 4,
        "Crossbred": 5
    }
    if "breed" in df_sorted.columns:
        df_sorted["breed_code"] = df_sorted["breed"].map(lambda b: breed_map.get(b, 0))
    else:
        df_sorted["breed_code"] = 0

    return df_sorted


def train_estrus_models(df):
    """
    Trains and compares:
    1. Basic Behavioural Model (Raw Sensors)
    2. Individualized & Context-Aware Model (Sensors + Baseline Ratios + Breed + Age + THI)

    Returns:
    - dict with keys: 'basic_model', 'individual_model', 'metrics_df',
      'basic_importances', 'individual_importances', 'cm_basic', 'cm_individual'
    """
    # 1. Filter dataset to rows with verified estrus labels (0 or 1)
    if "estrus_label" not in df.columns or df["estrus_label"].dropna().empty:
        return None, "No verified estrus labels (0 or 1) found in the dataset for training."

    df_labeled = df.dropna(subset=["estrus_label"]).copy()
    df_labeled["estrus_label"] = df_labeled["estrus_label"].astype(int)

    # Check class distribution
    class_counts = df_labeled["estrus_label"].value_counts().to_dict()
    if len(class_counts) < 2:
        return None, f"Dataset must contain both positive (1) and negative (0) estrus labels. Found only: {class_counts}"

    # 2. Engineer individualized features
    df_feat = engineer_individualized_features(df_labeled)

    # 3. Define feature sets
    basic_features = [
        "activity", "rumination_time", "lying_time",
        "eating_time", "milk_yield", "body_temperature"
    ]
    # Retain only features actually present in the dataset
    basic_features = [f for f in basic_features if f in df_feat.columns]

    individual_features = basic_features + [
        "activity_ratio", "rumination_ratio", "milk_ratio", "temp_deviation",
        "age_years", "parity", "breed_code", "thi"
    ]
    individual_features = [f for f in individual_features if f in df_feat.columns]

    # Fill any remaining NaNs in features with column median
    for col in individual_features:
        if df_feat[col].isnull().any():
            df_feat[col] = df_feat[col].fillna(df_feat[col].median())

    y = df_feat["estrus_label"]

    # 4. Cow-wise or Stratified Train/Test Split
    # Convert unique cows to a standard python list to ensure compatibility
    unique_cows = list(df_feat["cow_id"].unique())
    if len(unique_cows) >= 4:
        cows_train, cows_test = train_test_split(unique_cows, test_size=0.25, random_state=42)
        train_mask = df_feat["cow_id"].isin(cows_train).to_numpy()
        test_mask = df_feat["cow_id"].isin(cows_test).to_numpy()

        y_train_check = y[train_mask]
        y_test_check = y[test_mask]

        # Verify that both train and test have both classes (0 and 1)
        if len(np.unique(y_train_check)) < 2 or len(np.unique(y_test_check)) < 2:
            # Fall back to stratified split if one subset missed the minority class
            indices = np.arange(len(df_feat))
            train_idx, test_idx = train_test_split(
                indices, test_size=0.25, random_state=42, stratify=y.to_numpy()
            )
            X_train_basic = df_feat.iloc[train_idx][basic_features]
            X_test_basic = df_feat.iloc[test_idx][basic_features]
            X_train_indiv = df_feat.iloc[train_idx][individual_features]
            X_test_indiv = df_feat.iloc[test_idx][individual_features]
            y_train = y.iloc[train_idx]
            y_test = y.iloc[test_idx]
        else:
            X_train_basic = df_feat[train_mask][basic_features]
            X_test_basic = df_feat[test_mask][basic_features]
            X_train_indiv = df_feat[train_mask][individual_features]
            X_test_indiv = df_feat[test_mask][individual_features]
            y_train = y[train_mask]
            y_test = y[test_mask]
    else:
        # Standard stratified train-test split for small cow populations
        indices = np.arange(len(df_feat))
        train_idx, test_idx = train_test_split(
            indices, test_size=0.25, random_state=42, stratify=y.to_numpy()
        )
        X_train_basic = df_feat.iloc[train_idx][basic_features]
        X_test_basic = df_feat.iloc[test_idx][basic_features]
        X_train_indiv = df_feat.iloc[train_idx][individual_features]
        X_test_indiv = df_feat.iloc[test_idx][individual_features]
        y_train = y.iloc[train_idx]
        y_test = y.iloc[test_idx]

    # 5. Train Model 1: Basic Behavioural Random Forest
    rf_basic = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    rf_basic.fit(X_train_basic, y_train)
    y_pred_basic = rf_basic.predict(X_test_basic)

    # Metrics for Basic Model
    acc_basic = accuracy_score(y_test, y_pred_basic)
    prec_basic = precision_score(y_test, y_pred_basic, zero_division=0)
    rec_basic = recall_score(y_test, y_pred_basic, zero_division=0)
    f1_basic = f1_score(y_test, y_pred_basic, zero_division=0)
    cm_basic = confusion_matrix(y_test, y_pred_basic)

    # 6. Train Model 2: Individualized Context-Aware Random Forest
    rf_indiv = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42, class_weight="balanced")
    rf_indiv.fit(X_train_indiv, y_train)
    y_pred_indiv = rf_indiv.predict(X_test_indiv)

    # Metrics for Individualized Model
    acc_indiv = accuracy_score(y_test, y_pred_indiv)
    prec_indiv = precision_score(y_test, y_pred_indiv, zero_division=0)
    rec_indiv = recall_score(y_test, y_pred_indiv, zero_division=0)
    f1_indiv = f1_score(y_test, y_pred_indiv, zero_division=0)
    cm_indiv = confusion_matrix(y_test, y_pred_indiv)

    # Comparison DataFrame
    metrics_df = pd.DataFrame([
        {
            "Model": "1. Basic Behavioural",
            "Accuracy": round(acc_basic, 3),
            "Precision": round(prec_basic, 3),
            "Recall": round(rec_basic, 3),
            "F1-Score": round(f1_basic, 3),
            "Features Count": len(basic_features)
        },
        {
            "Model": "2. Individualized & Context-Aware",
            "Accuracy": round(acc_indiv, 3),
            "Precision": round(prec_indiv, 3),
            "Recall": round(rec_indiv, 3),
            "F1-Score": round(f1_indiv, 3),
            "Features Count": len(individual_features)
        }
    ])

    results = {
        "basic_model": rf_basic,
        "individual_model": rf_indiv,
        "basic_features": basic_features,
        "individual_features": individual_features,
        "metrics_df": metrics_df,
        "basic_importances": (basic_features, rf_basic.feature_importances_),
        "individual_importances": (individual_features, rf_indiv.feature_importances_),
        "cm_basic": cm_basic,
        "cm_individual": cm_indiv,
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "class_distribution": class_counts
    }

    return results, "Model training and evaluation completed successfully."


def predict_estrus_for_observation(model, feature_names, obs_dict):
    """
    Given a single cow observation dictionary and trained Random Forest model:
    Returns (prediction_label, probability_pct, contributing_features)
    """
    row_values = []
    for f in feature_names:
        row_values.append(obs_dict.get(f, 0.0))

    X_sample = pd.DataFrame([row_values], columns=feature_names)
    prob = model.predict_proba(X_sample)[0][1]  # Probability of class 1 (Estrus)
    pred = 1 if prob >= 0.50 else 0

    return pred, round(prob * 100, 1)
