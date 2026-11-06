"""
BOVIGENIX - Health Risk Monitoring (models/health_model.py)
-----------------------------------------------------------
This module implements health risk assessment for dairy cattle.

Biotechnology & Veterinary Science Principles:
1. Decision Support vs. Diagnosis:
   - Sensor deviations identify *risk flags* (e.g. fever, acute rumination drop).
   - These flags assist farm managers and veterinarians; they NEVER replace
     veterinary clinical diagnosis (auscultation, palpation, blood chemistry).
2. Rule-Based Deviations:
   - Transparent, explainable thresholds based on established dairy physiology:
     * Fever: Body temperature >= 39.5°C (normal: 38.0 - 39.2°C).
     * Rumination Crash: Drop > 30% below cow's baseline (sign of acidosis or acute infection).
     * Milk Drop: Drop > 25% below cow's baseline (sign of mastitis or systemic illness).
     * Lethargy / Lameness: Drop in walking activity combined with excessive lying (> 14 hrs).
3. Supervised Machine Learning:
   - Activated only when verified clinical health labels exist in the dataset.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report


def evaluate_rule_based_health_risks(obs, baseline=None):
    """
    Evaluates physiological sensor parameters against established veterinary thresholds
    and the individual cow's baseline.

    Parameters:
    - obs (dict or Series): Current sensor observation values.
    - baseline (dict, optional): Normal baseline values for this cow.

    Returns:
    - list of dicts: Identified health concerns with explanations and guidance.
    """
    alerts = []

    # Extract current values safely
    body_temp = obs.get("body_temperature")
    rumination = obs.get("rumination_time")
    milk = obs.get("milk_yield")
    activity = obs.get("activity")
    lying = obs.get("lying_time")

    # 1. Hyperthermia / Fever check
    if body_temp is not None and not np.isnan(body_temp):
        if body_temp >= 40.0:
            alerts.append({
                "severity": "Critical",
                "concern": "High Fever (Severe Hyperthermia)",
                "indicator": f"Body Temp: {body_temp:.1f} °C (Physiological Norm: 38.0 - 39.2 °C)",
                "explanation": "Severe elevation in core body temperature indicating acute systemic infection or heat stroke.",
                "action": "Immediate veterinary consultation recommended. Check for acute coliform mastitis, pneumonia, or metritis."
            })
        elif body_temp >= 39.5:
            alerts.append({
                "severity": "Warning",
                "concern": "Mild Fever / Elevated Temperature",
                "indicator": f"Body Temp: {body_temp:.1f} °C (Physiological Norm: 38.0 - 39.2 °C)",
                "explanation": "Moderate body temperature elevation. Could reflect early inflammatory response or thermal stress.",
                "action": "Monitor body temperature every 6 hours. Examine udder health and somatic cell counts."
            })

    # 2. Rumination Crash check
    base_rum = baseline.get("rumination_time", 480.0) if baseline else 480.0
    if rumination is not None and not np.isnan(rumination) and base_rum > 0:
        pct_rum_drop = ((base_rum - rumination) / base_rum) * 100
        if pct_rum_drop >= 35.0:
            alerts.append({
                "severity": "High",
                "concern": "Acute Rumination Depression",
                "indicator": f"Rumination: {rumination:.0f} min/day (-{pct_rum_drop:.1f}% below baseline {base_rum:.0f} min)",
                "explanation": "Sudden cessation or severe drop in rumination is a sensitive early marker for subacute ruminal acidosis (SARA), abomasal displacement, or pain.",
                "action": "Inspect rumen fill and motility. Check ration for fiber adequacy and buffer availability."
            })

    # 3. Sudden Milk Yield Loss check
    base_milk = baseline.get("milk_yield", 22.0) if baseline else 22.0
    if milk is not None and not np.isnan(milk) and base_milk > 5.0:  # Ignore dry cows / heifers
        pct_milk_drop = ((base_milk - milk) / base_milk) * 100
        if pct_milk_drop >= 25.0:
            alerts.append({
                "severity": "Warning",
                "concern": "Acute Milk Yield Drop",
                "indicator": f"Milk Yield: {milk:.1f} L (-{pct_milk_drop:.1f}% drop from baseline {base_milk:.1f} L)",
                "explanation": "Sudden milk depression frequently accompanies clinical mastitis or metabolic crisis.",
                "action": "Perform California Mastitis Test (CMT) on all quarters. Check for clinical udder swelling or milk clots."
            })

    # 4. Lethargy / Potential Lameness check
    base_act = baseline.get("activity", 260.0) if baseline else 260.0
    if activity is not None and not np.isnan(activity) and base_act > 0:
        pct_act_drop = ((base_act - activity) / base_act) * 100
        if pct_act_drop >= 35.0 and (lying is not None and lying >= 13.5):
            alerts.append({
                "severity": "Warning",
                "concern": "Severe Lethargy & Prolonged Lying",
                "indicator": f"Activity: -{pct_act_drop:.1f}%, Lying Time: {lying:.1f} hrs/day",
                "explanation": "Depressed movement coupled with excessive recumbency suggests digital dermatitis, claw lesions, or systemic fatigue.",
                "action": "Inspect locomotion and hooves in the trimming chute. Check for digital dermatitis or sole ulcers."
            })

    return alerts


def train_supervised_health_model(df):
    """
    Trains a multi-class Random Forest classifier if verified health labels exist
    (e.g., 'Healthy', 'Mastitis', 'Ketosis', 'Lameness').
    """
    if "health_label" not in df.columns or df["health_label"].dropna().empty:
        return None, "No verified health labels found in the dataset. Supervised ML is inactive; rule-based monitoring remains active."

    df_labeled = df.dropna(subset=["health_label"]).copy()
    valid_counts = df_labeled["health_label"].value_counts()

    if len(valid_counts) < 2:
        return None, f"At least 2 distinct health classes are required for ML training. Found: {valid_counts.to_dict()}"

    feature_cols = [
        "activity", "rumination_time", "lying_time",
        "eating_time", "milk_yield", "body_temperature"
    ]
    feature_cols = [f for f in feature_cols if f in df_labeled.columns]

    X = df_labeled[feature_cols].fillna(df_labeled[feature_cols].median())
    y = df_labeled["health_label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    rf_health = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    rf_health.fit(X_train, y_train)
    y_pred = rf_health.predict(X_test)

    acc = accuracy_score(y_test, y_pred)

    return {
        "model": rf_health,
        "features": feature_cols,
        "accuracy": round(acc, 3),
        "classes": list(rf_health.classes_),
        "importances": (feature_cols, rf_health.feature_importances_)
    }, "Supervised health model trained successfully."
