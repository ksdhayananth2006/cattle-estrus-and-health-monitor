"""
BOVIGENIX - Module E: Health Risk Monitoring (pages/health_monitoring.py)
------------------------------------------------------------------------
Monitors physiological deviations (fever, rumination depression, acute milk drop,
excessive lying) using rule-based veterinary logic and supervised machine learning
when verified clinical labels exist.
"""

import streamlit as st
import pandas as pd
import numpy as np
import database
from models.health_model import evaluate_rule_based_health_risks, train_supervised_health_model

st.markdown("""
<div class="bovi-header">
    <h1>🩺 Cattle Health-Risk Assessment</h1>
    <p>Early screening for subclinical mastitis, ruminal acidosis, fever, and locomotion disorders</p>
</div>
""", unsafe_allow_html=True)

# Non-diagnostic legal and veterinary notice
st.markdown("""
<div class="disclaimer-badge">
    <strong>Veterinary Disclaimer:</strong> All alerts generated below represent <em>physiological risk indicators</em> designed to prompt veterinary inspection. 
    They do <strong>not</strong> constitute a definitive clinical diagnosis. Always confirm findings with a licensed veterinary practitioner.
</div>
""", unsafe_allow_html=True)

tab_scanner, tab_cow_eval, tab_supervised = st.tabs([
    "📡 Herd-Wide Health Risk Scanner",
    "🔍 Individual Cow Physiological Check",
    "🤖 Supervised Health ML Classifier"
])

# ---------------------------------------------------------------------------
# TAB 1: HERD-WIDE HEALTH SCANNER
# ---------------------------------------------------------------------------
with tab_scanner:
    st.subheader("Herd Health Anomaly Scanner")
    st.caption("Scans latest observations across the herd for acute drops in rumination, sudden fever, or sharp milk losses.")

    active_cows = database.get_all_cows(status_filter="Active")

    if active_cows.empty:
        st.info("No active cows found.")
    else:
        herd_alerts = []
        for _, cow in active_cows.iterrows():
            cid = cow["cow_id"]
            obs = database.get_observations(cow_id=cid, limit=14)

            if len(obs) >= 1:
                obs = obs.sort_values("timestamp")
                latest = obs.iloc[-1].to_dict()

                # Baseline from available history
                base = {
                    "rumination_time": obs["rumination_time"].mean(),
                    "milk_yield": obs["milk_yield"].mean(),
                    "activity": obs["activity"].mean()
                }

                cow_risks = evaluate_rule_based_health_risks(latest, base)
                for r in cow_risks:
                    r["cow_id"] = cid
                    r["name"] = cow["name"]
                    r["breed"] = cow["breed"]
                    r["timestamp"] = latest["timestamp"]
                    herd_alerts.append(r)

        if herd_alerts:
            st.warning(f"⚠️ Identified {len(herd_alerts)} physiological risk flag(s) requiring management attention!")
            
            for item in herd_alerts:
                sev = item["severity"]
                border_color = "#dc2626" if sev == "Critical" else ("#ea580c" if sev == "High" else "#eab308")
                
                with st.container():
                    st.markdown(f"""
                    <div style="border: 1px solid #cbd5e1; border-left: 5px solid {border_color}; background-color: #ffffff; color: #0f172a; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                        <div style="display:flex; justify-content:space-between; align-items:center;">
                            <div style="font-size:1.05rem; font-weight:700; color:#0f172a;">
                                {item['concern']} — <span style="color:#0f766e;">{item['cow_id']}</span> <span style="color:#64748b; font-weight:normal;">({item['name']})</span>
                            </div>
                            <span style="background-color:{border_color}; color:#ffffff; padding:3px 10px; border-radius:4px; font-size:0.78rem; font-weight:bold;">{item['severity']}</span>
                        </div>
                        <div style="font-size:0.90rem; color:#1e293b; margin-top:8px; line-height:1.5;">
                            <strong style="color:#0f172a;">Indicator:</strong> {item['indicator']}<br>
                            <strong style="color:#0f172a;">Physiological Explanation:</strong> {item['explanation']}<br>
                            <strong style="color:#0f172a;">Recommended Action:</strong> {item['action']}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if st.button(f"Log Alert to System for {item['cow_id']}", key=f"btn_alert_{item['cow_id']}_{item['concern']}"):
                        database.add_alert(
                            cow_id=item["cow_id"],
                            alert_type="Health Risk",
                            severity=item["severity"],
                            message=f"{item['concern']}: {item['indicator']}",
                            suggested_action=item["action"]
                        )
                        st.success(f"Health alert logged for {item['cow_id']}!")
        else:
            st.success("✅ All monitored cattle are currently within normal physiological thresholds.")


# ---------------------------------------------------------------------------
# TAB 2: INDIVIDUAL COW PHYSIOLOGICAL CHECK
# ---------------------------------------------------------------------------
with tab_cow_eval:
    st.subheader("Individual Cow Physiological Profile")

    cow_ids = active_cows["cow_id"].tolist() if not active_cows.empty else []
    if cow_ids:
        selected_cid = st.selectbox("Select Cow:", cow_ids, key="health_cid_sel")
        c_obs = database.get_observations(cow_id=selected_cid, limit=20)

        if not c_obs.empty:
            c_obs = c_obs.sort_values("timestamp")
            latest_row = c_obs.iloc[-1].to_dict()

            st.write(f"**Latest Observation:** `{latest_row['timestamp']}`")

            # Reference Table
            ref_data = [
                {"Parameter": "Body Temperature", "Normal Reference Range": "38.0 - 39.2 °C", "Current Value": f"{latest_row.get('body_temperature', 'N/A')} °C"},
                {"Parameter": "Daily Rumination", "Normal Reference Range": "400 - 550 min/day", "Current Value": f"{latest_row.get('rumination_time', 'N/A')} min"},
                {"Parameter": "Lying / Rest Time", "Normal Reference Range": "9.0 - 13.0 hrs/day", "Current Value": f"{latest_row.get('lying_time', 'N/A')} hrs"},
                {"Parameter": "Milk Yield", "Normal Reference Range": "Breed Dependent", "Current Value": f"{latest_row.get('milk_yield', 'N/A')} L"},
                {"Parameter": "Eating Time", "Normal Reference Range": "200 - 320 min/day", "Current Value": f"{latest_row.get('eating_time', 'N/A')} min"},
            ]
            st.dataframe(pd.DataFrame(ref_data), use_container_width=True, hide_index=True)

            # Evaluate specific risks for this cow
            base = {
                "rumination_time": c_obs["rumination_time"].mean(),
                "milk_yield": c_obs["milk_yield"].mean(),
                "activity": c_obs["activity"].mean()
            }
            cow_alerts = evaluate_rule_based_health_risks(latest_row, base)

            if cow_alerts:
                st.write("**Specific Concerns Detected:**")
                for a in cow_alerts:
                    st.error(f"**{a['concern']} ({a['severity']}):** {a['indicator']}\n\n*Action:* {a['action']}")
            else:
                st.success("No physiological deviations detected for this cow.")
        else:
            st.info(f"No sensor observations recorded for Cow {selected_cid}.")


# ---------------------------------------------------------------------------
# TAB 3: SUPERVISED HEALTH ML CLASSIFIER
# ---------------------------------------------------------------------------
with tab_supervised:
    st.subheader("Supervised Health Classification Model")
    st.markdown("""
    Supervised learning requires **verified clinical health labels** (e.g. Mastitis, Ketosis, Lameness confirmed by a veterinarian).
    """)

    all_obs = database.get_observations_for_ml()
    has_health_labels = "health_label" in all_obs.columns and not all_obs["health_label"].dropna().empty

    if has_health_labels:
        label_counts = all_obs["health_label"].value_counts().to_dict()
        st.write(f"Found **{len(all_obs.dropna(subset=['health_label']))}** labeled records across classes:", label_counts)

        if st.button("Train Supervised Health Random Forest", type="primary"):
            with st.spinner("Training supervised health model..."):
                health_res, msg = train_supervised_health_model(all_obs)
                if health_res:
                    st.success(f"Model trained successfully! Test Accuracy: **{health_res['accuracy'] * 100:.1f}%**")
                    st.write("**Target Classes:**", health_res["classes"])
                else:
                    st.warning(msg)
    else:
        st.info("No verified health labels are present in the current dataset. In accordance with ethical machine learning standards, supervised classification is paused and transparent rule-based physiological screening is utilized.")
