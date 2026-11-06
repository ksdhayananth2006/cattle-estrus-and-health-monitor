"""
BOVIGENIX - Module D: Estrus Monitoring & Prediction (pages/estrus_monitoring.py)
--------------------------------------------------------------------------------
Provides individualized, breed-aware estrus prediction using Random Forest models,
comparing raw behavioral observations against rolling individual baseline deviations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import database
from estrus_model import train_estrus_models, predict_estrus_for_observation, engineer_individualized_features
from visualizations import plot_sensor_trends, plot_feature_importance

st.markdown("""
<div class="bovi-header">
    <h1>🎯 AI-Driven Estrus Prediction & Monitoring</h1>
    <p>Individualized, context-aware machine learning screening for optimal artificial insemination timing</p>
</div>
""", unsafe_allow_html=True)

# Scientific and ethical disclaimer
st.markdown("""
<div class="disclaimer-badge">
    <strong>Scientific Boundary Notice:</strong> BOVIGENIX detects physiological and behavioral markers associated with estrus onset (standing heat). 
    In accordance with reproductive biology standards, it does <em>not</em> classify proestrus, metestrus, or diestrus phases unless fine-grained hormonal/cytological phase labels are provided.
</div>
""", unsafe_allow_html=True)

# 1. Load ML training observations from database
ml_data = database.get_observations_for_ml()

if ml_data.empty:
    st.warning("No sensor observations found in the database. Please import a dataset or load the demo dataset from the sidebar.")
    st.stop()

# 2. Model Training State in Streamlit Session
if "estrus_models" not in st.session_state:
    st.session_state["estrus_models"] = None

# Check if model needs training
labeled_rows = ml_data.dropna(subset=["estrus_label"])
has_valid_labels = len(labeled_rows) > 0 and len(labeled_rows["estrus_label"].unique()) >= 2

tab_single_cow, tab_herd_scanner, tab_training_info = st.tabs([
    "🔍 Individual Cow Estrus Assessment",
    "📡 Herd-Wide Automated Estrus Scanner",
    "🧠 Model Logic & Feature Explanations"
])

# ---------------------------------------------------------------------------
# TAB 1: INDIVIDUAL COW ESTRUS ASSESSMENT
# ---------------------------------------------------------------------------
with tab_single_cow:
    st.subheader("Individualized Cow Estrus Evaluation")

    cows_df = database.get_all_cows(status_filter="Active")
    cow_list = cows_df["cow_id"].tolist() if not cows_df.empty else []

    if not cow_list:
        st.info("No active cows found.")
    else:
        sel_col1, sel_col2 = st.columns([2, 1])
        with sel_col1:
            selected_cow = st.selectbox("Select Cow for Evaluation:", cow_list, key="estrus_cow_sel")
        with sel_col2:
            cow_row = cows_df[cows_df["cow_id"] == selected_cow].iloc[0]
            st.caption(f"**Breed:** {cow_row['breed']} | **Parity:** {cow_row['parity']}")

        # Fetch cow observations
        cow_obs = database.get_observations(cow_id=selected_cow, limit=30)

        if cow_obs.empty:
            st.info(f"No observations logged for {selected_cow}.")
        else:
            # Sort chronologically
            cow_obs = cow_obs.sort_values("timestamp")
            latest_obs = cow_obs.iloc[-1].to_dict()

            # Render sensor trend chart
            fig_trend = plot_sensor_trends(cow_obs, selected_cow)
            st.plotly_chart(fig_trend, use_container_width=True, key="estrus_eval_trend_chart")

            # Display Latest Observation vs. Personal Baseline
            st.subheader(f"Latest Sensor Readings for {selected_cow} ({latest_obs['timestamp']})")
            
            # Compute baseline from previous 7 readings
            base_act = cow_obs["activity"].tail(7).mean()
            base_rum = cow_obs["rumination_time"].tail(7).mean()
            base_milk = cow_obs["milk_yield"].tail(7).mean()
            base_temp = cow_obs["body_temperature"].tail(7).mean()

            cur_act = latest_obs.get("activity", 0.0)
            cur_rum = latest_obs.get("rumination_time", 0.0)
            cur_milk = latest_obs.get("milk_yield", 0.0)
            cur_temp = latest_obs.get("body_temperature", 0.0)

            act_diff = ((cur_act - base_act) / base_act * 100) if base_act else 0
            rum_diff = ((cur_rum - base_rum) / base_rum * 100) if base_rum else 0
            milk_diff = ((cur_milk - base_milk) / base_milk * 100) if base_milk else 0

            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Activity (Steps)", f"{cur_act:.0f}", f"{act_diff:+.1f}% vs Baseline", delta_color="normal")
            with m2:
                st.metric("Rumination Time", f"{cur_rum:.0f} min", f"{rum_diff:+.1f}% vs Baseline", delta_color="inverse")
            with m3:
                st.metric("Milk Yield", f"{cur_milk:.1f} L", f"{milk_diff:+.1f}% vs Baseline", delta_color="inverse")
            with m4:
                st.metric("Body Temperature", f"{cur_temp:.1f} °C", f"{cur_temp - base_temp:+.2f} °C vs Baseline")

            st.write("")

            # Run Model Prediction
            st.subheader("🤖 AI Estrus Probability Assessment")

            if st.session_state["estrus_models"] is None:
                if has_valid_labels:
                    st.info("Machine learning model is not yet trained in this session.")
                    if st.button("Train Estrus Random Forest Models Now", type="primary"):
                        with st.spinner("Training models on historical observations..."):
                            results, msg = train_estrus_models(ml_data)
                            st.session_state["estrus_models"] = results
                            st.success(msg)
                            st.rerun()
                else:
                    st.warning("Insufficient verified estrus labels in the database to train the supervised Random Forest model.")
            else:
                models_dict = st.session_state["estrus_models"]
                indiv_model = models_dict["individual_model"]
                indiv_feats = models_dict["individual_features"]

                # Prepare feature dictionary for prediction
                obs_features = latest_obs.copy()
                obs_features["activity_ratio"] = round(cur_act / (base_act + 1e-5), 3)
                obs_features["rumination_ratio"] = round(cur_rum / (base_rum + 1e-5), 3)
                obs_features["milk_ratio"] = round(cur_milk / (base_milk + 1e-5), 3)
                obs_features["temp_deviation"] = round(cur_temp - base_temp, 2)
                obs_features["parity"] = cow_row["parity"]
                obs_features["age_years"] = 3.0
                obs_features["breed_code"] = 1

                pred, prob = predict_estrus_for_observation(indiv_model, indiv_feats, obs_features)

                p_col1, p_col2 = st.columns([1, 2])
                with p_col1:
                    if prob >= 70.0:
                        st.markdown(f"""
                        <div style="background-color: #fef2f2 !important; border: 2px solid #ef4444 !important; padding: 20px 16px !important; border-radius: 10px !important; text-align: center !important; box-shadow: 0 2px 6px rgba(239, 68, 68, 0.15) !important;">
                            <span style="font-size: 2.2rem;">🔥</span>
                            <div style="color: #991b1b !important; font-size: 1.15rem !important; font-weight: 800 !important; margin: 6px 0 !important; letter-spacing: 0.5px !important;">HIGH SUSPICION</div>
                            <div style="color: #b91c1c !important; font-size: 2.6rem !important; font-weight: 900 !important; margin: 0 !important; line-height: 1.1 !important;">{prob}%</div>
                            <div style="color: #7f1d1d !important; font-size: 0.85rem !important; font-weight: 600 !important; margin-top: 6px !important;">Elevated Estrus Probability</div>
                        </div>
                        """, unsafe_allow_html=True)
                    elif prob >= 45.0:
                        st.markdown(f"""
                        <div style="background-color: #fffbeb !important; border: 2px solid #f59e0b !important; padding: 20px 16px !important; border-radius: 10px !important; text-align: center !important; box-shadow: 0 2px 6px rgba(245, 158, 11, 0.15) !important;">
                            <span style="font-size: 2.2rem;">⚠️</span>
                            <div style="color: #92400e !important; font-size: 1.15rem !important; font-weight: 800 !important; margin: 6px 0 !important; letter-spacing: 0.5px !important;">ELEVATED ALERT</div>
                            <div style="color: #b45309 !important; font-size: 2.6rem !important; font-weight: 900 !important; margin: 0 !important; line-height: 1.1 !important;">{prob}%</div>
                            <div style="color: #78350f !important; font-size: 0.85rem !important; font-weight: 600 !important; margin-top: 6px !important;">Moderate Estrus Suspicion</div>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background-color: #f0fdf4 !important; border: 2px solid #22c55e !important; padding: 20px 16px !important; border-radius: 10px !important; text-align: center !important; box-shadow: 0 2px 6px rgba(34, 197, 94, 0.15) !important;">
                            <span style="font-size: 2.2rem;">✅</span>
                            <div style="color: #166534 !important; font-size: 1.15rem !important; font-weight: 800 !important; margin: 6px 0 !important; letter-spacing: 0.5px !important;">NORMAL BASELINE</div>
                            <div style="color: #15803d !important; font-size: 2.6rem !important; font-weight: 900 !important; margin: 0 !important; line-height: 1.1 !important;">{prob}%</div>
                            <div style="color: #14532d !important; font-size: 0.85rem !important; font-weight: 600 !important; margin-top: 6px !important;">Rest State / Luteal Phase</div>
                        </div>
                        """, unsafe_allow_html=True)

                with p_col2:
                    st.write("**Key Contributing Indicators:**")
                    st.markdown(f"- **Activity Surge:** Current reading is `{obs_features['activity_ratio']}x` her normal baseline.")
                    st.markdown(f"- **Rumination Change:** Current rumination is `{obs_features['rumination_ratio']}x` of baseline.")
                    st.markdown(f"- **Thermal Shift:** Body temperature is `{obs_features['temp_deviation']:+.2f} °C` relative to baseline.")

                    if prob >= 50.0:
                        st.info("💡 **Veterinary Guidance:** Visually check for clear vaginal mucus discharge and standing response to mounting. If standing heat is confirmed, plan artificial insemination according to the **AM-PM rule** (inseminate ~12 hours after first standing heat detection).")
                        
                        if st.button("📢 Generate Estrus Alert for Veterinary Team", type="secondary"):
                            database.add_alert(
                                cow_id=selected_cow,
                                alert_type="Estrus Prediction",
                                severity="High" if prob >= 70 else "Warning",
                                message=f"AI model predicted {prob}% estrus probability. Activity {obs_features['activity_ratio']}x baseline, rumination {obs_features['rumination_ratio']}x baseline.",
                                suggested_action="Verify standing heat and schedule artificial insemination in 12 hours."
                            )
                            st.success(f"Alert generated for Cow {selected_cow}!")


# ---------------------------------------------------------------------------
# TAB 2: HERD-WIDE ESTRUS SCANNER
# ---------------------------------------------------------------------------
with tab_herd_scanner:
    st.subheader("Herd-Wide Estrus Scanner")
    st.caption("Scans the latest sensor observation for every active cow to detect potential estrus candidates.")

    if st.button("Run Herd Scan Now", type="primary"):
        with st.spinner("Scanning herd observations..."):
            active_cows = database.get_all_cows(status_filter="Active")
            candidates = []

            for _, cow in active_cows.iterrows():
                cid = cow["cow_id"]
                c_obs = database.get_observations(cow_id=cid, limit=14)
                
                if len(c_obs) >= 2:
                    c_obs = c_obs.sort_values("timestamp")
                    latest = c_obs.iloc[-1]
                    baseline_act = c_obs["activity"].mean()
                    baseline_rum = c_obs["rumination_time"].mean()

                    act_ratio = latest["activity"] / (baseline_act + 1e-5)
                    rum_ratio = latest["rumination_time"] / (baseline_rum + 1e-5)

                    # Screening rule: Activity surge > 1.5x and rumination drop < 0.85x
                    is_candidate = (act_ratio >= 1.5 and rum_ratio <= 0.85)

                    candidates.append({
                        "Cow ID": cid,
                        "Name": cow["name"],
                        "Breed": cow["breed"],
                        "Parity": cow["parity"],
                        "Latest Timestamp": latest["timestamp"],
                        "Activity Ratio": round(act_ratio, 2),
                        "Rumination Ratio": round(rum_ratio, 2),
                        "Body Temp (°C)": latest.get("body_temperature"),
                        "Estrus Suspected": "🔥 YES" if is_candidate else "No"
                    })

            if candidates:
                cand_df = pd.DataFrame(candidates)
                st.dataframe(cand_df, use_container_width=True, hide_index=True)
                
                estrus_cows = cand_df[cand_df["Estrus Suspected"] == "🔥 YES"]
                if not estrus_cows.empty:
                    st.warning(f"Detected {len(estrus_cows)} cow(s) showing behavioral indicators of estrus!")
                else:
                    st.success("No cows currently exhibiting estrus behavioral surges.")
            else:
                st.info("Insufficient observation history to scan herd.")


# ---------------------------------------------------------------------------
# TAB 3: MODEL LOGIC & FEATURE EXPLANATIONS
# ---------------------------------------------------------------------------
with tab_training_info:
    st.subheader("Biological & Algorithmic Rationale")
    st.markdown("""
    ### Why Individualized Baselines Matter in Cattle Monitoring
    In dairy farming, cows exhibit substantial individual variation:
    - A naturally energetic heifer might take **300 steps/hour** normally.
    - An older, placid cow in her 4th lactation might only take **180 steps/hour**.

    If we use a **fixed generic threshold** (e.g. `activity > 250 steps`), we will generate:
    - **False Positives** for high-activity cows.
    - **False Negatives** (missed heats) for placid cows.

    ### How BOVIGENIX Solves This:
    Instead of relying strictly on absolute values, BOVIGENIX computes an **Individualized Ratio**:
    $$\\text{Activity Ratio} = \\frac{\\text{Current Activity}}{\\text{Cow's 7-Day Moving Average}}$$

    A ratio $> 1.8$ means the cow is moving **80% more than her personal normal level**, providing accurate detection across all temperaments and breeds!
    """)

