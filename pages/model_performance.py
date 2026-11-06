"""
BOVIGENIX - Module I: Model Performance & Comparative Analytics (pages/model_performance.py)
---------------------------------------------------------------------------------------------
Evaluates and benchmarks machine learning algorithms:
- Model 1: Basic Behavioural Random Forest (Raw sensor values)
- Model 2: Individualized Context-Aware Random Forest (Sensors + Baseline Deviations + Breed + Age + THI)
"""

import streamlit as st
import pandas as pd
import numpy as np
import database
from models.estrus_model import train_estrus_models
from utils.visualizations import (
    plot_model_comparison,
    plot_confusion_matrix_heatmap,
    plot_feature_importance
)

st.markdown("""
<div class="bovi-header">
    <h1>📈 Machine Learning Evaluation & Comparative Benchmarking</h1>
    <p>Empirical performance metrics, confusion matrices, and feature importance rankings</p>
</div>
""", unsafe_allow_html=True)

# 1. Fetch ML observations from database
ml_data = database.get_observations_for_ml()

if ml_data.empty:
    st.warning("No observations found in the database. Please import a dataset or load the demo dataset from the sidebar.")
    st.stop()

# Check verified labels
labeled_subset = ml_data.dropna(subset=["estrus_label"])
valid_classes = labeled_subset["estrus_label"].unique()

st.sidebar.markdown("### Model Controls")

tab_benchmark, tab_matrices, tab_importance, tab_viva = st.tabs([
    "📊 Comparative Benchmark",
    "🔲 Confusion Matrices",
    "🌟 Feature Importance",
    "🎓 Student Viva Guide"
])

# ---------------------------------------------------------------------------
# TAB 1: COMPARATIVE BENCHMARK
# ---------------------------------------------------------------------------
with tab_benchmark:
    st.subheader("Model 1 (Basic Behavioural) vs. Model 2 (Individualized & Context-Aware)")

    if len(labeled_subset) == 0 or len(valid_classes) < 2:
        st.error(f"""
        **Insufficient Verified Training Data:**
        - Supervised Random Forest training requires both positive (1 = Estrus) and negative (0 = No Estrus) labels.
        - Currently found: **{len(labeled_subset)}** labeled records with classes: **{list(valid_classes)}**.
        - To proceed, please import a labeled dataset or load the educational demo dataset from the sidebar.
        """)
    else:
        c_train1, c_train2 = st.columns([1, 2])
        with c_train1:
            st.metric("Total Labeled Observations", f"{len(labeled_subset)}")
            st.metric("Positive Estrus Events", f"{int((labeled_subset['estrus_label'] == 1).sum())}")
            st.metric("Negative Baseline Days", f"{int((labeled_subset['estrus_label'] == 0).sum())}")

            retrain_btn = st.button("🚀 Train & Benchmark Models", type="primary", use_container_width=True)

        with c_train2:
            st.markdown("""
            **Experiment Hypothesis:**
            Does incorporating **individual rolling baseline deviations** (e.g. activity ratio, rumination ratio) along with **breed**, **parity**, and **environmental THI** outperform raw sensor readings alone?
            """)

        # Execute training if button clicked or already in session
        if retrain_btn or "estrus_models" not in st.session_state or st.session_state["estrus_models"] is None:
            with st.spinner("Training Random Forest algorithms and computing out-of-sample metrics..."):
                results, msg = train_estrus_models(ml_data)
                if results:
                    st.session_state["estrus_models"] = results
                    
                    # Save metrics to database for audit
                    for _, row in results["metrics_df"].iterrows():
                        database.save_model_evaluation(
                            model_name=row["Model"],
                            accuracy=row["Accuracy"],
                            precision_val=row["Precision"],
                            recall_val=row["Recall"],
                            f1_val=row["F1-Score"],
                            sample_count=results["train_samples"] + results["test_samples"],
                            features_used=f"{row['Features Count']} features"
                        )
                    st.success(msg)
                else:
                    st.error(msg)

        if "estrus_models" in st.session_state and st.session_state["estrus_models"] is not None:
            results = st.session_state["estrus_models"]
            metrics_df = results["metrics_df"]

            st.write("### Evaluation Summary Table")
            st.dataframe(metrics_df, use_container_width=True, hide_index=True)

            # Visual Comparison Chart
            fig_comp = plot_model_comparison(metrics_df)
            st.plotly_chart(fig_comp, use_container_width=True, key="model_comp_barchart")


# ---------------------------------------------------------------------------
# TAB 2: CONFUSION MATRICES
# ---------------------------------------------------------------------------
with tab_matrices:
    st.subheader("Classification Confusion Matrices")

    if "estrus_models" in st.session_state and st.session_state["estrus_models"] is not None:
        res = st.session_state["estrus_models"]
        cm_col1, cm_col2 = st.columns(2)

        with cm_col1:
            st.markdown("#### Model 1: Basic Behavioural")
            fig_cm1 = plot_confusion_matrix_heatmap(res["cm_basic"], labels=["No Estrus (0)", "Estrus (1)"])
            st.plotly_chart(fig_cm1, use_container_width=True, key="cm_basic_heatmap")

        with cm_col2:
            st.markdown("#### Model 2: Individualized Context-Aware")
            fig_cm2 = plot_confusion_matrix_heatmap(res["cm_individual"], labels=["No Estrus (0)", "Estrus (1)"])
            st.plotly_chart(fig_cm2, use_container_width=True, key="cm_indiv_heatmap")

        st.markdown("""
        **How to Read the Confusion Matrix:**
        - **True Negatives (Top-Left):** Normal non-estrus days correctly identified.
        - **False Positives (Top-Right):** Normal days wrongly flagged as estrus (wastes expensive semen straws).
        - **False Negatives (Bottom-Left):** Estrus events missed by the model (prolongs calving interval).
        - **True Positives (Bottom-Right):** Real estrus events accurately caught for timely insemination.
        """)
    else:
        st.info("Train the models in the 'Comparative Benchmark' tab to generate confusion matrices.")


# ---------------------------------------------------------------------------
# TAB 3: FEATURE IMPORTANCE
# ---------------------------------------------------------------------------
with tab_importance:
    st.subheader("Gini Feature Importance Rankings")

    if "estrus_models" in st.session_state and st.session_state["estrus_models"] is not None:
        res = st.session_state["estrus_models"]
        feats, importances = res["individual_importances"]

        fig_fi = plot_feature_importance(feats, importances, title="Predictive Power of Sensor & Context Variables")
        st.plotly_chart(fig_fi, use_container_width=True, key="feature_importance_plot")

        st.markdown("""
        **Biological Insight from Feature Rankings:**
        - The **Activity Ratio** and **Rumination Ratio** typically dominate the rankings over raw steps.
        - This empirically validates that **relative change from a cow's normal baseline** is far more informative than absolute step counts!
        """)
    else:
        st.info("Train the models to inspect feature importance rankings.")


# ---------------------------------------------------------------------------
# TAB 4: STUDENT VIVA & EVALUATION GUIDE
# ---------------------------------------------------------------------------
with tab_viva:
    st.subheader("Key Concepts for University Project Evaluation")
    st.markdown("""
    ### 1. What is Precision vs. Recall in Cattle Estrus Detection?
    - **Precision** is $\\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Positives}}$.
      *Why it matters biologically:* High precision means when the system alerts you, the cow is truly in estrus. This prevents farmers from wasting expensive sexed semen straws ($30–$60 per straw) on cows that are not ovulating.
    - **Recall (Sensitivity)** is $\\frac{\\text{True Positives}}{\\text{True Positives} + \\text{False Negatives}}$.
      *Why it matters biologically:* High recall ensures that true estrus cycles are not missed. Each missed estrus extends the cow's non-productive 'days open' by 21 days, costing ~$5–$7 per cow per day in lost milk and feed.

    ### 2. Why Random Forest?
    - Handles complex non-linear biological interactions without needing heavy neural networks.
    - Scale-invariant: No complex feature scaling required across steps, temperature, and percentages.
    - Provides interpretable Gini feature importance to verify biological plausibility.

    ### 3. What is Data Leakage in Cow Time-Series?
    - If observations from the *same cow* are randomly distributed across both training and testing sets, the model might simply 'memorize' that individual cow's unique tag or baseline rather than learning generalizable biological principles of estrus.
    - In BOVIGENIX, we address this using **Cow-Wise train/test partitioning**.
    """)
