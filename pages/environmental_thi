BOVIGENIX - Module F: Environmental Monitoring & THI (pages/environmental_thi.py)
---------------------------------------------------------------------------------
Assesses thermal load and heat stress risk on dairy cattle using the standard
National Research Council (NRC) Temperature-Humidity Index (THI).
"""

import streamlit as st
import pandas as pd
import numpy as np
import database
from utils.calculations import calculate_thi, get_heat_stress_category
from utils.visualizations import plot_thi_gauge
import plotly.graph_objects as go

st.markdown("""
<div class="bovi-header">
    <h1>🌡️ Environmental Heat Stress & THI Assessment</h1>
    <p>Monitor ambient barn microclimate, compute Temperature-Humidity Index (THI), and prevent thermal distress</p>
</div>
""", unsafe_allow_html=True)

# Critical biological rule alert
st.markdown("""
<div class="disclaimer-badge">
    <strong>Fundamental Bioclimatological Rule:</strong> THI calculations strictly require <em>ambient environmental air temperature</em> and <em>relative humidity</em>. 
    <strong>Never</strong> substitute internal cow body temperature into THI formulas, as that conflates external environmental load with internal homeostatic core temperature.
</div>
""", unsafe_allow_html=True)

tab_calc, tab_trends, tab_science = st.tabs([
    "🎛️ Interactive THI Calculator & Gauge",
    "📈 Barn Microclimate Trends",
    "📚 Heat Stress Biology & Mitigation"
])

# ---------------------------------------------------------------------------
# TAB 1: INTERACTIVE THI CALCULATOR
# ---------------------------------------------------------------------------
with tab_calc:
    st.subheader("Interactive Ambient THI Simulator")
    st.caption("Simulate barn temperature and humidity to observe live heat stress classification.")

    c_left, c_right = st.columns([1, 1])

    with c_left:
        sim_temp = st.slider("Ambient Air Temperature (°C)", min_value=10.0, max_value=48.0, value=31.5, step=0.5)
        sim_rh = st.slider("Relative Humidity (%)", min_value=15.0, max_value=100.0, value=72.0, step=1.0)

        calc_value = calculate_thi(sim_temp, sim_rh)
        cat_name, cat_color, cat_desc = get_heat_stress_category(calc_value)

        st.markdown(f"""
        <div style="border: 1px solid #cbd5e1; border-left: 6px solid {cat_color}; background-color: #ffffff; color: #0f172a; padding: 18px 20px; border-radius: 8px; margin-top: 15px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
            <div style="font-size: 0.85rem; color: #475569; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px;">CALCULATED HEAT STRESS LEVEL</div>
            <div style="color: {cat_color}; margin: 6px 0 8px 0; font-size: 1.8rem; font-weight: 800;">{cat_name}</div>
            <p style="margin: 0; color: #1e293b; font-size: 0.95rem; line-height: 1.5;">{cat_desc}</p>
        </div>
        """, unsafe_allow_html=True)

    with c_right:
        fig_g = plot_thi_gauge(calc_value)
        st.plotly_chart(fig_g, use_container_width=True, key="thi_gauge_chart")

        st.markdown("""
        **Reference Scale:**
        - 🟢 **< 68:** Normal Comfort Zone
        - 🟡 **68 – 71.9:** Mild Heat Stress
        - 🟠 **72 – 78.9:** Moderate Heat Stress
        - 🔴 **79 – 83.9:** Severe Heat Stress
        - 🟣 **≥ 84:** Emergency Danger Zone
        """)


# ---------------------------------------------------------------------------
# TAB 2: BARN MICROCLIMATE TRENDS
# ---------------------------------------------------------------------------
with tab_trends:
    st.subheader("Historical Barn Environmental Observations")

    all_obs = database.get_observations(limit=500)

    if not all_obs.empty and "ambient_temperature" in all_obs.columns and "relative_humidity" in all_obs.columns:
        env_df = all_obs.dropna(subset=["ambient_temperature", "relative_humidity"]).copy()

        if not env_df.empty:
            env_df = env_df.sort_values("timestamp")
            
            # Line chart for Ambient Temp & Humidity
            fig_env = go.Figure()
            fig_env.add_trace(go.Scatter(
                x=env_df["timestamp"], y=env_df["ambient_temperature"],
                name="Ambient Temp (°C)", line=dict(color="#ea580c", width=2.2)
            ))
            fig_env.add_trace(go.Scatter(
                x=env_df["timestamp"], y=env_df["relative_humidity"],
                name="Relative Humidity (%)", line=dict(color="#0284c7", width=2.2, dash="dash")
            ))
            fig_env.add_trace(go.Scatter(
                x=env_df["timestamp"], y=env_df["thi"],
                name="Calculated THI", line=dict(color="#7c3aed", width=2.5)
            ))

            # Add horizontal heat stress thresholds
            fig_env.add_hline(y=72, line_dash="dot", line_color="#f97316", annotation_text="Moderate Stress (72)")
            fig_env.add_hline(y=79, line_dash="dot", line_color="#ef4444", annotation_text="Severe Stress (79)")

            fig_env.update_layout(
                title=dict(
                    text="<b>Environmental Microclimate & THI Trajectory</b>",
                    font=dict(size=16, color="#0f172a"),
                    x=0.02,
                    xanchor="left"
                ),
                template="plotly_white",
                paper_bgcolor="#ffffff",
                plot_bgcolor="#ffffff",
                xaxis=dict(
                    title=dict(text="<b>Observation Timestamp</b>", font=dict(color="#0f172a", size=12)),
                    tickfont=dict(color="#334155", size=11),
                    showgrid=True,
                    gridcolor="#f1f5f9"
                ),
                yaxis=dict(
                    title=dict(text="<b>Reading Value (°C / % / THI)</b>", font=dict(color="#0f172a", size=12)),
                    tickfont=dict(color="#334155", size=11),
                    showgrid=True,
                    gridcolor="#f1f5f9"
                ),
                height=430,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.22,
                    xanchor="center",
                    x=0.5,
                    font=dict(size=12, color="#0f172a"),
                    bgcolor="rgba(255, 255, 255, 0.9)",
                    bordercolor="#e2e8f0",
                    borderwidth=1
                ),
                margin=dict(l=60, r=40, t=65, b=85)
            )
            st.plotly_chart(fig_env, use_container_width=True, key="thi_trend_line_chart")

            # Summary stats
            e1, e2, e3 = st.columns(3)
            with e1:
                st.metric("Avg Barn Temperature", f"{env_df['ambient_temperature'].mean():.1f} °C")
            with e2:
                st.metric("Avg Relative Humidity", f"{env_df['relative_humidity'].mean():.1f} %")
            with e3:
                st.metric("Peak Recorded THI", f"{env_df['thi'].max():.1f}")
        else:
            st.info("No environmental records with valid ambient temperature and humidity found.")
    else:
        st.info("No environmental readings logged in the current dataset.")


# ---------------------------------------------------------------------------
# TAB 3: HEAT STRESS BIOLOGY & MITIGATION
# ---------------------------------------------------------------------------
with tab_science:
    st.subheader("Biological Impact of Heat Stress on Dairy Cattle")
    st.markdown("""
    ### Why High THI Harms Dairy Production:
    High ambient temperatures combined with high humidity create a severe physiological challenge:
    1. **Reduced Dry Matter Intake (DMI):** Cows consume 10–25% less feed to minimize metabolic heat generation from rumen fermentation.
    2. **Drop in Milk Production:** Direct milk loss of 2 to 5 Liters/cow/day during moderate-to-severe heat events.
    3. **Impaired Reproductive Performance:**
       - Poor oocyte quality and reduced follicular dominance.
       - Weakened behavioral signs of estrus (shorter standing heat duration).
       - Higher early embryonic mortality before day 42 post-insemination.

    ### Standard Barn Mitigation Interventions:
    - **Air Circulation:** Maintain air velocity of 2 to 3 m/s over cows using industrial ceiling or box fans.
    - **Evaporative Cooling:** Short cycles of sprinkler misting over feed bunks followed by high-speed air drying.
    - **Continuous Hydration:** Ensure unlimited access to chilled, clean drinking water (water intake rises > 40%).
    - **Ration Adjustments:** Increase dietary energy density and potassium/sodium electrolytes.
    """)
