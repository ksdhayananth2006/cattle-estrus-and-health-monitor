"""
BOVIGENIX - Module A: Dashboard (pages/dashboard.py)
----------------------------------------------------
Central operational dashboard displaying live herd statistics, active alerts,
reproductive schedules, and sensor time-series trends retrieved from the database.
"""

import streamlit as st
import database
import pandas as pd
from visualizations import plot_herd_status_donut, plot_sensor_trends
from calculations import calculate_age

st.markdown("""
<div class="bovi-header">
    <h1>📊 Herd Operational Dashboard</h1>
    <p>Real-time physiological indicators, automated alerts, and reproductive follow-ups</p>
</div>
""", unsafe_allow_html=True)

# Decision-support notice
st.markdown("""
<div class="disclaimer-badge">
    <strong>Veterinary Decision-Support Notice:</strong> BOVIGENIX provides predictive screening and baseline monitoring to assist dairy herd managers. It does not replace veterinary clinical diagnosis or physical rectal palpation.
</div>
""", unsafe_allow_html=True)

# 1. Fetch live metrics from database
stats = database.get_cows_summary_stats()
alert_stats = database.get_alerts_summary_stats()
follow_ups_df = database.get_upcoming_follow_ups()
recent_alerts_df = database.get_alerts(limit=5)
all_cows_df = database.get_all_cows()

# 2. Top Summary KPI Cards
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)

with kpi1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Registered Cows</div>
        <div class="metric-value">{stats['total']}</div>
        <div class="metric-subtitle">Total Herd Size</div>
    </div>
    """, unsafe_allow_html=True)

with kpi2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Monitoring</div>
        <div class="metric-value">{stats['active']}</div>
        <div class="metric-subtitle">{stats['inactive'] + stats['sold'] + stats['deceased']} Inactive/Sold</div>
    </div>
    """, unsafe_allow_html=True)

with kpi3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Pending Estrus</div>
        <div class="metric-value" style="color: #ea580c;">{alert_stats['pending_estrus']}</div>
        <div class="metric-subtitle">Requires Insemination Check</div>
    </div>
    """, unsafe_allow_html=True)

with kpi4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Health Risk Flags</div>
        <div class="metric-value" style="color: #dc2626;">{alert_stats['pending_health']}</div>
        <div class="metric-subtitle">Veterinary Action Needed</div>
    </div>
    """, unsafe_allow_html=True)

with kpi5:
    upcoming_count = len(follow_ups_df) if not follow_ups_df.empty else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Upcoming Tasks</div>
        <div class="metric-value" style="color: #0284c7;">{upcoming_count}</div>
        <div class="metric-subtitle">Reproductive Follow-ups</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")

# If herd is empty, show helpful empty state
if stats["total"] == 0:
    st.info("No cattle registered yet. Navigate to **Cow Management** to register cows or **Dataset Import** to load existing sensor datasets.")
    st.stop()

# 3. Main Dashboard Body: Visualizations & Recent Alerts
col_chart, col_status = st.columns([2, 1])

with col_chart:
    st.subheader("🐄 Cow Sensor Trajectory")
    
    # Dropdown to inspect a specific cow's recent time series
    active_cows = all_cows_df[all_cows_df["status"] == "Active"]
    cow_list = active_cows["cow_id"].tolist() if not active_cows.empty else all_cows_df["cow_id"].tolist()
    
    if cow_list:
        selected_cow = st.selectbox("Select Cow to View Sensor Trend:", cow_list, key="dash_cow_select")
        cow_obs = database.get_observations(cow_id=selected_cow, limit=60)
        
        if not cow_obs.empty:
            fig_trend = plot_sensor_trends(cow_obs, selected_cow)
            st.plotly_chart(fig_trend, use_container_width=True, key="dash_cow_trend_chart")
        else:
            st.info(f"No sensor observations logged for Cow {selected_cow} yet.")
    else:
        st.info("No cows available for trend visualization.")

with col_status:
    st.subheader("🧬 Herd Status")
    status_dict = {
        "Active": stats["active"],
        "Inactive": stats["inactive"],
        "Sold": stats["sold"],
        "Deceased": stats["deceased"]
    }
    fig_donut = plot_herd_status_donut(status_dict)
    st.plotly_chart(fig_donut, use_container_width=True, key="dash_herd_donut_chart")

    # Quick Status Summary Chips
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown(f"<div style='border:1px solid #bbf7d0;background:#f0fdf4;border-radius:6px;padding:6px 10px;text-align:center;font-size:0.82rem;font-weight:700;color:#166534;'>🟢 Active: {stats['active']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='border:1px solid #bfdbfe;background:#eff6ff;border-radius:6px;padding:6px 10px;text-align:center;font-size:0.82rem;font-weight:700;color:#1e40af;margin-top:6px;'>🔵 Sold: {stats['sold']}</div>", unsafe_allow_html=True)
    with sc2:
        st.markdown(f"<div style='border:1px solid #e2e8f0;background:#f8fafc;border-radius:6px;padding:6px 10px;text-align:center;font-size:0.82rem;font-weight:700;color:#475569;'>⚪ Inactive: {stats['inactive']}</div>", unsafe_allow_html=True)
        st.markdown(f"<div style='border:1px solid #cbd5e1;background:#f1f5f9;border-radius:6px;padding:6px 10px;text-align:center;font-size:0.82rem;font-weight:700;color:#334155;margin-top:6px;'>⚫ Deceased: {stats['deceased']}</div>", unsafe_allow_html=True)

st.divider()

# 4. Lower Dashboard: Recent Alerts & Upcoming Reproductive Follow-ups
col_alerts, col_follow = st.columns([1, 1])

with col_alerts:
    st.subheader("🔔 Recent Unresolved Alerts")
    if not recent_alerts_df.empty:
        for _, alert in recent_alerts_df.iterrows():
            severity = alert["severity"]
            sev_color = "#dc2626" if severity in ["Critical", "High"] else "#f59e0b"
            
            with st.container():
                st.markdown(f"""
                <div style="border: 1px solid #cbd5e1; border-left: 5px solid {sev_color}; background-color: #ffffff; color: #0f172a; padding: 12px 16px; border-radius: 8px; margin-bottom: 10px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                    <div style="font-size: 0.95rem; font-weight: 700; color: #0f172a;">
                        {alert['alert_type']} — <span style="color:#0f766e;">{alert['cow_id']}</span> <span style="color:#64748b; font-weight: normal;">({alert['name'] or 'Unknown'})</span>
                    </div>
                    <div style="font-size:0.88rem; color:#1e293b; margin: 5px 0 4px 0;">
                        {alert['message']}
                    </div>
                    <div style="font-size:0.78rem; color:#64748b;">
                        {alert['alert_date']} | Status: <strong style="color:#0f172a;">{alert['status']}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.success("No active alerts at this time. All animals are currently within normal baseline parameters.")

with col_follow:
    st.subheader("📅 Upcoming Reproductive Follow-ups")
    if not follow_ups_df.empty:
        display_follow = follow_ups_df[["cow_id", "name", "record_type", "follow_up_date", "remarks"]].copy()
        display_follow.columns = ["Cow ID", "Name", "Prior Event", "Scheduled Date", "Follow-up Goal"]
        st.dataframe(display_follow, use_container_width=True, hide_index=True)
    else:
        st.info("No reproductive follow-ups scheduled for the upcoming period.")
