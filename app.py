"""
BOVIGENIX - Main Application Entrypoint (app.py)
================================================
An AI-Driven, Individualized and Context-Aware System for Estrus Prediction,
Reproductive Management and Health-Risk Assessment in Dairy Cattle.

Developed for Biotechnology & Python for Biologists coursework.

Architecture:
- Frontend: Streamlit multi-page navigation
- Database: SQLite with transparent SQL operations in database.py
- Machine Learning: Scikit-learn Random Forest in models/estrus_model.py
- Scientific Analytics: utils/calculations.py & utils/visualizations.py
"""

import streamlit as st
import os
import database
from data_processing import generate_synthetic_cattle_dataset

# 1. Configure Streamlit Page Settings
st.set_page_config(
    page_title="BOVIGENIX | Dairy Cattle Monitoring",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Initialize Database on startup
database.init_db()

# 3. Custom CSS for a professional scientific livestock application theme
st.markdown("""
<style>
    /* Main theme accents */
    :root {
        --bovi-primary: #0f766e;
        --bovi-secondary: #0284c7;
        --bovi-dark: #0f172a;
        --bovi-bg: #f8fafc;
    }
    
    /* Enforce high-contrast text globally on all standard elements */
    html, body, [class*="css"], .stMarkdown, p, span, label, div {
        color: #0f172a;
    }
    
    /* Top title styling */
    .bovi-header {
        background: linear-gradient(135deg, #042f2e 0%, #0f766e 50%, #0369a1 100%) !important;
        padding: 24px 30px !important;
        border-radius: 12px !important;
        margin-bottom: 24px !important;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.08) !important;
    }
    .bovi-header h1 {
        color: #ffffff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        margin: 0 !important;
        letter-spacing: -0.5px !important;
    }
    .bovi-header p {
        color: #ccfbf1 !important;
        font-size: 1.0rem !important;
        margin-top: 6px !important;
        margin-bottom: 0 !important;
    }
    
    /* Metric Card Styling with guaranteed high contrast */
    .metric-card {
        background-color: #ffffff !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 10px !important;
        padding: 16px 18px !important;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.04) !important;
        transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    }
    .metric-card:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 5px 12px rgba(0, 0, 0, 0.08) !important;
    }
    .metric-title {
        color: #475569 !important;
        font-size: 0.82rem !important;
        font-weight: 700 !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    .metric-value {
        color: #0f172a !important;
        font-size: 1.85rem !important;
        font-weight: 800 !important;
        margin: 4px 0 !important;
        line-height: 1.2 !important;
    }
    .metric-subtitle {
        color: #0d9488 !important;
        font-size: 0.80rem !important;
        font-weight: 600 !important;
    }

    /* Decision support disclaimer badge */
    .disclaimer-badge {
        background-color: #f0fdf4 !important;
        border: 1px solid #bbf7d0 !important;
        border-left: 5px solid #16a34a !important;
        padding: 14px 18px !important;
        border-radius: 8px !important;
        font-size: 0.90rem !important;
        color: #14532d !important;
        margin-bottom: 20px !important;
        line-height: 1.5 !important;
    }
    .disclaimer-badge strong {
        color: #166534 !important;
    }

    /* Sidebar styling */
    .sidebar-brand {
        padding: 8px 0 14px 0 !important;
        border-bottom: 1px solid #e2e8f0 !important;
        margin-bottom: 14px !important;
    }

    /* Streamlit tabs styling for clean separation */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px !important;
    }
    .stTabs [data-baseweb="tab"] {
        font-weight: 600 !important;
        color: #334155 !important;
        padding: 8px 16px !important;
        border-radius: 6px 6px 0 0 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #0f766e !important;
        border-bottom-color: #0f766e !important;
    }
</style>
""", unsafe_allow_html=True)

# 4. Sidebar Branding & Quick Status
with st.sidebar:
    st.markdown("""
    <div class="sidebar-brand">
        <h2 style="color: #0f766e; margin:0; font-size:1.6rem; font-weight:800;">🐄 BOVIGENIX</h2>
        <span style="font-size:0.8rem; color:#64748b; font-weight:600;">Intelligent Dairy Cattle Management</span>
    </div>
    """, unsafe_allow_html=True)

    # Check database status
    stats = database.get_cows_summary_stats()
    alert_stats = database.get_alerts_summary_stats()

    st.caption("SYSTEM STATUS")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Total Herd", f"{stats['total']}")
    with c2:
        st.metric("Pending Alerts", f"{alert_stats['pending_total']}")

    st.divider()

    # Quick Demo Dataset Loader (Useful for immediate student evaluation & testing)
    if stats['total'] == 0:
        st.warning("Database is currently empty.")
        if st.button("Load Educational Demo Data", type="primary", use_container_width=True):
            with st.spinner("Populating database with educational cattle dataset..."):
                cows_df, obs_df = generate_synthetic_cattle_dataset()
                # Insert cows
                for _, row in cows_df.iterrows():
                    database.add_cow(
                        cow_id=row["cow_id"],
                        name=row["name"],
                        breed=row["breed"],
                        dob=row["dob"],
                        parity=row["parity"],
                        notes="Registered via Educational Demo Dataset"
                    )
                # Bulk insert observations
                database.bulk_insert_observations(obs_df)
                database.log_dataset_import("synthetic_demo_cattle.csv", len(obs_df), "Success", "Synthetic educational dataset")

                # Generate initial alerts for demonstration
                database.add_alert(
                    cow_id="COW-101",
                    alert_type="Estrus Prediction",
                    severity="High",
                    message="Activity surge (+95%) and rumination depression (-32%) detected on Day 18.",
                    suggested_action="Perform visual standing heat confirmation. Schedule artificial insemination in 12 hours (AM-PM rule)."
                )
                database.add_alert(
                    cow_id="COW-103",
                    alert_type="Health Risk",
                    severity="Critical",
                    message="Elevated body temperature (39.9°C) and acute rumination crash (-45%).",
                    suggested_action="Veterinary attention required. Perform California Mastitis Test (CMT) and check somatic cell count."
                )
                database.add_alert(
                    cow_id="COW-107",
                    alert_type="Heat Stress",
                    severity="Warning",
                    message="Environmental THI reached 81.2 (Severe Heat Stress zone).",
                    suggested_action="Activate barn cooling fans, shade curtains, and cold water misters."
                )
                
                # Add sample reproductive record
                database.add_reproductive_record(
                    cow_id="COW-102",
                    record_type="Artificial Insemination",
                    event_date="2026-10-15",
                    result="Inseminated with Bull HF-902 (Straw batch #8821)",
                    follow_up_date="2026-11-05",
                    technician_or_vet="Dr. R. Sharma (Vet Surgeon)",
                    remarks="Standing heat confirmed. Insemination completed smoothly."
                )

                st.success("Sample dataset loaded! You can now explore all modules.")
                st.rerun()

    st.markdown("""
    <div style="font-size:0.75rem; color:#64748b; margin-top:20px; line-height:1.4;">
        <strong>Project:</strong> BOVIGENIX<br>
        <strong>Course:</strong> Python for Biologists<br>
        <strong>Focus:</strong> AI Estrus & Health Monitoring
    </div>
    """, unsafe_allow_html=True)


# 5. Multi-page Navigation using modern Streamlit Pages
pages = [
    st.Page("pages/dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("pages/cow_management.py", title="Cow Management", icon="🐄"),
    st.Page("pages/dataset_import.py", title="Dataset Import & Processing", icon="📁"),
    st.Page("pages/estrus_monitoring.py", title="Estrus Monitoring & AI", icon="🎯"),
    st.Page("pages/health_monitoring.py", title="Health Risk Monitoring", icon="🩺"),
    st.Page("pages/environmental_thi.py", title="Environmental & THI", icon="🌡️"),
    st.Page("pages/reproductive_records.py", title="Reproductive Records", icon="📋"),
    st.Page("pages/alerts.py", title="Alerts & Veterinary Review", icon="🔔"),
    st.Page("pages/model_performance.py", title="ML Model Performance", icon="📈"),
]

nav = st.navigation(pages)
nav.run()
