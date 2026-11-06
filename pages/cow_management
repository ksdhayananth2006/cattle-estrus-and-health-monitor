"""
BOVIGENIX - Module B: Cow Management (pages/cow_management.py)
--------------------------------------------------------------
Enables registration, profile editing, age calculation, breed categorization,
and historical observation tracking for individual cattle.
"""

import streamlit as st
import database
import pandas as pd
from datetime import date, datetime
from utils.calculations import calculate_age
from utils.visualizations import plot_sensor_trends

st.markdown("""
<div class="bovi-header">
    <h1>🐄 Cow Profile & Herd Management</h1>
    <p>Register individual cattle, monitor age and parity, and maintain historical lifetime records</p>
</div>
""", unsafe_allow_html=True)

# Tabs for structured workflow
tab_directory, tab_register, tab_edit, tab_history = st.tabs([
    "📋 Herd Directory",
    "➕ Register New Cow",
    "✏️ Edit Cow Profile",
    "📜 Individual Animal History"
])

# ---------------------------------------------------------------------------
# TAB 1: HERD DIRECTORY & FILTERS
# ---------------------------------------------------------------------------
with tab_directory:
    st.subheader("Registered Herd Profiles")

    cows_df = database.get_all_cows()

    if cows_df.empty:
        st.info("No cows registered in the system. Use the 'Register New Cow' tab or load the demo dataset from the sidebar.")
    else:
        # Search & Filter Controls
        f1, f2, f3 = st.columns([2, 1, 1])
        with f1:
            search_query = st.text_input("🔍 Search by Cow ID or Name:", placeholder="e.g. COW-101 or Bella")
        with f2:
            status_options = ["All"] + sorted(cows_df["status"].unique().tolist())
            selected_status = st.selectbox("Status Filter:", status_options)
        with f3:
            breed_options = ["All"] + sorted(cows_df["breed"].unique().tolist())
            selected_breed = st.selectbox("Breed Filter:", breed_options)

        # Apply Filters
        filtered_df = cows_df.copy()
        if search_query:
            q = search_query.strip().lower()
            filtered_df = filtered_df[
                filtered_df["cow_id"].str.lower().str.contains(q) |
                filtered_df["name"].str.lower().str.contains(q)
            ]
        if selected_status != "All":
            filtered_df = filtered_df[filtered_df["status"] == selected_status]
        if selected_breed != "All":
            filtered_df = filtered_df[filtered_df["breed"] == selected_breed]

        # Calculate live age for each cow
        filtered_df["Age"] = filtered_df["dob"].apply(lambda d: calculate_age(d)[2])

        # Format display table
        display_df = filtered_df[[
            "cow_id", "name", "breed", "dob", "Age", "parity", "status", "notes"
        ]].copy()
        display_df.columns = ["Cow ID", "Name", "Breed", "Date of Birth", "Calculated Age", "Parity", "Status", "Notes"]

        st.caption(f"Showing {len(display_df)} animal profile(s)")
        st.dataframe(display_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------------------------
# TAB 2: REGISTER NEW COW
# ---------------------------------------------------------------------------
with tab_register:
    st.subheader("Register a New Cow Profile")
    st.caption("Each registered cow must have a unique Cow ID tag.")

    with st.form("register_cow_form", clear_on_submit=True):
        rc1, rc2 = st.columns(2)
        with rc1:
            new_id = st.text_input("Cow ID / Tag Number *", placeholder="e.g. COW-109").strip().upper()
            new_name = st.text_input("Cow Name *", placeholder="e.g. Blossom").strip()
            new_breed = st.selectbox("Breed *", [
                "Holstein Friesian",
                "Jersey",
                "Gir",
                "Sahiwal",
                "Crossbred",
                "Red Sindhi",
                "Tharparkar",
                "Other"
            ])
        with rc2:
            new_dob = st.date_input("Date of Birth *", max_value=date.today(), value=date(2022, 1, 1))
            new_parity = st.number_input("Parity (0 for Heifer) *", min_value=0, max_value=15, value=1, step=1)
            new_status = st.selectbox("Initial Status", ["Active", "Inactive"])

        new_notes = st.text_area("Veterinary / Management Notes (Optional)", placeholder="e.g. Purchased from Model Dairy Farm; vaccinated for FMD.")

        submitted = st.form_submit_button("Register Cow Profile", type="primary")

        if submitted:
            if not new_id:
                st.error("Cow ID cannot be blank.")
            elif not new_name:
                st.error("Cow Name cannot be blank.")
            else:
                success, msg = database.add_cow(
                    cow_id=new_id,
                    name=new_name,
                    breed=new_breed,
                    dob=str(new_dob),
                    parity=new_parity,
                    status=new_status,
                    notes=new_notes
                )
                if success:
                    st.success(f"Cow {new_id} ({new_name}) registered successfully!")
                    st.rerun()
                else:
                    st.error(msg)


# ---------------------------------------------------------------------------
# TAB 3: EDIT COW PROFILE
# ---------------------------------------------------------------------------
with tab_edit:
    st.subheader("Modify Existing Cow Profile")

    cows_list = cows_df["cow_id"].tolist() if not cows_df.empty else []

    if not cows_list:
        st.info("No cows available to edit.")
    else:
        edit_cow_id = st.selectbox("Select Cow ID to Update:", cows_list, key="edit_cow_select")
        cow_data = database.get_cow_by_id(edit_cow_id)

        if cow_data:
            # Parse existing DOB safely
            try:
                curr_dob = datetime.strptime(cow_data["dob"][:10], "%Y-%m-%d").date()
            except Exception:
                curr_dob = date.today()

            with st.form("edit_cow_form"):
                ec1, ec2 = st.columns(2)
                with ec1:
                    st.text_input("Cow ID (Immutable Identifier)", value=cow_data["cow_id"], disabled=True)
                    updated_name = st.text_input("Cow Name", value=cow_data["name"])
                    
                    breed_options = ["Holstein Friesian", "Jersey", "Gir", "Sahiwal", "Crossbred", "Red Sindhi", "Tharparkar", "Other"]
                    curr_breed_idx = breed_options.index(cow_data["breed"]) if cow_data["breed"] in breed_options else 0
                    updated_breed = st.selectbox("Breed", breed_options, index=curr_breed_idx)

                with ec2:
                    updated_dob = st.date_input("Date of Birth", value=curr_dob, max_value=date.today())
                    updated_parity = st.number_input("Parity", min_value=0, max_value=15, value=int(cow_data["parity"]), step=1)
                    
                    status_choices = ["Active", "Inactive", "Sold", "Deceased"]
                    curr_status_idx = status_choices.index(cow_data["status"]) if cow_data["status"] in status_choices else 0
                    updated_status = st.selectbox("Herd Status", status_choices, index=curr_status_idx)

                updated_notes = st.text_area("Notes", value=cow_data.get("notes") or "")

                st.caption("ℹ️ Note: Changing status to Inactive, Sold, or Deceased preserves all lifetime health and reproductive history.")
                
                save_btn = st.form_submit_button("Update Profile", type="primary")

                if save_btn:
                    success, msg = database.update_cow(
                        cow_id=edit_cow_id,
                        name=updated_name,
                        breed=updated_breed,
                        dob=str(updated_dob),
                        parity=updated_parity,
                        status=updated_status,
                        notes=updated_notes
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


# ---------------------------------------------------------------------------
# TAB 4: INDIVIDUAL ANIMAL LIFETIME HISTORY
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Individual Lifetime History")

    if not cows_list:
        st.info("No cows available.")
    else:
        hist_cow_id = st.selectbox("Select Cow for Comprehensive History:", cows_list, key="hist_cow_select")
        cow_info = database.get_cow_by_id(hist_cow_id)

        if cow_info:
            _, _, age_str = calculate_age(cow_info["dob"])
            
            # Profile summary banner
            st.markdown(f"""
            <div style="border: 1px solid #cbd5e1; border-left: 5px solid #0f766e; background-color: #ffffff; color: #0f172a; padding: 14px 18px; border-radius: 8px; margin-bottom: 16px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                <h4 style="margin: 0; color: #0f766e; font-size: 1.15rem; font-weight: 700;">{cow_info['name']} ({cow_info['cow_id']})</h4>
                <div style="font-size: 0.9rem; color: #1e293b; margin-top: 6px; line-height: 1.6;">
                    <strong style="color: #0f172a;">Breed:</strong> {cow_info['breed']} &nbsp;|&nbsp; 
                    <strong style="color: #0f172a;">Age:</strong> {age_str} &nbsp;|&nbsp; 
                    <strong style="color: #0f172a;">Parity:</strong> {cow_info['parity']} &nbsp;|&nbsp; 
                    <strong style="color: #0f172a;">Status:</strong> <span style="background-color:#e2e8f0; padding:2px 8px; border-radius:4px; font-weight:600; color:#0f172a;">{cow_info['status']}</span> &nbsp;|&nbsp; 
                    <strong style="color: #0f172a;">Registered:</strong> {cow_info['created_at']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Historical sub-tabs
            htab1, htab2, htab3 = st.tabs(["📊 Sensor History", "📋 Reproductive Records", "🔔 Alert Logs"])

            with htab1:
                cow_obs = database.get_observations(cow_id=hist_cow_id, limit=200)
                if not cow_obs.empty:
                    fig_h = plot_sensor_trends(cow_obs, hist_cow_id)
                    st.plotly_chart(fig_h, use_container_width=True, key="cow_hist_trend_chart")
                    st.dataframe(cow_obs, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No sensor observations logged for {hist_cow_id}.")

            with htab2:
                cow_repro = database.get_reproductive_records(cow_id=hist_cow_id)
                if not cow_repro.empty:
                    st.dataframe(cow_repro, use_container_width=True, hide_index=True)
                else:
                    st.info(f"No reproductive events recorded for {hist_cow_id} yet.")

            with htab3:
                alerts_df = database.get_alerts()
                if not alerts_df.empty:
                    cow_alerts = alerts_df[alerts_df["cow_id"] == hist_cow_id]
                    if not cow_alerts.empty:
                        st.dataframe(cow_alerts, use_container_width=True, hide_index=True)
                    else:
                        st.success(f"No alerts generated for {hist_cow_id}.")
                else:
                    st.info("No alerts in the system.")
