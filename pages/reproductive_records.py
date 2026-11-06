"""
BOVIGENIX - Module G: Reproductive Record Management (pages/reproductive_records.py)
-----------------------------------------------------------------------------------
Tracks estrus observations, artificial insemination (AI), pregnancy diagnoses,
and calving events with automatic follow-up scheduling.
"""

import streamlit as st
import pandas as pd
from datetime import date, timedelta
import database

st.markdown("""
<div class="bovi-header">
    <h1>📋 Reproductive Lifecycle & Breeding Records</h1>
    <p>Track artificial inseminations, pregnancy diagnosis results, and calving outcomes with strict veterinary validation</p>
</div>
""", unsafe_allow_html=True)

# Mandatory reproductive logic reminder
st.markdown("""
<div class="disclaimer-badge">
    <strong>Reproductive Science Principle:</strong> A missed return to estrus at 18–24 days post-insemination generates a <em>possible-pregnancy reminder</em>. 
    It does <strong>never</strong> automatically confirm pregnancy. 
    Definitive pregnancy confirmation strictly requires an ultrasound exam or rectal palpation performed by a qualified veterinarian.
</div>
""", unsafe_allow_html=True)

tab_entry, tab_history, tab_schedule = st.tabs([
    "➕ Record Reproductive Event",
    "📜 Reproductive History Log",
    "📅 Scheduled Follow-ups & Reminders"
])

all_cows_df = database.get_all_cows()
cow_ids = all_cows_df["cow_id"].tolist() if not all_cows_df.empty else []

# ---------------------------------------------------------------------------
# TAB 1: RECORD REPRODUCTIVE EVENT
# ---------------------------------------------------------------------------
with tab_entry:
    st.subheader("Log Reproductive Action or Outcome")

    if not cow_ids:
        st.info("No cows registered in the system. Register a cow first.")
    else:
        with st.form("repro_event_form", clear_on_submit=True):
            r1, r2 = st.columns(2)

            with r1:
                sel_cow = st.selectbox("Select Cow ID *", cow_ids)
                event_type = st.selectbox("Event Type *", [
                    "Artificial Insemination (AI)",
                    "Estrus Observation (Standing Heat)",
                    "Pregnancy Diagnosis (PD)",
                    "Calving / Birth Record"
                ])
                event_date = st.date_input("Event Date *", value=date.today())

            with r2:
                # Dynamic result input depending on event type
                if event_type == "Artificial Insemination (AI)":
                    result_val = st.text_input("Bull ID / Semen Straw Batch *", placeholder="e.g. Bull HF-902 (Batch #8821)")
                    auto_follow_days = 21  # Check for return to estrus
                    follow_date_default = event_date + timedelta(days=auto_follow_days)
                    follow_label = "Scheduled Return-to-Estrus Check (Day 21)"
                elif event_type == "Pregnancy Diagnosis (PD)":
                    result_val = st.selectbox("Diagnosis Result *", ["Positive (Confirmed Pregnant)", "Negative (Open / Not Pregnant)", "Inconclusive / Recheck Needed"])
                    if "Positive" in result_val:
                        auto_follow_days = 223  # Approx remaining days to calving (283 total gestation - 60 days PD)
                        follow_date_default = event_date + timedelta(days=auto_follow_days)
                        follow_label = "Expected Calving Follow-up"
                    else:
                        auto_follow_days = 21
                        follow_date_default = event_date + timedelta(days=auto_follow_days)
                        follow_label = "Next Cycle Estrus Synchronization Check"
                elif event_type == "Calving / Birth Record":
                    result_val = st.text_input("Calving Details *", placeholder="e.g. Single female calf born alive (Tag: CALF-301, Weight: 38 kg)")
                    follow_date_default = event_date + timedelta(days=60)
                    follow_label = "Postpartum Breeding Readiness Check (VWP 60 days)"
                else:  # Estrus Observation
                    result_val = st.text_input("Heat Details", placeholder="e.g. Clear mucous discharge, mounting other cows")
                    follow_date_default = event_date + timedelta(days=1)
                    follow_label = "Insemination Follow-up (AM-PM Rule)"

                follow_up_input = st.date_input(f"Scheduled Follow-up Date ({follow_label})", value=follow_date_default)
                vet_name = st.text_input("Technician / Veterinarian Name", placeholder="e.g. Dr. A. Verma")

            remarks_val = st.text_area("Clinical Notes / Remarks", placeholder="e.g. Cervical mucus clear and elastic. Normal insemination.")

            submit_repro = st.form_submit_button("Save Reproductive Record", type="primary")

            if submit_repro:
                if not result_val:
                    st.error("Please enter specific details or results for this event.")
                else:
                    success, msg = database.add_reproductive_record(
                        cow_id=sel_cow,
                        record_type=event_type,
                        event_date=str(event_date),
                        result=result_val,
                        follow_up_date=str(follow_up_input),
                        technician_or_vet=vet_name,
                        remarks=remarks_val
                    )
                    if success:
                        st.success(msg)
                        st.rerun()
                    else:
                        st.error(msg)


# ---------------------------------------------------------------------------
# TAB 2: REPRODUCTIVE HISTORY LOG
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Herd Lifetime Breeding Log")

    filter_cow = st.selectbox("Filter History by Cow ID:", ["All Cows"] + cow_ids)
    
    if filter_cow == "All Cows":
        records_df = database.get_reproductive_records()
    else:
        records_df = database.get_reproductive_records(cow_id=filter_cow)

    if not records_df.empty:
        display_repro = records_df[[
            "cow_id", "name", "record_type", "event_date", "result",
            "follow_up_date", "technician_or_vet", "remarks"
        ]].copy()
        display_repro.columns = [
            "Cow ID", "Cow Name", "Event Type", "Event Date", "Result / Details",
            "Follow-up Date", "Vet / Technician", "Remarks"
        ]
        st.dataframe(display_repro, use_container_width=True, hide_index=True)
    else:
        st.info("No reproductive records found.")


# ---------------------------------------------------------------------------
# TAB 3: SCHEDULED FOLLOW-UPS
# ---------------------------------------------------------------------------
with tab_schedule:
    st.subheader("Upcoming Scheduled Follow-ups & Reminders")
    st.caption("Automatically populated follow-up dates based on gestational and cycle intervals.")

    upcoming_df = database.get_upcoming_follow_ups()

    if not upcoming_df.empty:
        for _, item in upcoming_df.iterrows():
            st.markdown(f"""
            <div style="border: 1px solid #cbd5e1; border-left: 5px solid #0284c7; background-color: #ffffff; color: #0f172a; padding: 14px 18px; border-radius: 8px; margin-bottom: 12px; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div style="font-size: 1.05rem; font-weight: 700; color: #0f172a;">{item['cow_id']} <span style="color:#64748b; font-weight:normal;">({item['name']})</span></div>
                    <span style="background-color: #e0f2fe; color: #0369a1; padding: 3px 10px; border-radius: 4px; font-weight: 700; font-size: 0.82rem;">Due: {item['follow_up_date']}</span>
                </div>
                <div style="font-size: 0.90rem; color: #1e293b; margin-top: 6px; line-height: 1.5;">
                    <strong style="color:#0f172a;">Prior Action:</strong> {item['record_type']} on {item['event_date']}<br>
                    <strong style="color:#0f172a;">Details:</strong> {item['result']}<br>
                    <strong style="color:#0f172a;">Follow-up Goal:</strong> {item['remarks']}
                </div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.info("No reproductive follow-ups currently scheduled for the coming weeks.")
