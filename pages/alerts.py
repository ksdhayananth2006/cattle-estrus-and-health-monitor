"""
BOVIGENIX - Module H: Alerts & Veterinary Review (pages/alerts.py)
------------------------------------------------------------------
Manages automated decision-support notifications and provides a formal review
station for farm veterinarians to classify alerts as Confirmed, False Positive,
or Inconclusive.
"""

import streamlit as st
import pandas as pd
import database

st.markdown("""
<div class="bovi-header">
    <h1>🔔 Alert Feed & Veterinary Review Station</h1>
    <p>Audit decision-support alerts, log clinical verifications, and record feedback to improve model accuracy</p>
</div>
""", unsafe_allow_html=True)

# Fetch alerts
all_alerts_df = database.get_alerts()

tab_review, tab_history, tab_manual = st.tabs([
    "📥 Review Pending Alerts",
    "📜 Alert Audit Log",
    "➕ Generate Manual Herd Alert"
])

# ---------------------------------------------------------------------------
# TAB 1: REVIEW PENDING ALERTS
# ---------------------------------------------------------------------------
with tab_review:
    st.subheader("Unresolved Alerts Requiring Action")

    if all_alerts_df.empty:
        st.info("No alerts found in the database.")
    else:
        pending_alerts = all_alerts_df[all_alerts_df["status"] == "Pending"].copy()

        if pending_alerts.empty:
            st.success("🎉 All alerts have been reviewed! No pending alerts remain.")
        else:
            st.write(f"Displaying **{len(pending_alerts)}** pending alert(s):")

            for _, alert in pending_alerts.iterrows():
                alert_id = alert["id"]
                cid = alert["cow_id"]
                sev = alert["severity"]
                sev_color = "#dc2626" if sev in ["Critical", "High"] else "#ea580c"

                with st.expander(f"⚠️ {alert['alert_type']}: {cid} ({alert['name'] or 'Cow'}) — {alert['alert_date']}", expanded=True):
                    st.markdown(f"""
                    <div style="border: 1px solid #cbd5e1; border-left: 5px solid {sev_color}; background-color: #ffffff; color: #0f172a; padding: 14px 18px; border-radius: 8px; margin-bottom: 14px; box-shadow: 0 1px 4px rgba(0,0,0,0.03);">
                        <div style="font-size: 0.95rem; color: #0f172a; line-height: 1.5;"><strong style="color:#0f172a;">Observation:</strong> {alert['message']}</div>
                        <div style="font-size: 0.90rem; color: #0f766e; margin-top: 6px; line-height: 1.5;"><strong style="color:#0f766e;">Suggested Action:</strong> {alert['suggested_action']}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Review Form
                    with st.form(f"review_form_{alert_id}"):
                        rc1, rc2 = st.columns(2)
                        with rc1:
                            new_status = st.selectbox(
                                "Clinical Determination *",
                                ["Confirmed", "False Positive", "Inconclusive"],
                                key=f"status_{alert_id}"
                            )
                            reviewer = st.text_input("Reviewer Name / Title *", placeholder="e.g. Dr. K. Sharma (Herd Vet)", key=f"rev_{alert_id}")
                        with rc2:
                            notes = st.text_area("Clinical Findings & Notes *", placeholder="e.g. Examined animal. Standing heat confirmed with clear mucus. Proceeded to AI.", key=f"notes_{alert_id}")

                        submit_review = st.form_submit_button("Record Clinical Decision", type="primary")

                        if submit_review:
                            if not reviewer:
                                st.error("Please enter the reviewer's name.")
                            else:
                                success, msg = database.update_alert_review(
                                    alert_id=alert_id,
                                    status=new_status,
                                    reviewer_name=reviewer,
                                    reviewer_notes=notes
                                )
                                if success:
                                    st.success(f"Alert #{alert_id} updated to '{new_status}'!")
                                    st.rerun()
                                else:
                                    st.error(msg)


# ---------------------------------------------------------------------------
# TAB 2: ALERT AUDIT LOG
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Historical Alert Archive")

    if not all_alerts_df.empty:
        # Filter controls
        c1, c2 = st.columns(2)
        with c1:
            status_filter = st.selectbox("Filter by Status:", ["All"] + all_alerts_df["status"].unique().tolist())
        with c2:
            type_filter = st.selectbox("Filter by Alert Type:", ["All"] + all_alerts_df["alert_type"].unique().tolist())

        filtered_alerts = all_alerts_df.copy()
        if status_filter != "All":
            filtered_alerts = filtered_alerts[filtered_alerts["status"] == status_filter]
        if type_filter != "All":
            filtered_alerts = filtered_alerts[filtered_alerts["alert_type"] == type_filter]

        display_cols = [
            "id", "cow_id", "alert_type", "severity", "alert_date",
            "message", "status", "reviewer_name", "reviewer_notes", "reviewed_at"
        ]
        display_df = filtered_alerts[display_cols].copy()
        display_df.columns = [
            "Alert ID", "Cow ID", "Type", "Severity", "Date",
            "Message", "Review Status", "Reviewer", "Notes", "Reviewed At"
        ]
        st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("No alerts logged in the archive.")


# ---------------------------------------------------------------------------
# TAB 3: GENERATE MANUAL HERD ALERT
# ---------------------------------------------------------------------------
with tab_manual:
    st.subheader("Create Manual Decision-Support Alert")
    st.caption("Farm managers can manually dispatch an alert for clinical follow-up.")

    cows_df = database.get_all_cows()
    if cows_df.empty:
        st.info("No cows registered.")
    else:
        with st.form("manual_alert_form", clear_on_submit=True):
            mc1, mc2 = st.columns(2)
            with mc1:
                manual_cow = st.selectbox("Select Cow ID", cows_df["cow_id"].tolist())
                manual_type = st.selectbox("Alert Category", [
                    "Estrus Prediction",
                    "Health Risk",
                    "Heat Stress",
                    "Possible-Pregnancy / Delayed Return Reminder",
                    "Veterinary Recheck"
                ])
                manual_sev = st.selectbox("Severity Level", ["Info", "Warning", "High", "Critical"])
            with mc2:
                manual_msg = st.text_area("Alert Description", placeholder="e.g. Cow exhibits swollen left hind leg and reluctance to walk.")
                manual_action = st.text_input("Suggested Action", placeholder="e.g. Schedule claw inspection in hoof trimming chute.")

            create_btn = st.form_submit_button("Broadcast Alert", type="primary")

            if create_btn:
                if not manual_msg:
                    st.error("Please provide an alert description.")
                else:
                    database.add_alert(
                        cow_id=manual_cow,
                        alert_type=manual_type,
                        severity=manual_sev,
                        message=manual_msg,
                        suggested_action=manual_action
                    )
                    st.success(f"Alert broadcasted for {manual_cow}!")
                    st.rerun()
