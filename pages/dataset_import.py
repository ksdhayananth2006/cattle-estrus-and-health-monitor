"""
BOVIGENIX - Module C: Dataset Import & Processing (pages/dataset_import.py)
-------------------------------------------------------------------------
Enables uploading CSV and Excel sensor observation datasets, automatic column
mapping, data-type validation, duplicate detection, and secure SQLite storage.
"""

import streamlit as st
import pandas as pd
import os
import database
from data_processing import (
    normalize_column_names,
    validate_dataset,
    STANDARD_COLUMNS,
    generate_synthetic_cattle_dataset
)
from utils.calculations import calculate_thi

st.markdown("""
<div class="bovi-header">
    <h1>📁 Dataset Import & Quality Processing</h1>
    <p>Upload sensor observation datasets, map variables, validate data quality, and store into the database</p>
</div>
""", unsafe_allow_html=True)

tab_upload, tab_sample, tab_history = st.tabs([
    "📤 Upload Dataset (CSV/Excel)",
    "🧪 Educational Sample Dataset",
    "📜 Import Audit History"
])

# ---------------------------------------------------------------------------
# TAB 1: UPLOAD DATASET
# ---------------------------------------------------------------------------
with tab_upload:
    st.subheader("Upload Cattle Sensor Observations")
    st.caption("Supports CSV (.csv) and Excel (.xlsx) formats. Files are validated prior to database ingestion.")

    uploaded_file = st.file_uploader("Choose a dataset file", type=["csv", "xlsx"])

    if uploaded_file is not None:
        try:
            # 1. Read file into pandas DataFrame
            filename = uploaded_file.name
            if filename.endswith(".csv"):
                raw_df = pd.read_csv(uploaded_file)
            else:
                raw_df = pd.read_excel(uploaded_file)

            st.success(f"File **{filename}** uploaded successfully! ({len(raw_df)} rows detected)")

            # 2. Preview original data
            with st.expander("🔍 View Raw Upload Preview (First 5 Rows)", expanded=False):
                st.dataframe(raw_df.head(), use_container_width=True)

            # 3. Column Normalization & Mapping
            clean_df, mapping = normalize_column_names(raw_df)

            st.subheader("1. Column Mapping & Recognition")
            col_map_summary = []
            for orig, mapped in mapping.items():
                col_map_summary.append({"Uploaded Column Header": orig, "Recognized BOVIGENIX Variable": mapped})
            
            if col_map_summary:
                st.dataframe(pd.DataFrame(col_map_summary), use_container_width=True, hide_index=True)
            else:
                st.warning("No column headers matched our standard names. Please ensure columns include 'cow_id' and 'timestamp'.")

            # 4. Data Quality & Biological Range Validation
            st.subheader("2. Data Quality & Validation Summary")
            report = validate_dataset(clean_df)

            q1, q2, q3 = st.columns(3)
            with q1:
                st.metric("Total Records", f"{report['total_rows']}")
            with q2:
                dup_count = report["duplicate_count"]
                st.metric("Duplicate Rows", f"{dup_count}", delta="- Action needed" if dup_count > 0 else "Clean", delta_color="inverse")
            with q3:
                cols_found = len(report["columns_found"])
                st.metric("Recognized Standard Variables", f"{cols_found} / {len(STANDARD_COLUMNS)}")

            # Mandatory column check
            if not report["is_valid"]:
                st.error(f"Validation Error: Mandatory column(s) missing: {', '.join(report['missing_required'])}. Cannot import.")
            else:
                st.success("Validation Passed: Mandatory columns ('cow_id', 'timestamp') are present.")

            # Biological warnings
            if report["biological_warnings"]:
                with st.expander("⚠️ Biological Range Warnings", expanded=True):
                    for w in report["biological_warnings"]:
                        st.warning(w)

            # Missing values summary
            with st.expander("📊 Missing Values Breakdown"):
                st.dataframe(report["missing_value_summary"], use_container_width=True)

            # 5. Database Storage Controls
            st.subheader("3. Ingest into BOVIGENIX Database")

            if report["is_valid"]:
                c1, c2 = st.columns(2)
                with c1:
                    auto_register = st.checkbox("Automatically register new cows found in dataset if not already registered", value=True)
                with c2:
                    drop_dups = st.checkbox("Automatically drop duplicate (cow_id, timestamp) rows", value=True)

                if st.button("Store In Database", type="primary"):
                    with st.spinner("Processing and storing observations..."):
                        save_df = clean_df.copy()

                        # Drop duplicates if checked
                        if drop_dups and "cow_id" in save_df.columns and "timestamp" in save_df.columns:
                            save_df = save_df.drop_duplicates(subset=["cow_id", "timestamp"])

                        # Auto-register newly discovered cows if checked
                        if auto_register:
                            unique_cows = save_df["cow_id"].unique()
                            for cid in unique_cows:
                                cid_str = str(cid).strip().upper()
                                if not database.get_cow_by_id(cid_str):
                                    breed = save_df[save_df["cow_id"] == cid]["breed"].iloc[0] if "breed" in save_df.columns else "Crossbred"
                                    parity = save_df[save_df["cow_id"] == cid]["parity"].iloc[0] if "parity" in save_df.columns else 1
                                    database.add_cow(
                                        cow_id=cid_str,
                                        name=f"Cow {cid_str}",
                                        breed=str(breed),
                                        dob="2021-01-01",
                                        parity=int(parity) if pd.notnull(parity) else 1,
                                        notes="Auto-registered during dataset upload"
                                    )

                        # Calculate THI if ambient temp & humidity exist
                        if "ambient_temperature" in save_df.columns and "relative_humidity" in save_df.columns:
                            save_df["thi"] = save_df.apply(
                                lambda row: calculate_thi(row.get("ambient_temperature"), row.get("relative_humidity")),
                                axis=1
                            )

                        # Filter dataframe to only standard database columns
                        cols_to_keep = [c for c in STANDARD_COLUMNS + ["thi", "is_synthetic"] if c in save_df.columns]
                        final_df = save_df[cols_to_keep]

                        success, count_or_err = database.bulk_insert_observations(final_df)
                        if success:
                            database.log_dataset_import(filename, len(final_df), "Success", f"Stored {len(final_df)} observations.")
                            st.success(f"Successfully stored {len(final_df)} observations into the database!")
                        else:
                            database.log_dataset_import(filename, 0, "Failed", str(count_or_err))
                            st.error(f"Import failed: {count_or_err}")

        except Exception as e:
            st.error(f"Error processing file: {str(e)}")


# ---------------------------------------------------------------------------
# TAB 2: EDUCATIONAL SAMPLE DATASET
# ---------------------------------------------------------------------------
with tab_sample:
    st.subheader("Educational Synthetic Cattle Dataset")
    st.markdown("""
    <div style="border: 1px solid #cbd5e1; border-left: 5px solid #0284c7; background-color: #ffffff; color: #0f172a; padding: 14px 18px; border-radius: 8px; font-size: 0.92rem; line-height: 1.6; box-shadow: 0 1px 4px rgba(0,0,0,0.04);">
        <strong style="color: #0369a1; font-size: 1.0rem;">Scientific & Ethical Transparency Note:</strong><br>
        This dataset is generated for testing and demonstration purposes and is explicitly labeled <code style="background-color:#e2e8f0; color:#0f172a; padding:2px 6px; border-radius:4px;">is_synthetic = 1</code>.<br>
        It models real physiological estrus cycles and health anomalies (mastitis, lethargy) to allow full testing of the machine learning and alert modules without requiring proprietary sensor hardware.
    </div>
    """, unsafe_allow_html=True)

    st.write("")

    # Check if pre-generated file exists
    sample_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "sample_cattle_sensor_data.csv")

    if os.path.exists(sample_file_path):
        sample_df = pd.read_csv(sample_file_path)
        st.write(f"Sample dataset available: **{len(sample_df)} observations** across {sample_df['cow_id'].nunique()} cows.")
        
        st.dataframe(sample_df.head(10), use_container_width=True)

        col_dl, col_load = st.columns(2)
        with col_dl:
            with open(sample_file_path, "rb") as f:
                st.download_button(
                    label="📥 Download Sample CSV Dataset",
                    data=f,
                    file_name="bovigenix_sample_cattle_sensors.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        with col_load:
            if st.button("Direct Ingest Sample Data to Database", type="primary", use_container_width=True):
                with st.spinner("Ingesting sample dataset..."):
                    cows_df, obs_df = generate_synthetic_cattle_dataset()
                    for _, cow in cows_df.iterrows():
                        database.add_cow(
                            cow_id=cow["cow_id"],
                            name=cow["name"],
                            breed=cow["breed"],
                            dob=cow["dob"],
                            parity=cow["parity"],
                            notes="Imported via Educational Sample"
                        )
                    database.bulk_insert_observations(obs_df)
                    database.log_dataset_import("sample_cattle_sensor_data.csv", len(obs_df), "Success", "Direct educational import")
                    st.success("Sample dataset ingested into database successfully!")
                    st.rerun()


# ---------------------------------------------------------------------------
# TAB 3: IMPORT AUDIT HISTORY
# ---------------------------------------------------------------------------
with tab_history:
    st.subheader("Dataset Ingestion History")
    imports_df = database.get_dataset_imports()

    if not imports_df.empty:
        st.dataframe(imports_df, use_container_width=True, hide_index=True)
    else:
        st.info("No datasets have been imported yet.")
