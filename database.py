"""
BOVIGENIX - Database Layer (database.py)
----------------------------------------
This module manages all database interactions for the BOVIGENIX application.

Concepts for Students:
1. SQLite: A lightweight, serverless relational database built right into Python.
   Data is stored in a single file on your computer (bovigenix.db).
2. Tables: Like spreadsheets with rows (records) and columns (attributes).
3. Primary Key: A unique identifier for every row (e.g., cow_id or a unique integer id).
4. Foreign Key: A column that links a record to another table (e.g., cow_id in the alerts
   table links to cow_id in the cows table).
5. CRUD Operations:
   - Create (INSERT INTO ...)
   - Read (SELECT ... FROM ...)
   - Update (UPDATE ... SET ...)
   - Delete (DELETE FROM ...)
"""

import sqlite3
import os
import pandas as pd
from datetime import datetime

# Path to the local SQLite database file
DB_FILE = os.path.join(os.path.dirname(__file__), "bovigenix.db")


def get_connection():
    """
    Establishes and returns a connection to the SQLite database.
    We set timeout=10.0 to prevent database locked errors when multiple
    pages read/write simultaneously.
    """
    conn = sqlite3.connect(DB_FILE, timeout=10.0)
    # Enable Foreign Key enforcement in SQLite
    conn.execute("PRAGMA foreign_keys = ON;")
    # Return rows as dictionary-like objects for easy access by column name
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """
    Creates all necessary tables if they do not already exist.
    This function is run once when the application starts up.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        # 1. Cows Table: Holds master profiles for each animal
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS cows (
            cow_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            breed TEXT NOT NULL,
            dob TEXT NOT NULL,
            parity INTEGER NOT NULL DEFAULT 0,
            status TEXT NOT NULL DEFAULT 'Active',
            notes TEXT,
            created_at TEXT NOT NULL
        );
        """)

        # 2. Sensor Observations Table: Holds time-series physiological & behavioral data
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS sensor_observations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            activity REAL,
            rumination_time REAL,
            lying_time REAL,
            eating_time REAL,
            milk_yield REAL,
            body_temperature REAL,
            ambient_temperature REAL,
            relative_humidity REAL,
            thi REAL,
            estrus_label INTEGER,
            health_label TEXT,
            is_synthetic INTEGER DEFAULT 0,
            FOREIGN KEY (cow_id) REFERENCES cows(cow_id) ON DELETE CASCADE
        );
        """)

        # 3. Reproductive Records Table: Artificial Insemination, pregnancy checks, calving
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS reproductive_records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_id TEXT NOT NULL,
            record_type TEXT NOT NULL,
            event_date TEXT NOT NULL,
            result TEXT,
            follow_up_date TEXT,
            technician_or_vet TEXT,
            remarks TEXT,
            created_at TEXT NOT NULL,
            FOREIGN KEY (cow_id) REFERENCES cows(cow_id) ON DELETE CASCADE
        );
        """)

        # 4. Alerts Table: Decision-support alerts for estrus, health, heat stress, follow-ups
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cow_id TEXT NOT NULL,
            alert_type TEXT NOT NULL,
            alert_date TEXT NOT NULL,
            severity TEXT NOT NULL,
            message TEXT NOT NULL,
            suggested_action TEXT,
            status TEXT NOT NULL DEFAULT 'Pending',
            reviewer_name TEXT,
            reviewer_notes TEXT,
            reviewed_at TEXT,
            FOREIGN KEY (cow_id) REFERENCES cows(cow_id) ON DELETE CASCADE
        );
        """)

        # 5. Dataset Import History Table: Tracks uploaded files
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_imports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT NOT NULL,
            import_date TEXT NOT NULL,
            row_count INTEGER NOT NULL,
            status TEXT NOT NULL,
            notes TEXT
        );
        """)

        # 6. ML Model Evaluations Table: Records model training performance
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_evaluations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name TEXT NOT NULL,
            train_date TEXT NOT NULL,
            accuracy REAL,
            precision_score REAL,
            recall_score REAL,
            f1_score REAL,
            sample_count INTEGER,
            features_used TEXT,
            notes TEXT
        );
        """)

        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# COW CRUD FUNCTIONS
# ---------------------------------------------------------------------------

def add_cow(cow_id, name, breed, dob, parity, status="Active", notes=""):
    """
    Registers a new cow in the database.
    Returns True if successful, or False if cow_id already exists.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    try:
        cursor.execute("""
            INSERT INTO cows (cow_id, name, breed, dob, parity, status, notes, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (cow_id.strip().upper(), name.strip(), breed.strip(), dob, int(parity), status, notes.strip(), now_str))
        conn.commit()
        return True, "Cow registered successfully!"
    except sqlite3.IntegrityError:
        return False, f"Cow ID '{cow_id}' already exists in the system."
    except Exception as e:
        return False, f"Database error: {str(e)}"
    finally:
        conn.close()


def update_cow(cow_id, name, breed, dob, parity, status, notes):
    """
    Updates an existing cow's information.
    """
    conn = get_connection()
    cursor = conn.cursor()

    try:
        cursor.execute("""
            UPDATE cows
            SET name = ?, breed = ?, dob = ?, parity = ?, status = ?, notes = ?
            WHERE cow_id = ?
        """, (name.strip(), breed.strip(), dob, int(parity), status, notes.strip(), cow_id.strip().upper()))
        conn.commit()
        return True, f"Cow {cow_id} updated successfully."
    except Exception as e:
        return False, f"Failed to update cow: {str(e)}"
    finally:
        conn.close()


def get_all_cows(status_filter=None):
    """
    Retrieves all cows from the database as a Pandas DataFrame.
    Can filter by status (e.g. 'Active', 'Inactive', 'Sold', 'Deceased').
    """
    conn = get_connection()
    try:
        if status_filter and status_filter != "All":
            query = "SELECT * FROM cows WHERE status = ? ORDER BY cow_id ASC"
            df = pd.read_sql_query(query, conn, params=(status_filter,))
        else:
            query = "SELECT * FROM cows ORDER BY cow_id ASC"
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_cow_by_id(cow_id):
    """
    Returns single cow profile as a dictionary or None if not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT * FROM cows WHERE cow_id = ?", (cow_id.strip().upper(),))
        row = cursor.fetchone()
        if row:
            return dict(row)
        return None
    finally:
        conn.close()


def get_cows_summary_stats():
    """
    Returns high-level counts for the dashboard:
    Total cows, active cows, inactive/sold/deceased counts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM cows")
        total = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cows WHERE status = 'Active'")
        active = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cows WHERE status = 'Inactive'")
        inactive = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cows WHERE status = 'Sold'")
        sold = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM cows WHERE status = 'Deceased'")
        deceased = cursor.fetchone()[0]

        return {
            "total": total,
            "active": active,
            "inactive": inactive,
            "sold": sold,
            "deceased": deceased
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# SENSOR OBSERVATIONS FUNCTIONS
# ---------------------------------------------------------------------------

def add_sensor_observation(cow_id, timestamp, activity, rumination, lying, eating,
                           milk_yield, body_temp, ambient_temp, humidity, thi,
                           estrus_label=None, health_label=None, is_synthetic=0):
    """
    Inserts a single sensor observation record.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("""
            INSERT INTO sensor_observations
            (cow_id, timestamp, activity, rumination_time, lying_time, eating_time,
             milk_yield, body_temperature, ambient_temperature, relative_humidity,
             thi, estrus_label, health_label, is_synthetic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (cow_id, timestamp, activity, rumination, lying, eating, milk_yield,
              body_temp, ambient_temp, humidity, thi, estrus_label, health_label, is_synthetic))
        conn.commit()
        return True
    finally:
        conn.close()


def bulk_insert_observations(df):
    """
    Efficiently inserts multiple sensor observations from a pandas DataFrame.
    """
    conn = get_connection()
    try:
        # Use pandas to_sql in append mode
        df.to_sql("sensor_observations", conn, if_exists="append", index=False)
        return True, len(df)
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


def get_observations(cow_id=None, limit=1000):
    """
    Retrieves sensor observations. If cow_id is given, filters by that cow.
    Returns a pandas DataFrame sorted chronologically.
    """
    conn = get_connection()
    try:
        if cow_id:
            query = """
                SELECT * FROM sensor_observations
                WHERE cow_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(cow_id, limit))
        else:
            query = """
                SELECT * FROM sensor_observations
                ORDER BY timestamp DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    finally:
        conn.close()


def get_observations_for_ml():
    """
    Retrieves all sensor observations along with cow profile details
    (breed, age, parity) joined for training machine learning models.
    """
    conn = get_connection()
    try:
        query = """
            SELECT 
                s.*,
                c.breed,
                c.dob,
                c.parity,
                c.status as cow_status
            FROM sensor_observations s
            JOIN cows c ON s.cow_id = c.cow_id
            ORDER BY s.cow_id, s.timestamp ASC
        """
        df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# REPRODUCTIVE RECORDS FUNCTIONS
# ---------------------------------------------------------------------------

def add_reproductive_record(cow_id, record_type, event_date, result="",
                            follow_up_date=None, technician_or_vet="", remarks=""):
    """
    Logs an estrus observation, artificial insemination, pregnancy diagnosis, or calving event.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT INTO reproductive_records
            (cow_id, record_type, event_date, result, follow_up_date, technician_or_vet, remarks, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (cow_id.strip().upper(), record_type, event_date, result,
              follow_up_date, technician_or_vet, remarks, now_str))
        conn.commit()
        return True, "Reproductive event recorded successfully."
    except Exception as e:
        return False, f"Failed to record reproductive event: {str(e)}"
    finally:
        conn.close()


def get_reproductive_records(cow_id=None):
    """
    Retrieves reproductive history records, optionally filtered by cow_id.
    """
    conn = get_connection()
    try:
        if cow_id:
            query = """
                SELECT r.*, c.name, c.breed
                FROM reproductive_records r
                JOIN cows c ON r.cow_id = c.cow_id
                WHERE r.cow_id = ?
                ORDER BY r.event_date DESC
            """
            df = pd.read_sql_query(query, conn, params=(cow_id,))
        else:
            query = """
                SELECT r.*, c.name, c.breed
                FROM reproductive_records r
                JOIN cows c ON r.cow_id = c.cow_id
                ORDER BY r.event_date DESC
            """
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()


def get_upcoming_follow_ups():
    """
    Retrieves upcoming scheduled reproductive follow-ups (e.g. pregnancy tests,
    return-to-estrus checks, expected calving).
    """
    conn = get_connection()
    try:
        today_str = datetime.now().strftime("%Y-%m-%d")
        query = """
            SELECT r.*, c.name, c.breed
            FROM reproductive_records r
            JOIN cows c ON r.cow_id = c.cow_id
            WHERE r.follow_up_date IS NOT NULL 
              AND r.follow_up_date != ''
              AND r.follow_up_date >= ?
            ORDER BY r.follow_up_date ASC
            LIMIT 20
        """
        df = pd.read_sql_query(query, conn, params=(today_str,))
        return df
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# ALERTS & REVIEW FUNCTIONS
# ---------------------------------------------------------------------------

def add_alert(cow_id, alert_type, severity, message, suggested_action=""):
    """
    Inserts a newly generated alert into the database.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT INTO alerts
            (cow_id, alert_type, alert_date, severity, message, suggested_action, status)
            VALUES (?, ?, ?, ?, ?, ?, 'Pending')
        """, (cow_id.strip().upper(), alert_type, now_str, severity, message, suggested_action))
        conn.commit()
        return True
    except Exception as e:
        print(f"Error creating alert: {e}")
        return False
    finally:
        conn.close()


def get_alerts(status_filter=None, limit=200):
    """
    Retrieves alerts with cow details, optionally filtered by status:
    'Pending', 'Confirmed', 'False Positive', 'Inconclusive'.
    """
    conn = get_connection()
    try:
        if status_filter and status_filter != "All":
            query = """
                SELECT a.*, c.name, c.breed
                FROM alerts a
                JOIN cows c ON a.cow_id = c.cow_id
                WHERE a.status = ?
                ORDER BY a.alert_date DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(status_filter, limit))
        else:
            query = """
                SELECT a.*, c.name, c.breed
                FROM alerts a
                JOIN cows c ON a.cow_id = c.cow_id
                ORDER BY a.alert_date DESC
                LIMIT ?
            """
            df = pd.read_sql_query(query, conn, params=(limit,))
        return df
    finally:
        conn.close()


def update_alert_review(alert_id, status, reviewer_name, reviewer_notes):
    """
    Records a reviewer's decision on an alert (Confirm, False Positive, Inconclusive).
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            UPDATE alerts
            SET status = ?, reviewer_name = ?, reviewer_notes = ?, reviewed_at = ?
            WHERE id = ?
        """, (status, reviewer_name.strip(), reviewer_notes.strip(), now_str, alert_id))
        conn.commit()
        return True, "Alert status updated successfully."
    except Exception as e:
        return False, f"Failed to update alert: {str(e)}"
    finally:
        conn.close()


def get_alerts_summary_stats():
    """
    Returns counts of pending alerts, confirmed alerts, and categorized counts.
    """
    conn = get_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("SELECT COUNT(*) FROM alerts WHERE status = 'Pending'")
        pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_type = 'Estrus Prediction' AND status = 'Pending'")
        estrus_pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_type = 'Health Risk' AND status = 'Pending'")
        health_pending = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM alerts WHERE alert_type = 'Heat Stress' AND status = 'Pending'")
        heat_pending = cursor.fetchone()[0]

        return {
            "pending_total": pending,
            "pending_estrus": estrus_pending,
            "pending_health": health_pending,
            "pending_heat": heat_pending
        }
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# DATASET IMPORTS & MODEL EVALUATIONS
# ---------------------------------------------------------------------------

def log_dataset_import(filename, row_count, status="Success", notes=""):
    """
    Logs an uploaded file event in the database for auditing and history.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT INTO dataset_imports (filename, import_date, row_count, status, notes)
            VALUES (?, ?, ?, ?, ?)
        """, (filename, now_str, row_count, status, notes))
        conn.commit()
    finally:
        conn.close()


def get_dataset_imports():
    """
    Retrieves dataset import history.
    """
    conn = get_connection()
    try:
        query = "SELECT * FROM dataset_imports ORDER BY import_date DESC"
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()


def save_model_evaluation(model_name, accuracy, precision_val, recall_val, f1_val,
                          sample_count, features_used, notes=""):
    """
    Saves ML model metrics to the database to compare model iterations.
    """
    conn = get_connection()
    cursor = conn.cursor()
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    try:
        cursor.execute("""
            INSERT INTO model_evaluations
            (model_name, train_date, accuracy, precision_score, recall_score, f1_score,
             sample_count, features_used, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (model_name, now_str, accuracy, precision_val, recall_val, f1_val,
              sample_count, features_used, notes))
        conn.commit()
    finally:
        conn.close()


def get_model_evaluations():
    """
    Retrieves all logged machine learning evaluation records.
    """
    conn = get_connection()
    try:
        query = "SELECT * FROM model_evaluations ORDER BY train_date DESC"
        return pd.read_sql_query(query, conn)
    finally:
        conn.close()
