# db.py — Aiven-compatible MySQL helper (uses ssl_ca when provided)
import os
import json
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv
from pathlib import Path

# Force load .env from this file's directory (robust)
load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")

# Build raw config from environment (keep ssl_ca key name: mysql-connector expects 'ssl_ca')
_RAW_DB = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)) if os.getenv("DB_PORT") else 3306,
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "resume_db"),
    "ssl_ca": os.getenv("DB_SSL_CERT")  # path to aiven-ca.pem or None
}

# Filter out empty values to avoid passing None to mysql.connector
DB_CONFIG = {k: v for k, v in _RAW_DB.items() if v not in (None, "", "None")}

def _connect_no_db():
    """
    Connect without selecting a database (used to run CREATE DATABASE).
    Keeps ssl_ca if present in DB_CONFIG.
    """
    cfg = DB_CONFIG.copy()
    cfg.pop("database", None)
    # build connect args explicitly to avoid passing unexpected values
    connect_args = {
        "host": cfg.get("host", "localhost"),
        "port": int(cfg.get("port", 3306)),
        "user": cfg.get("user", "root"),
        "password": cfg.get("password", ""),
        "autocommit": True
    }
    if cfg.get("ssl_ca"):
        connect_args["ssl_ca"] = cfg.get("ssl_ca")
    return mysql.connector.connect(**connect_args)

def get_connection():
    """
    Return a mysql.connector connection using DB_CONFIG.
    Caller should close the connection when done.
    """
    try:
        connect_args = {
            "host": DB_CONFIG.get("host", "localhost"),
            "port": int(DB_CONFIG.get("port", 3306)),
            "user": DB_CONFIG.get("user", "root"),
            "password": DB_CONFIG.get("password", ""),
            "database": DB_CONFIG.get("database")
        }
        # include ssl_ca only if present
        if DB_CONFIG.get("ssl_ca"):
            connect_args["ssl_ca"] = DB_CONFIG.get("ssl_ca")
        conn = mysql.connector.connect(**connect_args)
        return conn
    except Exception as e:
        print("❌ Error creating MySQL connection:", e)
        return None

def _create_database_if_missing():
    """
    Create the target database if it does not exist.
    Requires that the configured user has CREATE DATABASE privilege.
    """
    try:
        conn = _connect_no_db()
        cursor = conn.cursor()
        target = DB_CONFIG.get("database")
        if not target:
            print("❌ No database name configured; cannot create database.")
            cursor.close()
            conn.close()
            return False
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{target}` DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        cursor.close()
        conn.close()
        print(f"✅ Database '{target}' ensured.")
        return True
    except Exception as e:
        print("❌ Error creating database:", e)
        return False

def init_db():
    """
    Ensure the database and the main table exist. Returns a live connection (or None).
    NOTE: Caller should close the returned connection when finished.
    """
    # Try a normal connection first
    try:
        conn = get_connection()
        if conn is None:
            raise Error("Connection creation returned None")
    except Error as e:
        # Handle unknown database (error 1049) by creating DB and retrying
        msg = str(e)
        if getattr(e, "errno", None) == 1049 or "Unknown database" in msg or "1049" in msg:
            print("⚠️ Database not found. Attempting to create it...")
            if _create_database_if_missing():
                conn = get_connection()
                if conn is None:
                    print("❌ Still could not connect after creating database.")
                    return None
            else:
                return None
        else:
            print("❌ Error while connecting to MySQL:", e)
            return None

    # Ensure table exists
    try:
        if conn.is_connected():
            cursor = conn.cursor()
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS shortlisted_resumes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    candidate_name VARCHAR(255) UNIQUE,
                    file_name VARCHAR(255),
                    score FLOAT,
                    best_sentence_score FLOAT,
                    top_sentences JSON,
                    resume_text LONGTEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) CHARACTER SET utf8mb4;
            """)
            conn.commit()
            cursor.close()
            print("✅ Table 'shortlisted_resumes' ensured.")
        return conn
    except Exception as e:
        print("❌ Error ensuring table exists:", e)
        try:
            conn.close()
        except Exception:
            pass
        return None

def insert_resume(candidate_name, file_name, score, best_sentence_score, top_sentences, resume_text):
    """
    Insert or update a shortlisted resume entry.
    `top_sentences` should be serializable (we store as JSON).
    """
    try:
        conn = get_connection()
        if not conn:
            print("❌ No DB connection available for insert.")
            return
        cursor = conn.cursor()
        cursor.execute("""
            INSERT INTO shortlisted_resumes
            (candidate_name, file_name, score, best_sentence_score, top_sentences, resume_text)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON DUPLICATE KEY UPDATE
                score = VALUES(score),
                best_sentence_score = VALUES(best_sentence_score),
                top_sentences = VALUES(top_sentences),
                resume_text = VALUES(resume_text),
                created_at = CURRENT_TIMESTAMP
        """, (
            candidate_name,
            file_name,
            score,
            best_sentence_score,
            json.dumps(top_sentences, ensure_ascii=False),
            resume_text
        ))
        conn.commit()
        cursor.close()
        conn.close()
    except Error as e:
        print("❌ Error inserting resume:", e)
    except Exception as e:
        print("❌ Unexpected error inserting resume:", e)

def fetch_resumes(min_score=0.0, search_name=""):
    """
    Fetch resumes with score >= min_score. Optionally filter by candidate_name LIKE search_name.
    Returns a list of dicts (using cursor(dictionary=True)).
    """
    try:
        conn = get_connection()
        resumes = []
        if not conn:
            print("❌ No DB connection available for fetch.")
            return resumes
        cursor = conn.cursor(dictionary=True)
        query = "SELECT * FROM shortlisted_resumes WHERE score >= %s"
        params = [min_score]

        if search_name:
            query += " AND candidate_name LIKE %s"
            params.append(f"%{search_name}%")

        query += " ORDER BY score DESC"
        cursor.execute(query, tuple(params))
        resumes = cursor.fetchall()
        cursor.close()
        conn.close()
        return resumes
    except Error as e:
        print("❌ Error fetching resumes:", e)
        return []
    except Exception as e:
        print("❌ Unexpected error fetching resumes:", e)
        return []
