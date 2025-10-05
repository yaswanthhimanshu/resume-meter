# db.py — automatically creates database & table if missing
import os
import json
import mysql.connector
from mysql.connector import Error
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": int(os.getenv("DB_PORT", 3306)),
    "user": os.getenv("DB_USER", "root"),
    "password": os.getenv("DB_PASSWORD", ""),
    "database": os.getenv("DB_NAME", "resume_db")
}


def _connect_no_db():
    """Connect without selecting a database (for CREATE DATABASE)."""
    cfg = DB_CONFIG.copy()
    cfg.pop("database", None)
    return mysql.connector.connect(
        host=cfg["host"],
        port=cfg["port"],
        user=cfg["user"],
        password=cfg["password"],
        autocommit=True
    )


def _create_database_if_missing():
    try:
        conn = _connect_no_db()
        cursor = conn.cursor()
        cursor.execute(
            f"CREATE DATABASE IF NOT EXISTS `{DB_CONFIG['database']}` "
            "DEFAULT CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;"
        )
        cursor.close()
        conn.close()
        print(f"✅ Database '{DB_CONFIG['database']}' ensured.")
        return True
    except Exception as e:
        print("❌ Error creating database:", e)
        return False


def init_db():
    """Connect to MySQL; create DB/table if missing; return connection."""
    try:
        conn = mysql.connector.connect(**DB_CONFIG)
    except Error as e:
        if getattr(e, "errno", None) == 1049 or "Unknown database" in str(e):
            if _create_database_if_missing():
                conn = mysql.connector.connect(**DB_CONFIG)
            else:
                print("❌ Could not create database automatically.")
                return None
        else:
            print("❌ Error while connecting to MySQL:", e)
            return None

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
                )
            """)
            conn.commit()
            cursor.close()
            print("✅ Table 'shortlisted_resumes' ensured.")
        return conn
    except Exception as e:
        print("❌ Error ensuring table exists:", e)
        conn.close()
        return None


def insert_resume(candidate_name, file_name, score, best_sentence_score, top_sentences, resume_text):
    try:
        conn = init_db()
        if conn:
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


def fetch_resumes(min_score=0.0, search_name=""):
    try:
        conn = init_db()
        resumes = []
        if conn:
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