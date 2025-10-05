# SkillSync — AI / Data-Driven Resume Matching

### Developed by: **Yaswanth Himanshu**

SkillSync is an AI-powered resume matching web application that helps recruiters find the best candidates efficiently using semantic similarity.

---

### 🚀 Features
- Resume parsing (PDF, DOCX, TXT)
- AI-based semantic similarity scoring using Sentence Transformers
- Shortlisting based on threshold score
- Streamlit UI with custom design
- SQLite database integration for resume storage

---

### 🧠 Tech Stack
- **Frontend:** Streamlit (Custom CSS UI)
- **Backend:** Python (Flask for DB + Streamlit for UI)
- **ML Model:** SentenceTransformer (`all-MiniLM-L6-v2`)
- **Database:** MySQL (via MySQL Workbench)
- **Libraries:** `PyPDF2`, `pdfplumber`, `docx`, `nltk`, `numpy`, `pandas`

---

### 📦 Setup & Run
```bash
pip install -r requirements.txt
streamlit run app.py
