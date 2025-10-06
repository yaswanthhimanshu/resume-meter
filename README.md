# 💼 ResumeMeter  
**AI and Data-Driven Resume Scoring and Shortlisting**  
Developed by **Yaswanth Himanshu**

ResumeMeter is an AI-powered web app that evaluates and ranks resumes against job descriptions using semantic similarity.  
It helps recruiters quickly identify the most relevant candidates.

---

## 🚀 Features
- Upload resumes (PDF, DOCX, TXT)
- Extract skills and key sections
- Compute semantic similarity between resume and job description
- Automatically shortlist top candidates
- Connected to **Aiven Cloud MySQL** for secure data storage

---

## 🧠 Tech Stack
- **Frontend:** Streamlit  
- **Backend:** Python  
- **ML Model:** SentenceTransformer (`all-MiniLM-L6-v2`)  
- **Database:** MySQL (Aiven Cloud)  
- **Libraries:** Pandas, NumPy, scikit-learn, nltk, sentence-transformers

---

## ⚙️ Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
