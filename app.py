import streamlit as st
from sentence_transformers import SentenceTransformer, util
import os, tempfile, io, uuid
from io import BytesIO
import PyPDF2, docx, pdfplumber
import nltk
from nltk.tokenize import sent_tokenize
import numpy as np
import pandas as pd
import json
import streamlit.components.v1 as components
from db import init_db, insert_resume, fetch_resumes

# ---------- Init ----------
# ✅ Robust NLTK punkt setup (local nltk_data + download fallback)
HERE = os.path.dirname(__file__)
NLTK_DATA_DIR = os.path.join(HERE, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    try:
        st.info("Downloading NLTK punkt tokenizer (first run).")
        nltk.download("punkt", download_dir=NLTK_DATA_DIR, quiet=True)
    except Exception as e:
        st.warning(f"Could not download NLTK punkt. Using regex fallback for sentence splitting. ({e})")
# ---------- End punkt setup ----------

# initialize DB safely (don't crash UI if DB isn't configured)
try:
    init_db()
except Exception as e:
    # show non-blocking warning
    st.warning("Database init failed; continuing without persistent DB. Error: " + str(e))

st.set_page_config(
    page_title="ResumeMeter — AI and Data-Driven Resume Scoring and Shortlisting",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ---------- Session state ----------
if "show_settings" not in st.session_state:
    st.session_state.show_settings = False
if "model_name" not in st.session_state:
    st.session_state.model_name = "all-MiniLM-L6-v2"
if "top_k" not in st.session_state:
    st.session_state.top_k = 10
if "score_threshold" not in st.session_state:
    st.session_state.score_threshold = 0.45

# ---------- Global CSS ----------
st.markdown("""
<style>
/* Hide Streamlit default header & top decoration & dev toolbar */
div[data-testid="stDecoration"]{ display:none !important; }
header[data-testid="stHeader"]{ display:none !important; }
div[data-testid="stToolbar"]{ display:none !important; }

/* NOTE: removed the forced light-theme rule so Streamlit's theme toggle (Light/Dark/System)
   can work correctly. Do NOT set `base = "light"` in .streamlit/config.toml if you want users
   to be able to switch themes. */

/* Content spacing (we control our own topbar) */
.block-container{ padding-top:0 !important; }

/* ---------- Topbar ---------- */
.topbar{
  position:sticky; top:0; z-index:1000;
  width:100%; background:#FFFFFF; border-bottom:1px solid #E7ECEA;
  padding:14px 0;
}
.topbar .row{display:flex; align-items:center; justify-content:space-between; gap:12px;}

/* Brand (bigger) */
.logo-pill{background:#42A779; display:grid; place-items:center; color:#fff; font-weight:800; letter-spacing:.4px;}
.logo-lg{width:56px;height:56px;border-radius:14px;font-size:20px;}
.brand{color:#111111;}
.brand-lg{font-size:32px; font-weight:800; letter-spacing:-.01em;}

/* Responsive tweak */
@media (max-width:860px){
  .logo-lg{ width:44px; height:44px; border-radius:12px; font-size:16px; }
  .brand-lg{ font-size:24px; }
}

/* Header button */
.header-btn .stButton>button{
  background:#42A779; color:#fff; border:none; border-radius:50%;
  width:40px; height:40px; font-size:18px; cursor:pointer;
  display:grid; place-items:center;
  box-shadow:0 6px 14px rgba(66,167,121,.18);
}
.header-btn .stButton>button:hover{ background:#2E8F65; }

/* Hero + sections */
.hero{padding:44px 0 12px; text-align:center;}
.hero h1{margin:0 0 8px; font-size:40px; line-height:1.1; font-weight:800; letter-spacing:-.02em;}
.hero .line2{color:#42A779;}
.hero .sub{color:#667479; max-width:780px; margin:0 auto 28px; font-size:18px;}
@media (max-width:860px){
  .hero h1{font-size:30px;}
}

/* Upload panel */
.upload-panel{
  background:#EAF6F1; border:2px dashed #CDE7DB; border-radius:16px;
  padding:34px 22px; text-align:center; max-width:880px; margin:0 auto 28px;
}
.icon-circle{width:64px;height:64px;border-radius:999px;background:#E3F2EB;
  display:grid;place-items:center;margin:0 auto 10px;}
.upload-title{margin:6px 0 8px; font-size:22px; font-weight:700;}
.muted{color:#667479;} .small{font-size:14px;}

/* Card */
.card{
  background:#FFFFFF; border:1px solid #E7ECEA; border-radius:16px;
  padding:22px; max-width:780px; margin:0 auto 28px; text-align:center;
  box-shadow:0 6px 18px rgba(17,24,39,.06);
}
.card h3{margin:6px 0 18px; font-size:22px;}
textarea{
  width:100%; height:140px; resize:vertical; border:1px solid #E7ECEA;
  border-radius:12px; padding:14px 16px; font-size:15px; color:#111111; outline:none;
  background:#FFFFFF;
}

/* Features */
.features{padding:12px 0 40px;}
.features h2{text-align:center; font-size:32px; margin:0 0 20px;}
.grid{display:grid; grid-template-columns:repeat(3,1fr); gap:18px; max-width:980px; margin:0 auto;}
.feature{background:#fff; border:1px solid #E7ECEA; border-radius:14px; padding:18px 18px 20px;
  box-shadow:0 6px 18px rgba(17,24,39,.06);}
.icon-tile{width:44px;height:44px;border-radius:10px;background:#E3F2EB;
  display:grid;place-items:center;margin-bottom:8px;}
.feature h4{margin:0 0 6px; font-size:20px;}
.feature p{margin:0; color:#667479;}

/* Drawer */
.drawer{
  position: fixed; top:80px; right:18px; z-index: 1100;
  width: 320px; max-width: 90vw;
  background: #WHITE; border:1px solid #E7ECEA; border-radius:16px;
  box-shadow:0 16px 40px rgba(17,24,39,.18);
  padding:18px;
}
.drawer h4{margin:0 0 10px; font-size:18px;}
.drawer .hint{color:#667479; font-size:13px; margin-bottom:10px;}

/* Footer */
.footer{border-top:1px solid #E7ECEA; padding:18px 0; background:#fff; margin-top:24px;}
.footer-row{display:flex; align-items:center; justify-content:space-between; gap:12px; flex-wrap:wrap;}

/* === Narrow rule: hide only the exact helper span you provided ===
   This selector came directly from your browser's 'Copy selector'.
   It hides only that single span element (the "Limit 200MB" helper),
   leaving all other UI intact.
*/
#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-zy6yx3.e4man114 > div > div:nth-child(6) > div > div > div > div.stColumn.st-emotion-cache-yqzq1d.e196pkbe1 > div > div > div > div > section > div > div > span.st-emotion-cache-1sct1q3.e16n7gab4 {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

selector = "#root > div:nth-child(1) > div.withScreencast > div > div > div > section > div.stMainBlockContainer.block-container.st-emotion-cache-zy6yx3.e4man114 > div > div:nth-child(6) > div > div > div > div.stColumn.st-emotion-cache-yqzq1d.e196pkbe1 > div > div > div > div > section > div > div > span.st-emotion-cache-1sct1q3.e16n7gab4"

# Build JS safely by concatenating a single literal Python string with the JSON-encoded selector.
js = (
    "<script>(function(){"
    "const sel = " + json.dumps(selector) + ";"
    "const desiredText = 'Limit 5 MB per file • PDF, DOCX, TXT';"

    # Try to hide exact selector (handles Streamlit re-renders)
    "function hideSelector(){"
      "try{"
        "const el = document.querySelector(sel);"
        "if(el){ el.style.display = 'none'; return true; }"
      "}catch(e){}"
      "return false;"
    "}"

    # Fallback: replace text nodes containing 200MB -> 5 MB (safe: modifies text nodes only)
    "function replaceTextNodes(root){"
      "const walker = document.createTreeWalker(root, NodeFilter.SHOW_TEXT, null, false);"
      "let node;"
      "const nodes = [];"
      "while(node = walker.nextNode()){ nodes.push(node); }"
      "for(const n of nodes){"
        "const t = n.nodeValue; if(!t || !t.trim()) continue;"
        "if(/\\b200\\s?MB\\b/i.test(t) || /Limit\\s*200\\s?MB/i.test(t)){"
          "let newTxt = t.replace(/\\b200\\s?MB\\b/ig, '5 MB');"
          "newTxt = newTxt.replace(/Limit\\s*5\\s?MB/ig, 'Limit 5 MB');"
          "if(/Limit\\s*5\\s?MB/i.test(newTxt) && !/per file/i.test(newTxt)){"
            "newTxt = newTxt + ' per file • PDF, DOCX, TXT';"
          "}"
          "try{ n.nodeValue = newTxt; }catch(e){}"
        "}"
      "}"
    "}"

    # initial attempts
    "hideSelector();"
    "setTimeout(hideSelector, 150);"
    "setTimeout(hideSelector, 600);"
    "replaceTextNodes(document.body);"
    "setTimeout(()=>replaceTextNodes(document.body), 200);"
    "setTimeout(()=>replaceTextNodes(document.body), 800);"

    # Observe DOM and reapply on changes
    "const ob = new MutationObserver(()=>{ hideSelector(); replaceTextNodes(document.body); });"
    "ob.observe(document.body, { childList:true, subtree:true });"

    "})();</script>"
)

components.html(js, height=0, scrolling=False)



# ---------- Header ----------
with st.container():
    c = st.columns([1, 8, 1])[1]
    with c:
        c1, csp, c2 = st.columns([4, 6, 1])
        with c1:
            st.markdown(
                '<div class="topbar"><div class="row">'
                '<div style="display:flex; align-items:center; gap:14px;">'
                '<div class="logo-pill logo-lg">RM</div>'
                '<div class="brand brand-lg">ResumeMeter</div>'
                '</div></div></div>',
                unsafe_allow_html=True
            )
        with c2:
            st.markdown('<div class="header-btn">', unsafe_allow_html=True)
            if st.button("»", key="toggle_drawer", help="Toggle Advanced Settings"):
                st.session_state.show_settings = not st.session_state.show_settings
            st.markdown('</div>', unsafe_allow_html=True)

# ---------- Hero ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        st.markdown(
            """
            <section class="hero">
              <h1><span>AI &amp; Data-Driven</span><br/><span class="line2">Resume Scoring</span></h1>
              <p class="sub">Measure and shortlist candidates with AI-powered scoring and explainable insights.</p>
            </section>
            """,
            unsafe_allow_html=True
        )

# ---------- Hint ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        st.markdown(
            '<p style="text-align:center; color:#667479; margin-top:-6px;">'
            'Tip: Use the <b>»</b> button (top-right) to open <b>Advanced Settings</b>.'
            '</p>',
            unsafe_allow_html=True
        )

# ---------- Advanced Settings Drawer ----------
if st.session_state.show_settings:
    st.markdown('<div class="drawer">', unsafe_allow_html=True)
    st.markdown('<h4>Advanced Settings</h4>', unsafe_allow_html=True)
    st.markdown('<div class="hint">Tune the matching to your needs.</div>', unsafe_allow_html=True)

    st.session_state.model_name = st.selectbox(
        "Embedding model",
        ["all-MiniLM-L6-v2", "all-mpnet-base-v2"],
        index=0 if st.session_state.model_name == "all-MiniLM-L6-v2" else 1
    )
    st.session_state.top_k = st.number_input(
        "Top K candidates", min_value=1, max_value=50, value=int(st.session_state.top_k)
    )
    st.session_state.score_threshold = st.slider(
        "Score threshold ≥", 0.0, 1.0, float(st.session_state.score_threshold)
    )
    st.markdown('</div>', unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource(show_spinner=False)
def load_model(name):
    return SentenceTransformer(name)

model = load_model(st.session_state.model_name)

# ---------- Upload Panel ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        uploader_label = "Upload Your Resume (Limit 5 MB per file • PDF, DOCX, TXT)"
        uploaded_files = st.file_uploader(
            uploader_label,
            type=["pdf", "docx", "txt"],
            accept_multiple_files=True,
            key="uploader_dup",
            label_visibility="visible"
        )

        # ---- File size limit (5 MB per file) ----
        MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB in bytes
        valid_files = []
        for f in uploaded_files or []:
            try:
                size = f.size
            except Exception:
                try:
                    f.seek(0, io.SEEK_END)
                    size = f.tell()
                    f.seek(0)
                except Exception:
                    size = None
            if size is None:
                valid_files.append(f)
            else:
                if size > MAX_FILE_SIZE:
                    st.warning(f"⚠️ {f.name} is too large ({size/1024/1024:.2f} MB). Max allowed: 5 MB.")
                else:
                    valid_files.append(f)
        uploaded_files = valid_files

# ---------- JD ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        jd_text = st.text_area("Paste the job description here", height=140)
        analyze = st.button("Analyze Match")

# ---------- Helpers ----------
def extract_text_from_pdf(file_bytes):
    text = ""
    try:
        with pdfplumber.open(BytesIO(file_bytes)) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception:
        try:
            reader = PyPDF2.PdfReader(BytesIO(file_bytes))
            for p in reader.pages:
                page_text = p.extract_text()
                if page_text:
                    text += page_text + "\n"
        except Exception as e:
            st.warning(f"PDF parsing error: {e}")
    return text

def extract_text_from_docx(file_bytes):
    text = ""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".docx") as tmp:
            tmp.write(file_bytes)
            tmp.flush()
            d = docx.Document(tmp.name)
            for p in d.paragraphs:
                if p.text:
                    text += p.text + "\n"
    finally:
        try:
            os.remove(tmp.name)
        except Exception:
            pass
    return text

def extract_text_from_txt(file_bytes):
    try:
        return file_bytes.decode(errors="ignore")
    except Exception:
        return str(file_bytes)

def parse_resume(uploaded_file):
    name = uploaded_file.name or "unknown_resume"
    raw = uploaded_file.read()
    ext = name.split(".")[-1].lower()
    text = ""
    if ext == "pdf":
        text = extract_text_from_pdf(raw)
    elif ext == "docx":
        text = extract_text_from_docx(raw)
    elif ext == "txt":
        text = extract_text_from_txt(raw)
    else:
        st.warning(f"Unsupported format: {name}")
    return name, text.strip()

# ---------- sentence_split (robust fallback) ----------
import re
def sentence_split(text):
    """
    Prefer NLTK sent_tokenize, but fall back to a regex-based splitter if punkt is missing.
    Returns a list of non-empty sentence strings with length > 10.
    """
    if not isinstance(text, str) or not text.strip():
        return []
    try:
        sents = sent_tokenize(text)
    except LookupError:
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        sents = [p for p in parts if p]
    return [s.strip() for s in sents if len(s.strip()) > 10]

def compute_resume_score(jd_embedding, resume_text, model):
    sents = sentence_split(resume_text)
    if not sents:
        return {"score": 0.0, "top_matches": [], "resume_sentences": []}
    sent_embs = model.encode(sents, convert_to_tensor=True, show_progress_bar=False)
    cos_scores = util.pytorch_cos_sim(jd_embedding, sent_embs)[0].cpu().numpy()
    best_idx = int(np.argmax(cos_scores))
    best_score = float(cos_scores[best_idx])
    top_n = min(5, len(cos_scores))
    overall = float(np.mean(sorted(cos_scores, reverse=True)[:top_n]))
    top_matches_idx = np.argsort(-cos_scores)[:top_n]
    top_matches = [{"sentence": sents[int(i)], "score": float(cos_scores[int(i)])} for i in top_matches_idx]
    return {"score": overall, "best_sentence_score": best_score, "top_matches": top_matches, "resume_sentences": sents}

# ---------- Run pipeline ----------
if analyze:
    if not uploaded_files:
        st.error("Please upload at least one resume.")
    elif not jd_text.strip():
        st.error("Please paste/enter a job description.")
    else:
        with st.spinner("Processing..."):
            jd_embedding = model.encode(jd_text, convert_to_tensor=True, show_progress_bar=False)
            results = []
            for f in uploaded_files:
                name, text = parse_resume(f)
                if not text:
                    results.append({"name": name, "score": 0.0, "top_matches": [], "error": "no text extracted"})
                    continue
                res = compute_resume_score(jd_embedding, text, model)
                res["name"] = name
                results.append(res)
            results_sorted = sorted(results, key=lambda x: x.get("score", 0.0), reverse=True)
            shortlisted = [r for r in results_sorted if r.get("score", 0.0) >= st.session_state.score_threshold][:int(st.session_state.top_k)]

        st.success(f"Processed {len(results)} resumes — shortlisted {len(shortlisted)}")

        for i, r in enumerate(shortlisted, start=1):
            unique_id = str(uuid.uuid4())[:8]
            unique_name = f"{i}_{unique_id}_{r['name']}"
            try:
                insert_resume(
                    unique_name,
                    r['name'],
                    r['score'],
                    r['best_sentence_score'],
                    r['top_matches'],
                    "\n".join(r['resume_sentences'])
                )
            except Exception as e:
                st.warning(f"Failed to save to DB: {e}")

        col1, col2 = st.columns([1, 1.2])
        with col1:
            st.header("Shortlisted Candidates")
            if not shortlisted:
                st.info("No candidate passed the threshold.")
            else:
                for i, r in enumerate(shortlisted, start=1):
                    st.subheader(f"{i}. {r['name']}")
                    st.write(f"Score: **{r['score']:.4f}** | Best sentence score: {r['best_sentence_score']:.4f}")
                    with st.expander("Top matching sentences"):
                        for tm in r["top_matches"]:
                            st.markdown(f"- ({tm['score']:.3f}) {tm['sentence']}")
                        st.download_button(
                            label="Download extracted resume text",
                            data="\n".join(r["resume_sentences"]),
                            file_name=f"{r['name']}_extracted.txt"
                        )
        with col2:
            st.header("All Candidates (ranked)")
            for i, r in enumerate(results_sorted, start=1):
                st.write(f"{i}. {r['name']} — score: **{r['score']:.4f}**")

# ---------- Features ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        st.markdown(
            """
            <section class="features">
              <h2>Features</h2>
              <div class="grid">
                <div class="feature"><div class="icon-tile"></div>
                  <h4>Resume Parsing</h4>
                  <p>Upload PDF, DOCX, and TXT files with intelligent text extraction.</p>
                </div>
                <div class="feature"><div class="icon-tile"></div>
                  <h4>AI Matching</h4>
                  <p>Advanced NLP models understand meaning beyond keywords.</p>
                </div>
                <div class="feature"><div class="icon-tile"></div>
                  <h4>Smart Scoring</h4>
                  <p>Matches resumes to job descriptions with explainable results.</p>
                </div>
              </div>
            </section>
            """,
            unsafe_allow_html=True
        )

# ---------- How to Use ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        st.markdown(
            """
            <section class="card" id="how-to-use-resumemeter">
              <h3>How to Use ResumeMeter</h3>
              <ol style="text-align:left; line-height:1.6; font-size:16px; color:#444;">
                <li><b>Upload Resume(s):</b> Drag and drop or browse to upload one or more resumes (PDF, DOCX, TXT).</li>
                <li><b>Paste Job Description:</b> Enter the JD of the role you’re hiring for.</li>
                <li><b>Analyze Match:</b> Click <b>Analyze Match</b> to rank resumes by semantic similarity.</li>
                <li><b>Review Results:</b> See scores and top matching sentences for explainability.</li>
                <li><b>Export:</b> Download shortlisted results as CSV/Excel or store them in the database.</li>
              </ol>

              <h4 style="margin-top:22px; display:flex; align-items:center; gap:8px; color:#111;">
                ⚙️ Advanced Settings
              </h4>
              <ul style="text-align:left; line-height:1.6; font-size:16px; color:#444;">
                <li><b>Embedding model:</b> <i>MiniLM</i> is faster; <i>mpnet</i> is usually more accurate.</li>
                <li><b>Top K candidates:</b> How many ranked resumes to show.</li>
                <li><b>Score threshold:</b> Minimum similarity a resume must meet to be shortlisted.</li>
              </ul>

              <p style="margin-top:10px; font-size:15px; color:#667479;">
                Use the <b>»</b> button (top-right) to toggle the Advanced Settings drawer.
              </p>
            </section>
            """,
            unsafe_allow_html=True
        )

# ---------- Footer ----------
with st.container():
    col = st.columns([1, 8, 1])[1]
    with col:
        st.markdown(
            """
            <footer class="footer">
              <div class="footer-row">
                <div style="display:flex;align-items:center;gap:10px">
                  <div class="logo-pill" style="width:30px;height:30px;border-radius:8px;font-size:12px">RM</div>
                  <div class="brand">ResumeMeter</div>
                </div>
                <div style="color:#667479; font-size:14px;">© 2025 ResumeMeter. Developed by Yaswanth Himanshu</div>
              </div>
            </footer>
            """,
            unsafe_allow_html=True
        )
