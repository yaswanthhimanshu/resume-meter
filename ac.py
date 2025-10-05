import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import os


df = pd.read_csv("sample_resumes_dataset.csv")


if 'resume_file' not in df.columns or 'Category' not in df.columns:
    raise ValueError("CSV must have 'resume_file' and 'Category' columns")


def read_resume(path):
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            return str(path)  
    except:
        return str(path)

df['resume_text'] = df['resume_file'].apply(read_resume)


embedding_models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]


classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Linear SVC": LinearSVC(max_iter=3000)
}


X_train, X_test, y_train, y_test = train_test_split(
    df['resume_text'], df['Category'], test_size=0.2, random_state=42
)


for emb_model_name in embedding_models:
    print(f"\n🔹 Using Embeddings from: {emb_model_name}")
    emb_model = SentenceTransformer(emb_model_name)

    
    X_train_emb = emb_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_emb = emb_model.encode(X_test.tolist(), show_progress_bar=True)

    for clf_name, clf in classifiers.items():
        clf.fit(X_train_emb, y_train)
        y_pred = clf.predict(X_test_emb)
        accuracy = accuracy_score(y_test, y_pred) * 100
        print(f"   ✅ {clf_name} Accuracy: {accuracy:.2f}%")
