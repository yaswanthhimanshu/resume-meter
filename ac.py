# Import required libraries for data handling, ML models, and embeddings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sentence_transformers import SentenceTransformer
import os

# Load the sample dataset containing resume files and their respective categories
df = pd.read_csv("sample_resumes_dataset.csv")

# Ensure the CSV has the required columns before proceeding
if 'resume_file' not in df.columns or 'Category' not in df.columns:
    raise ValueError("CSV must have 'resume_file' and 'Category' columns")

# Function to read the contents of a resume file safely
def read_resume(path):
    try:
        # If the file path exists, read the file contents
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8", errors="ignore") as f:
                return f.read()
        else:
            # If not a path, just return it as text (in case it's already plain text)
            return str(path)
    except:
        # If any error occurs during file reading, return the path as string
        return str(path)

# Apply the function to all resume file paths to get the full resume text
df['resume_text'] = df['resume_file'].apply(read_resume)

# Define the list of sentence-transformer models to test embeddings
embedding_models = [
    "all-MiniLM-L6-v2",
    "all-mpnet-base-v2"
]

# Define a dictionary of classifiers to compare performance
classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=2000),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "Linear SVC": LinearSVC(max_iter=3000)
}

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    df['resume_text'], df['Category'], test_size=0.2, random_state=42
)

# Loop through each embedding model to test its performance
for emb_model_name in embedding_models:
    print(f"\n🔹 Using Embeddings from: {emb_model_name}")
    emb_model = SentenceTransformer(emb_model_name)  # Load the embedding model

    # Convert the resume text into numerical embeddings
    X_train_emb = emb_model.encode(X_train.tolist(), show_progress_bar=True)
    X_test_emb = emb_model.encode(X_test.tolist(), show_progress_bar=True)

    # Train and evaluate each classifier on the embeddings
    for clf_name, clf in classifiers.items():
        clf.fit(X_train_emb, y_train)  # Train the classifier
        y_pred = clf.predict(X_test_emb)  # Predict the test labels
        accuracy = accuracy_score(y_test, y_pred) * 100  # Calculate accuracy
        print(f"   ✅ {clf_name} Accuracy: {accuracy:.2f}%")  # Display result

