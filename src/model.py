# src/model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.utils import resample
import joblib
import os

# --- Configuration ---
DATA_FILE_PATH = "../news aggregator folder/newsCorpora.csv"  # Updated Path
MODEL_DIR = "../tfidf_logreg_classifier"

CATEGORY_MAP = {'b': 0, 't': 1, 'e': 2, 'm': 3}
NUM_LABELS = len(CATEGORY_MAP)

def load_and_balance_data():
    """Loads, cleans, maps, and upsamples the data."""
    print("Loading data...")
    columns = ['ID', 'TITLE', 'URL', 'PUBLISHER', 'CATEGORY', 'STORY', 'HOSTNAME', 'TIMESTAMP']
    df = pd.read_csv(DATA_FILE_PATH, sep='\t', names=columns, encoding='utf-8', on_bad_lines='skip')

    df = df[['TITLE', 'CATEGORY']]
    df = df[df['CATEGORY'].isin(CATEGORY_MAP.keys())]
    df['LABEL'] = df['CATEGORY'].map(CATEGORY_MAP)
    
    # --- Upsampling Logic ---
    print("Balancing data...")
    TARGET_COUNT = df['CATEGORY'].value_counts().max()
    df_majority = df[df['CATEGORY'] == 'e']
    
    df_upsampled = []
    for cat in ['b', 't', 'm']:
        df_minority = df[df['CATEGORY'] == cat]
        df_upsampled.append(resample(df_minority, 
                                     replace=True, 
                                     n_samples=TARGET_COUNT, 
                                     random_state=42))

    df_balanced = pd.concat([df_majority] + df_upsampled)
    df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
    
    texts = df_balanced['TITLE'].tolist()
    labels = df_balanced['LABEL'].tolist()
    
    return texts, labels

def train_and_save():
    texts, labels = load_and_balance_data()
    
    # 1. Split Data
    X_train_raw, _, y_train, _ = train_test_split(
        texts, labels, test_size=0.1, random_state=42, stratify=labels
    )
    
    # 2. Feature Extraction (TF-IDF)
    print("Fitting TF-IDF Vectorizer...")
    vectorizer = TfidfVectorizer(max_features=10000, stop_words='english')
    X_train_tfidf = vectorizer.fit_transform(X_train_raw)
    
    # 3. Model Training
    print("Training Logistic Regression Model...")
    model = LogisticRegression(max_iter=500, random_state=42, solver='liblinear')
    model.fit(X_train_tfidf, y_train)

    # 4. Saving Artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODEL_DIR, 'logreg_model.pkl'))
    joblib.dump(vectorizer, os.path.join(MODEL_DIR, 'tfidf_vectorizer.pkl'))
    print(f"Model and Vectorizer saved to {MODEL_DIR}")

if __name__ == '__main__':
    train_and_save()