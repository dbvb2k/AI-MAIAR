import os
import sys
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from loguru import logger
import config
import re
from tqdm import tqdm

# Ensure models folder exists
os.makedirs('models', exist_ok=True)

# Robust field matching
def normalize_field_name(name):
    return re.sub(r'[^a-z0-9]', '', name.lower())

def find_field(meta_keys, candidates):
    norm_keys = {normalize_field_name(k): k for k in meta_keys}
    for cand in candidates:
        norm_cand = normalize_field_name(cand)
        if norm_cand in norm_keys:
            return norm_keys[norm_cand]
    return None

def load_training_data():
    texts = []
    labels = []
    file_list = list(config.file_list)
    for fname in tqdm(file_list, desc="Files", unit="file"):
        file_path = os.path.join(config.data_folder, fname)
        ext = os.path.splitext(file_path)[1].lower()
        try:
            if ext == '.csv':
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                except UnicodeDecodeError:
                    df = pd.read_csv(file_path, encoding='latin1')
            elif ext in ['.xls', '.xlsx']:
                df = pd.read_excel(file_path, engine=None)
            else:
                logger.warning(f"Skipping unsupported file: {file_path}")
                continue
        except Exception as e:
            logger.error(f"Failed to load {file_path}: {e}")
            continue
        # Find fields for features and label
        feature_fields = config.fields_to_embed
        app_candidates = [
            'Service*+', 'Service Category', 'Service', 'Application', 'Classification', 'Group'
        ]
        app_field = find_field(df.columns, app_candidates)
        if not app_field:
            logger.warning(f"No application field found in {file_path}, skipping file.")
            continue
        # Build features and labels
        for _, row in tqdm(df.iterrows(), total=len(df), desc=f"Rows in {fname}", unit="row"):
            parts = []
            for field in feature_fields:
                col = find_field(df.columns, [field])
                val = str(row[col]).strip() if col and pd.notnull(row[col]) else ''
                parts.append(val)
            text = " ".join(parts)
            text = " ".join(text.split()).lower()
            label = str(row[app_field]).strip()
            if not label:
                continue  # skip rows with empty label
            texts.append(text)
            labels.append(label)
    return texts, labels

def main():
    logger.info("Loading training data...")
    texts, labels = load_training_data()
    logger.info(f"Loaded {len(texts)} samples for training.")
    if not texts:
        print("No training data found. Exiting.")
        return
    # Vectorize text
    vectorizer = TfidfVectorizer(max_features=2000)
    X = vectorizer.fit_transform(texts)
    y = labels
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    n_estimators = 100
    clf = RandomForestClassifier(n_estimators=1, warm_start=True, random_state=42)
    # tqdm progress bar for training
    for i in tqdm(range(1, n_estimators + 1), desc="Training Random Forest", unit="tree"):
        clf.set_params(n_estimators=i)
        clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Random Forest accuracy on test set: {acc:.4f}")
    print(classification_report(y_test, y_pred, zero_division=0))
    # Save model and vectorizer
    joblib.dump(clf, 'models/rf_classifier.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("Model and vectorizer saved to models/ directory.")
    # Print class distribution
    from collections import Counter
    print("Class distribution:")
    for label, count in Counter(labels).most_common():
        print(f"  {label}: {count}")

if __name__ == "__main__":
    main() 