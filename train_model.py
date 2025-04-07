# train_svm.py
import pandas as pd
import re
import joblib
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import hstack

# 1. Load and clean data
print("Loading dataset...")
df = pd.read_csv("data\\MBTI 500.csv")

def clean_text(text):
    text = re.sub(r'http\S+', '', text)
    text = text.replace("|||", " ")  # Replace post separators
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove special characters
    return text.lower().strip()[:500]  # Truncate to 500 characters

print("Cleaning text data...")
df['cleaned_posts'] = df['posts'].apply(clean_text)

# 2. Feature Engineering
print("Creating TF-IDF features...")
tfidf = TfidfVectorizer(
    max_features=3000,
    ngram_range=(1, 2),
    stop_words='english',
    sublinear_tf=True
)
X_tfidf = tfidf.fit_transform(df['cleaned_posts'])

# Add text length as a numeric feature
text_lengths = df['cleaned_posts'].apply(len).values.reshape(-1, 1)

# Combine features using sparse matrices
X = hstack([X_tfidf, text_lengths])

# 3. Prepare Labels
print("Encoding labels...")
le = LabelEncoder()
y = le.fit_transform(df['type'])

# 4. Train SVM
print("Training SVM model...")
svm_model = SVC(
    kernel='linear',
    class_weight='balanced',
    C=1.0,
    probability=True,
    verbose=True
)
svm_model.fit(X, y)  # Train on full dataset

# 5. Save Models
print("\nSaving models...")
joblib.dump({
    'svm_model': svm_model,
    'tfidf_vectorizer': tfidf,
    'label_encoder': le,
}, 'models/big_mbti_svm_model.pkl')

print("Training complete! Model saved to models/big_mbti_svm_model.pkl")