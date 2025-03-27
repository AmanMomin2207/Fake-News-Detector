import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib

# Load datasets
fake_df = pd.read_csv('Fake.csv')
true_df = pd.read_csv('True.csv')

# Add labels
fake_df['label'] = 0
true_df['label'] = 1

# Combine datasets
data = pd.concat([fake_df, true_df], ignore_index=True)

# Data Preprocessing
def preprocess_text(text):
    text = str(text).lower()
    text = ''.join(char for char in text if char.isalnum() or char.isspace())
    return text

data['text'] = data['text'].apply(preprocess_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(data['text']).toarray()

# Save the vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
print('Vectorizer saved successfully!')