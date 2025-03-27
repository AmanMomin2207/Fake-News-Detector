from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import nltk
import torch
import torch.nn as nn

nltk.download('stopwords')

app = Flask(__name__)

# CNN Model
class FakeNewsCNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FakeNewsCNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100, output_size)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# Load the model and vectorizer
try:
    model = joblib.load('cnn_model.pkl')
    vectorizer = joblib.load('vectorizer.pkl')
except FileNotFoundError:
    print("Error: cnn_model.pkl or vectorizer.pkl not found. Please ensure these files are in the correct directory.")
    model = None
    vectorizer = None

# Preprocessing function
def preprocess_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [word for word in words if word not in stop_words]
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None or vectorizer is None:
        return jsonify({'error': 'Model or vectorizer not loaded'}), 500

    text = request.form['text']
    processed_text = preprocess_text(text)
    try:
        vectorized_text = vectorizer.transform([processed_text])
        # Convert to tensor
        vectorized_text_tensor = torch.tensor(vectorized_text.toarray(), dtype=torch.float32)
        # Make prediction
        with torch.no_grad():
            prediction = model(vectorized_text_tensor)
            # Assuming the model outputs a probability for the 'fake' class
            # and the classes are ordered as [true, fake]
            if prediction[0][1] > 0.5:
                result = "True"
            else:
                result = "Fake"
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)