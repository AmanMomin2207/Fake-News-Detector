import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Load datasets
fake_df = pd.read_csv('data/Fake.csv')
true_df = pd.read_csv('data/True.csv')

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
y = data['label'].values

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert to tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# DataLoader
train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

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

# Initialize model
model = FakeNewsCNN(input_size=5000, output_size=2)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training
epochs = 10
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = model(X_batch)
        loss = criterion(output, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f'Epoch {epoch+1}/{epochs}, Loss: {total_loss:.4f}')

# Evaluation
model.eval()
preds = []
with torch.no_grad():
    for X_batch, _ in test_loader:
        output = model(X_batch)
        preds.extend(torch.argmax(output, dim=1).cpu().numpy())

test_acc = accuracy_score(y_test, preds)
print(f'Test Accuracy: {test_acc:.2f}')
print('Classification Report:\n', classification_report(y_test, preds))

import joblib

# Assuming 'model' is your trained CNN model
joblib.dump(model, 'cnn_model.pkl')
print("Model saved as cnn_model.pkl")

# Assuming 'vectorizer' is your trained vectorizer
joblib.dump(vectorizer, 'vectorizer.pkl')
print("Vectorizer saved as vectorizer.pkl")


# Load the model
model = joblib.load('cnn_model.pkl')

# Load the vectorizer
vectorizer = joblib.load('vectorizer.pkl')

print("Model and vectorizer loaded successfully!")