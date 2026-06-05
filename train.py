import json
import pickle

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Prepare data
sentences = []
labels_list = []
for intent in data['intents']:
    for pattern in intent['patterns']:
        sentences.append(pattern.lower())
        labels_list.append(intent['tag'])

# Convert labels to numeric
labels = sorted(set(labels_list))
y = [labels.index(tag) for tag in labels_list]

# TF-IDF vectorization
vectorizer = TfidfVectorizer(
    ngram_range=(1,2)
)
X = vectorizer.fit_transform(sentences)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X, y)
# Save everything
pickle.dump((model, vectorizer, labels), open("model.pkl", "wb"))

print("Model trained successfully!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# Predict on training data
y_pred = model.predict(X)

# Accuracy
print(f"Accuracy:  {accuracy_score(y, y_pred) * 100:.2f}%")

# Precision & Recall
print(f"Precision: {precision_score(y, y_pred, average='weighted') * 100:.2f}%")
print(f"Recall:    {recall_score(y, y_pred, average='weighted') * 100:.2f}%")
