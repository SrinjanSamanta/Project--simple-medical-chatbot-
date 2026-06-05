import random
import json
import pickle
import nltk

nltk.download('punkt')
nltk.download('wordnet')

# Load trained model
model, vectorizer, labels = pickle.load(open("model.pkl", "rb"))

# Load intents
with open('intents.json') as file:
    data = json.load(file)

# Predict intent using TF-IDF
def predict_class(sentence):
    X = vectorizer.transform([sentence.lower()])
    probs = model.predict_proba(X)[0]

    max_prob = max(probs)
    index = probs.argmax()

    # DEBUG
    print(f"DEBUG: max_prob = {max_prob}, predicted tag = {labels[index]}")

    if max_prob < 0.05:
        return "unknown"

    return labels[index]
# Get response
def get_response(tag):
    if tag == "unknown":
        return "I'm not sure I understand. Can you describe your symptoms more clearly?"

    for intent in data['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])

# Chat loop
print("Bot is running! (type 'exit' to quit)")

while True:
    msg = input("You: ")

    if msg.lower() == "exit":
        break

    tag = predict_class(msg)
    response = get_response(tag)

    print("Bot:", response)