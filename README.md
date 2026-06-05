# Project: Simple Medical Chatbot

A Python-based conversational AI agent designed to respond to user queries regarding medical and healthcare intents. This project utilizes Natural Language Processing (NLP) and machine learning to classify user inputs and deliver accurate, context-aware responses.

## Features
* *Intent Classification:* Uses a neural network trained on structured intents (`intents.json`).
* *Dynamic Responses:* Matches user queries with predefined tags to generate human-like text responses.
* *Efficient Storage:* Models and training parameters are securely serialized using Pickle (`model.pkl`) for fast execution.

## Structure
* `train.py` - Script responsible for processing intents data and training the deep learning model.
* `chat.py` - The execution script used to run and interact with the chatbot in real-time.
* `intents.json` - The dataset containing patterns, tags, and corresponding medical responses.

## How to Run
1. Ensure you have Python installed.
2. Train the model first:
```bash
   python train.py
