import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import pandas as pd
import re
from nltk.stem import WordNetLemmatizer
import nltk
import json 
import numpy as np
import streamlit  as st
from fastapi import FastAPI, HTTPException

app = FastAPI()

nltk.download('wordnet')
# Load your TensorFlow model
model = tf.keras.models.load_model("my_model.h5")
lemmatizer=WordNetLemmatizer()
maxlen = 41
with open('tokenizer.json', 'r', encoding='utf-8') as f:
    tokenizer = tokenizer_from_json(json.load(f))

def preprocessing(text):
    # Ensure the input is a string, otherwise return an empty string
    if not isinstance(text, str):
        return ''

    cleaned_text = re.sub(r'(http|https|www)\S+', '', text)  # Remove URLs
    cleaned_text = re.sub(r'[@#]\w+', '', cleaned_text) # Remove mentions (like @username) and hashtgs

    cleaned_text = re.sub(r'[^a-zA-Z\s]', '', text)
    cleaned_text = cleaned_text.replace('\n', ' ')
    cleaned_text = re.sub(r'\s+', ' ', cleaned_text)

    cleaned_text = cleaned_text.split()
    filtered_words = [lemmatizer.lemmatize(word, pos='v') for word in cleaned_text]
    text = ' '.join(filtered_words)
    return text

def getPrediction(input):
    input = pd.DataFrame(input, columns=['text'])
    input['text'] = input['text'].apply(preprocessing)
    print(input['text'][0], end=", ")
    input = tokenizer.texts_to_sequences(input['text'])
    input = pad_sequences(input, maxlen = maxlen,  padding = 'post', truncating = 'post')
    prediction = model.predict(input, verbose=0)
    # calculate confidence score
    confidence_score = np.max(prediction, axis=1)/np.sum(prediction, axis=1)
    result = np.argmax(prediction, axis=1)
    for i in range(len(confidence_score)):
        if confidence_score[i] < 0.7:
            result[i] = 2
    print(prediction, confidence_score)
    return result, confidence_score

def getSentiment(idx):
   return {0: "Negative", 1: "Positive", 2: "Neutral"}.get(idx, "Neutral")


@app.post("/predict")
async def predict(text: str):
    prediction, confidence_score = getPrediction(text)
    return {
        "prediction": getSentiment(prediction) + " Statement",
        "confidence": f"{confidence_score * 100:.2f}%"
    }
       
# Streamlit UI
st.title("Sentiment Analysis")
text = st.text_area("Enter Text...")

if text:
    prediction, confidence_score = getPrediction([text])  # Modify if preprocessing is needed
    # Convert prediction to a human-readable format
    response = {"prediction": getSentiment(prediction[0]) + " Statement",
                "confidence": "{:.2f}".format(float(confidence_score[0] * 100)) + "%"}  # Adjust as necessary for output formatting
    st.json(response)


