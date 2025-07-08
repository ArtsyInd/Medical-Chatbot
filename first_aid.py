import streamlit as st
import json
import nltk
import numpy as np
import random
import tensorflow as tf
from nltk.stem import LancasterStemmer

stemmer = LancasterStemmer()

# Load the trained model
model = tf.keras.models.load_model('EngChatbotModel.h5')

# Load the intents data
with open('EnglishData.json', 'r') as file:
    data = json.load(file)

# Extract words and labels
words = []
labels = []

for intent in data['intents']:
    for pattern in intent['patterns']:
        wrds = nltk.word_tokenize(pattern)
        words.extend(wrds)
    if intent['tag'] not in labels:
        labels.append(intent['tag'])

# Word Stemming
words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))
labels = sorted(labels)

# Function to convert input to a bag of words
def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return [bag]

# Function to get a response from the chatbot
def get_response(inp):
    results = model.predict(np.array(bag_of_words(inp, words)))
    results_index = np.argmax(results)
    tag = labels[results_index]

    for intent in data['intents']:
        if intent['tag'] == tag:
            responses = intent['responses']

    return random.choice(responses)

# Streamlit App
def main():
    st.title("First Aid Chatbot")
    st.write("This is a first aid chatbot that allows user to search for the recommended first aid steps in a medical emergency")

    st.write("Start talking with the bot (type 'thank you' to stop)!")
    inp = st.text_input("You:")
    if inp.lower() == "thank you":
        return

    if inp:
        response = get_response(inp)
        st.write("Bot:", response)
if __name__ == "__main__":
    main()
