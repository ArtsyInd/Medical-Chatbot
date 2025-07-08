import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from tensorflow.keras.preprocessing.text import Tokenizer # type: ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences # type: ignore
import pandas as pd
import pickle

def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def load_icd_codes(filename):
    dataset = pd.read_csv(filename)
    icd_codes = dataset['icd_code'].tolist()
    billing_info = dataset['total_cost'].tolist()
    code_dict = {code: billing for code, billing in zip(icd_codes, billing_info)}
    return code_dict

def predict_icd(model, tokenizer, code_dict, input_text):
    # Tokenize and pad the input sequence
    input_sequence = tokenizer.texts_to_sequences([input_text])
    padded_sequence = pad_sequences(input_sequence, maxlen=max_sequence_length)

    # Predict using the LSTM model
    predicted_probabilities = model.predict(padded_sequence)[0]
    predicted_index = np.argmax(predicted_probabilities)
    
    if predicted_index < len(code_dict):
        predicted_icd = list(code_dict.keys())[predicted_index]
        billing = float(code_dict[predicted_icd].replace('$', ''))  
        return predicted_icd, billing
    else:
        return None, None

# Load the saved LSTM model
model = load_model("my_model.keras")

# Load the tokenizer
tokenizer = load_tokenizer("tokenizer.pkl")

# Load ICD codes and billing information
code_dict = load_icd_codes("patient_data.csv")

max_sequence_length = 100

# Example usage

def chatbot():
    print("Welcome to the Medical Chatbot!")
    total_billing = 0
    while True:
        # Input fields
        user_input = input("Please describe your symptoms: ")
        user_severity = input("How severe are your symptoms (mild, moderate, severe): ")
        user_pre_existing_condition = input("Do you have any pre-existing conditions: ")
        user_temporal_info = input("Enter temporal information: ")
        user_past_treatments = input("Enter past treatments: ")
        user_surgeries = input("Enter surgeries: ")
        user_medications = input("Enter medications: ")
        user_treatments = input("Enter treatments administered: ")

        # Combine input fields into input text
        input_text = f"{user_temporal_info} {user_past_treatments} {user_surgeries} {user_medications} {user_treatments} {user_pre_existing_condition} {user_input} {user_severity}"
        
        # Predict ICD code and billing
        predicted_icd, billing = predict_icd(model, tokenizer, code_dict, input_text)
        
        if predicted_icd:
            print("Predicted ICD code:", predicted_icd)
            print("Billing Information: $", billing)
            # Accumulate billing
            total_billing += billing
        else:
            print("Sorry, I couldn't predict the ICD code for your symptoms.")
        
        choice = input("Do you want to add more symptoms? (yes/no): ")
        if choice.lower() != 'yes':
            break

    # Print total billing
    print("Total Billing: $", total_billing)

# Start the chatbot
chatbot()