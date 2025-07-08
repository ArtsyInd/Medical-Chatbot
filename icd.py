import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
from tensorflow.keras.preprocessing.text import Tokenizer #type:ignore
from tensorflow.keras.preprocessing.sequence import pad_sequences #type:ignore
import pandas as pd
import pickle

# Load tokenizer
def load_tokenizer(filename):
    with open(filename, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

# Load ICD codes and billing information
def load_icd_codes(filename):
    dataset = pd.read_csv(filename)
    icd_codes = dataset['icd_code'].tolist()
    billing_info = dataset['total_cost'].tolist()
    code_dict = {code: billing for code, billing in zip(icd_codes, billing_info)}
    return code_dict

# Predict ICD code and billing
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

# Streamlit App
def main():
    st.title("Medical Chatbot")

    # Initialize session state variables
    if 'total_billing' not in st.session_state:
        st.session_state.total_billing = 0
    if 'round_counter' not in st.session_state:
        st.session_state.round_counter = 1
    if 'make_another_prediction' not in st.session_state:
        st.session_state.make_another_prediction = True
    if 'prediction_made' not in st.session_state:
        st.session_state.prediction_made = False

    round_counter = st.session_state.round_counter

    if st.session_state.make_another_prediction:
        user_input = st.text_area("Please describe your symptoms:", key=f"user_input_{round_counter}")
        user_severity = st.selectbox("How severe are your symptoms:", ["Mild", "Moderate", "Severe"], key=f"user_severity_{round_counter}")
        user_pre_existing_condition = st.text_area("Please describe your pre existing conditions:", key=f"user_pre_existing_condition_{round_counter}")
        user_temporal_info = st.text_input("Enter temporal information:", key=f"user_temporal_info_{round_counter}")
        user_past_treatments = st.text_input("Enter past treatments:", key=f"user_past_treatments_{round_counter}")
        user_surgeries = st.text_input("Enter surgeries:", key=f"user_surgeries_{round_counter}")
        user_medications = st.text_input("Enter medications:", key=f"user_medications_{round_counter}")
        user_treatments = st.text_input("Enter treatments administered:", key=f"user_treatments_{round_counter}")

        if st.button("Predict", key=f"predict_button_{round_counter}"):
            if user_input.strip() != "" and user_temporal_info.strip() != "":
                # Combine input fields into input text
                input_text = f"{user_temporal_info} {user_past_treatments} {user_surgeries} {user_medications} {user_treatments} {'Yes' if user_pre_existing_condition == 'Yes' else 'No'} {user_input} {user_severity}"

                # Predict ICD code and billing
                predicted_icd, billing = predict_icd(model, tokenizer, code_dict, input_text)

                if predicted_icd:
                    st.success(f"Predicted ICD code: {predicted_icd}")
                    st.success(f"Billing Information: ${billing}")
                    st.session_state.total_billing += billing
                    st.session_state.prediction_made = True
                else:
                    st.error("Sorry, I couldn't predict the ICD code for your symptoms.")
                    st.session_state.prediction_made = False
            else:
                st.warning("Please provide a description of your symptoms and temporal information.")
                st.session_state.prediction_made = False

        if st.session_state.prediction_made:
            if st.button("Make another prediction", key=f"continue_button_{round_counter}"):
                st.session_state.round_counter += 1
                st.session_state.prediction_made = False
                st.experimental_rerun()

    st.write(f"The total billing amount for all predictions is: ${st.session_state.total_billing}")

if __name__ == "__main__":
    main()
