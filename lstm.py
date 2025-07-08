import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, GlobalMaxPooling1D,LSTM
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import BinaryAccuracy, Precision, Recall, AUC
from gensim.models import Word2Vec
import pandas as pd
import pickle


def save_tokenizer(tokenizer, filename):
    with open(filename, 'wb') as f:
        pickle.dump(tokenizer, f, protocol=pickle.HIGHEST_PROTOCOL)

# Load your dataset here
dataset = pd.read_csv("patient_data.csv")

# Extract relevant columns from the dataset and rename them
temporal_info = dataset['temporal_info'].tolist()
past_treatments = dataset['past_treatments'].tolist()
surgeries = dataset['surgeries'].tolist()
medications = dataset['medications_prescribed'].tolist()
treatments = dataset['treatments_administered'].tolist()
pre_existing_conditions = dataset['pre_existing_conditions'].tolist()
symptoms = dataset['symptoms'].tolist()
icd_codes = dataset['icd_code'].tolist()
descriptions = dataset['severity'].tolist()
billing_info = dataset['total_cost'].tolist()


# Combine all text features
combined_features = [f"{temp} {past} {surgery} {med} {treat} {pre_existing} {symptom} {description} {billing}" 
                     for temp, past, surgery, med, treat, pre_existing, symptom, description, billing
                     in zip(temporal_info, past_treatments, surgeries, medications, treatments, pre_existing_conditions, symptoms, descriptions, billing_info)]

# Train Word2Vec model on combined features
word2vec_model = Word2Vec(sentences=[feature.split() for feature in combined_features], vector_size=100, window=5, min_count=1, workers=4)

# Tokenize the combined features
tokenizer = Tokenizer()
tokenizer.fit_on_texts(combined_features)
X = tokenizer.texts_to_sequences(combined_features)

# Pad sequences to ensure uniform length
max_sequence_length = 100  # Adjust as needed based on your data
X = pad_sequences(X, maxlen=max_sequence_length)

# Convert ICD codes to one-hot encoding
code_dict = {code: i for i, code in enumerate(set(icd_codes))}
y = np.zeros((len(icd_codes), len(code_dict)))
for i, code in enumerate(icd_codes):
    y[i, code_dict[code]] = 1

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_classes = len(code_dict)

# Define the LSTM model
model = Sequential()
model.add(Embedding(len(tokenizer.word_index) + 1, 100))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
model.add(LSTM(100, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(num_classes, activation='softmax'))
# Compile the model
model.compile(loss=BinaryCrossentropy(), optimizer='adam', metrics=[BinaryAccuracy(), Precision(), Recall(), AUC()])

# Train the model
model.fit(X_train, y_train, epochs=15, batch_size=32, validation_data=(X_test, y_test))

# Save the LSTM model
model.save('my_model.keras')

save_tokenizer(tokenizer, 'tokenizer.pkl')

