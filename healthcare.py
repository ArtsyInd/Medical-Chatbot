import streamlit as st
import re
import pandas as pd
from sklearn import preprocessing
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load datasets
@st.cache_data
def load_data():
    training = pd.read_csv('Training.csv')
    testing = pd.read_csv('Testing.csv')
    severity = pd.read_csv('symptom_severity.csv')
    description = pd.read_csv('symptom_Description.csv')
    precaution = pd.read_csv('symptom_precaution.csv')

    return training, testing, severity, description, precaution

# Preprocess datasets
def preprocess_data(training, testing):
    cols = training.columns[:-1]
    x = training[cols]
    y = training['prognosis']
    le = preprocessing.LabelEncoder()
    le.fit(y)
    y = le.transform(y)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)
    testx = testing[cols]
    testy = testing['prognosis']
    testy = le.transform(testy)
    return x_train, x_test, y_train, y_test, testx, testy, cols, le

# Train models
def train_models(x_train, y_train):
    clf = DecisionTreeClassifier().fit(x_train, y_train)
    scores = cross_val_score(clf, x_train, y_train, cv=3)
    svm_model = SVC().fit(x_train, y_train)
    return clf, svm_model, scores

# Load dictionaries
def load_dictionaries(severity, description, precaution):
    severity_dict = dict(zip(severity.iloc[:, 0], severity.iloc[:, 1]))
    description_dict = dict(zip(description.iloc[:, 0], description.iloc[:, 1]))
    precaution_dict = {row[0]: row[1:].tolist() for _, row in precaution.iterrows()}
    return severity_dict, description_dict, precaution_dict

# Chatbot logic
def check_pattern(dis_list, inp):
    pred_list = []
    inp = inp.replace(' ', '_')
    patt = f"{inp}"
    regexp = re.compile(patt)
    pred_list = [item for item in dis_list if regexp.search(item)]
    if len(pred_list) > 0:
        return 1, pred_list
    else:
        return 0, []

def calc_condition(exp, days, severity_dict):
    total_severity = sum(severity_dict.get(item, 0) for item in exp)
    condition = (total_severity * days) / (len(exp) + 1)
    return condition > 13

def sec_predict(symptoms_exp, cols, le, x_train, y_train):
    input_vector = np.zeros(len(cols))
    for item in symptoms_exp:
        if item in cols:
            input_vector[cols.get_loc(item)] = 1
    rf_clf = DecisionTreeClassifier().fit(x_train, y_train)
    prediction = rf_clf.predict([input_vector])
    return le.inverse_transform(prediction)

# Streamlit Interface
def main():
    st.title('Healthcare Chatbot')

    # Initialize session state variables
    if 'chat_history' not in st.session_state:
        st.session_state['chat_history'] = []
    if 'symptoms_exp' not in st.session_state:
        st.session_state['symptoms_exp'] = []

    # Load and preprocess data
    training, testing, severity, description, precaution = load_data()
    x_train, x_test, y_train, y_test, testx, testy, cols, le = preprocess_data(training, testing)
    clf, svm_model, scores = train_models(x_train, y_train)
    severity_dict, description_dict, precaution_dict = load_dictionaries(severity, description, precaution)

    def display_chat():
        for chat in st.session_state.chat_history:
            st.write(chat)

    def add_to_chat(message):
        st.session_state.chat_history.append(message)

    def get_symptom_input():
        user_symptom = st.text_input("Enter the symptom you are experiencing:")
        if user_symptom:
            add_to_chat(f"User: {user_symptom}")
            return user_symptom
        return None

    def confirm_symptom(symptom_list):
        selected_symptom = st.selectbox("Select the one you meant:", symptom_list)
        if selected_symptom:
            add_to_chat(f"User selected: {selected_symptom}")
            return selected_symptom
        return None

    def get_days_input():
        num_days = st.number_input("Okay. For how many days?", min_value=1, step=1)
        if num_days:
            add_to_chat(f"User has been experiencing symptoms for {num_days} days.")
            return num_days
        return None

    def ask_symptoms(symptoms_given):
        user_responses = []
        for symptom in symptoms_given:
            response = st.radio(f"Are you experiencing {symptom}?", ('yes', 'no'), key=symptom)
            if response == 'yes':
                user_responses.append(symptom)
        return user_responses

    # Conversation Logic
    display_chat()
    add_to_chat("Bot: Hello! I am your healthcare assistant. Let's start by identifying your symptoms.")
    symptom_input = get_symptom_input()

    if symptom_input:
        conf, cnf_dis = check_pattern(cols, symptom_input)
        if conf == 1:
            selected_symptom = confirm_symptom(cnf_dis)
            if selected_symptom:
                num_days = get_days_input()
                if num_days:
                    add_to_chat(f"Bot: Let me analyze your symptoms...")

                    # Decision Tree recursion logic
                    def recurse(node, depth):
                        if clf.tree_.feature[node] != _tree.TREE_UNDEFINED:
                            name = cols[clf.tree_.feature[node]]
                            threshold = clf.tree_.threshold[node]
                            val = 1 if name == selected_symptom else 0
                            if val <= threshold:
                                recurse(clf.tree_.children_left[node], depth + 1)
                            else:
                                symptoms_present.append(name)
                                recurse(clf.tree_.children_right[node], depth + 1)
                        else:
                            present_disease = le.inverse_transform(clf.tree_.value[node][0].nonzero()[0])
                            symptoms_given = cols[training.groupby('prognosis').max().loc[present_disease].values[0].nonzero()]
                            symptoms_exp = ask_symptoms(symptoms_given)
                            second_prediction = sec_predict(symptoms_exp, cols, le, x_train, y_train)
                            condition = calc_condition(symptoms_exp, num_days, severity_dict)
                            if condition:
                                add_to_chat("Bot: You should take consultation from a doctor.")
                            else:
                                add_to_chat("Bot: It might not be that bad, but you should take precautions.")
                            add_to_chat(f"Bot: You may have {present_disease[0]}")
                            add_to_chat(f"Bot: {description_dict[present_disease[0]]}")
                            precaution_list = precaution_dict[present_disease[0]]
                            add_to_chat("Bot: Take the following measures:")
                            for i, precaution in enumerate(precaution_list, 1):
                                add_to_chat(f"{i}) {precaution}")
                            add_to_chat("Bot: Please note that this consultation is meant to help you reduce the field of possibilities. It is not a substitute for professional medical advice. Please consult a doctor for a proper diagnosis.")
                            return

                    symptoms_present = []
                    recurse(0, 1)

if __name__ == "__main__":
    main()
