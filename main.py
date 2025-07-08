import streamlit as st
from icd import main as icd_main
from healthcare import main as healthcare_main
from first_aid import main as first_aid_main

def main():
    st.title("Medical Assistance Home Page")
    st.write("Welcome to the Medical Assistance App!")
    st.write("Welcome to Medical Aide, your trusted companion in healthcare excellence. With our suite of innovative tools, we empower users with personalized solutions for diagnosis, care, and emergency response. Seamlessly navigate the complexities of medical coding with our ICD code recommender, ensuring accuracy and efficiency in documentation. Explore our healthcare recommender for tailored care recommendations, enhancing patient outcomes and satisfaction. In times of urgency, rely on our first aid recommender for immediate, life-saving guidance. Join us on the journey to better health and well-being. Experience the future of healthcare with Medical Aide today.")

    # Sidebar navigation
    selected_module = st.sidebar.selectbox("Select Module", ["ICD Codes", "First Aid","Healthcare"])

    # Link to different modules
    if selected_module == "ICD Codes":
        st.write("You selected the ICD Codes module.")
        icd_main()  # Call the main function of icd.py
    elif selected_module == "Healthcare":
        st.write("You selected the Healthcare module.")
        healthcare_main()  # Call the main function of healthcare.py
    elif selected_module == "First Aid":
        st.write("You selected the First Aid module.")
        first_aid_main()  # Call the main function of first_aid.py

if __name__ == "__main__":
    main()
