# Medical Chatbot System

## Overview

**Medical Chatbot System | Python, TensorFlow, scikit-learn, NLP | 2024**

This repository implements an **AI-driven medical chatbot system with three integrated modules** for improving healthcare accessibility and preliminary support:

- **ICD Recommender (LSTM)**: Predicts ICD codes from user-described symptoms using deep learning (TensorFlow).
- **Healthcare Chatbot (Decision Tree)**: Provides preliminary diagnoses based on user symptom inputs.
- **First Aid Bot (Pre-trained Models)**: Guides users with immediate first aid steps during emergencies.

## Repository Structure

- `main.py`: Entry point for the **Streamlit multi-page app**, allowing users to navigate between the three chatbot modules seamlessly.
- `icd.py`, `lstm.py`: Implements the **LSTM-based ICD code recommender**.
- `healthcare.py`: Contains the **healthcare chatbot** using a decision tree for symptom-based diagnostics.
- `first_aid.py`: Contains the **first aid guidance bot**.
- `EngChatbotModel.h5`: Pre-trained model for language handling in chatbot interactions.
- `health_image.jpeg`: Visual asset for chatbot interface.
- `templates/`, `static/`, `img/`: Front-end assets for UI rendering in Streamlit.
- `README.md`: Documentation for this project.

## Notes on Data

Please note that **the datasets used for model training and evaluation have not been added to this repository** due to size and privacy considerations. To replicate the workflow, you will need structured symptom and ICD datasets aligned with the system’s CSV-based ingestion pipeline.

## Features

- Provides **ICD code prediction** using LSTM-based neural networks.
- Offers **preliminary medical advice** based on user symptoms.
- Delivers **first aid guidance** for emergency situations.
- Modular design, easily extendable with additional medical functionalities.
- Uses **TensorFlow, scikit-learn, and NLP methods** for robust medical text interpretation.
- **Streamlit-based GUI** for user-friendly interaction.

## Getting Started

1. Clone the repository:
    ```bash
    git clone <repo_url>
    cd <repo_name>
    ```

2. Install required packages:
    ```bash
    pip install -r requirements.txt
    ```

3. Run the unified Streamlit app:
    ```bash
    streamlit run main.py
    ```

    Navigate through the ICD Recommender, Healthcare Chatbot, and First Aid modules via the sidebar.

## License

This project is intended for **academic and research learning purposes** in healthcare AI development.

---

For any queries regarding usage or contributions, please feel free to raise an issue or discussion in this repository.

