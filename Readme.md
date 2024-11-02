Here's a well-structured `README.md` for your GitHub repository. This document includes instructions for local setup, running the application, and additional information regarding training and deployment:

```markdown
# Reaction Class Predictor

## Overview
The **Reaction Class Predictor** is a web application built using Streamlit that allows users to input chemical reaction SMILES and predicts the reaction class using a trained machine learning model. The application also visualizes the molecular structures and highlights the most important features influencing the prediction.

## Live Demo
You can check out a live demo of the application here: [Reaction Classifier Live Demo](https://reaction-classifier-1dbabbc88010.herokuapp.com/)

## Prerequisites
Before running the application locally, ensure you have the following:
- Python 3.x installed on your machine.
- Basic understanding of command-line interface usage.

## How to View the Application Locally

1. **Clone the Repository**
   ```bash
   git clone https://github.com/yourusername/repository-name.git
   cd repository-name
   ```

2. **Set Up the Required Files**
   Ensure that the following model files are present in the `Reaction` folder:
   - `final_model_svc.pkl.zip`
   - `label_encoder.pkl`
   - `lime_explainer.dill`
   - `app.py`
   - `requirements.txt`

## How to Run the App

1. **Create a Virtual Environment**
   ```bash
   python3 -m venv env
   ```

2. **Activate the Virtual Environment**
   - On macOS/Linux:
     ```bash
     source env/bin/activate
     ```
   - On Windows:
     ```bash
     .\env\Scripts\activate
     ```

3. **Install Required Packages**
   If `requirements.txt` is not already populated, you can generate it using:
   ```bash
   pip freeze > requirements.txt
   ```
   Then, install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the Streamlit Application**
   ```bash
   streamlit run app.py
   ```

5. **Access the Application**
   After running the command, you should see a URL in the terminal. Open it in your web browser to view the application.

## Exploring the Training Code
To understand how the models were created, check out the training code in this repository. It includes detailed explanations and methodologies used for model training and evaluation.

## Important Notes on Deployment
- Due to memory limitations during deployment, the number of samples used for the explainer was reduced. This means that while the application runs smoothly in a production environment, users may experiment with increasing the sample size when running the application locally.
