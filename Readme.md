# Reaction Class Predictor

🔬 **Reaction Class Predictor** is a web application built using Streamlit that predicts the class of a chemical reaction based on its SMILES representation. Utilizing machine learning techniques and molecular fingerprints, this application provides a user-friendly interface for chemists and researchers to quickly analyze chemical reactions.

## Live Demo

You can access the live application here: [Reaction Class Predictor on Heroku](https://reaction-classifier-1dbabbc88010.herokuapp.com/)

## Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [How It Works](#how-it-works)
- [Exploring the Training Code](#exploring-the-training-code)
- [Important Notes on Deployment](#important-notes-on-deployment)
- [Contributing](#contributing)
- [License](#license)

## Features

- **User Input for SMILES**: Enter chemical reaction SMILES strings to predict their classes.
- **Reaction Visualization**: Automatically generates visual representations of the input reactions.
- **Model Predictions**: Provides predicted reaction classes along with confidence levels.
- **LIME Explainer**: Displays bit-level importance of molecular features influencing the predictions.
- **Alternative Suggestions**: If the confidence level is low, alternative reaction classes are suggested.

## Technologies Used

- [Streamlit](https://streamlit.io/) - Framework for building web apps
- [RDKit](https://www.rdkit.org/) - Collection of cheminformatics and machine learning tools
- [scikit-learn](https://scikit-learn.org/stable/) - Machine learning library for Python
- [LIME](https://github.com/marcotcr/lime) - Local Interpretable Model-agnostic Explanations
- [joblib](https://joblib.readthedocs.io/en/latest/) - Lightweight pipelining in Python
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [NumPy](https://numpy.org/) - Numerical computing in Python

## Installation

To run this project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/reaction-class-predictor.git
   cd reaction-class-predictor
   ```
   If you do not have Git installed, you can download the project as a ZIP file from GitHub [here](https://github.com/yourusername/reaction-class-predictor) and extract it to access the folder.

2. Open your terminal and navigate to the project directory (if not already there).

3. Create a new virtual environment:
   ```bash
   python -m venv venv
   ```

4. Activate the virtual environment:
   - On Windows:
     ```bash
     venv\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source venv/bin/activate
     ```

5. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

6. Ensure you have the following files in your project directory:
   - `label_encoder.pkl`
   - `final_model_svc.pkl.zip`
   - `lime_explainer.dill`

7. Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Usage

1. Open your browser and navigate to `http://localhost:8501` to view the app.
2. Enter a reaction in SMILES format in the input box (e.g., `C1=CC=CC=C1>>C1=CC=C(C=C1)C(=O)O`).
3. Click Enter to see the predicted reaction class and visualizations.
4. Review the bit-level importance and confidence levels in the prediction.

## How It Works

The application performs the following steps:

1. **Input Handling**: Accepts user input in the form of SMILES strings.
2. **ECFP Conversion**: Converts the SMILES representation to Extended Connectivity Fingerprints (ECFP) for model input.
3. **Model Prediction**: Uses a pre-trained machine learning model to predict the reaction class and confidence level.
4. **Reaction Visualization**: Generates a visual representation of the chemical reaction using RDKit.
5. **LIME Explanation**: Utilizes LIME to explain the model's predictions by identifying important molecular features.

## Exploring the Training Code

To understand how the models were created, check out the training code in this repository. It includes explanations and methodologies used for model training and evaluation.

## Important Notes on Deployment

- The application has been successfully deployed on **Heroku**. Users can access the live app online for predictions.
- Due to memory limitations during deployment, the number of samples used for the LIME explainer was reduced. This means that while the application runs smoothly in a production environment, users may experiment with increasing the sample size when running the application locally to enhance explanation quality.

```
