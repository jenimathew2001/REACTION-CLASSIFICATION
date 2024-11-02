import streamlit as st
import numpy as np
import joblib
import pandas as pd
from rdkit.Chem import AllChem, DataStructs, Draw
from rdkit import Chem
import dill
from io import BytesIO
import matplotlib.pyplot as plt
from PIL import Image
import os
import zipfile
import sys


def memory_usage():
    """Reports the current memory usage of the Python process."""
    mem = sys.getsizeof(globals())
    st.write(f"Current memory usage: {mem / (1024 ** 2):.2f} MB")


# LOAD MODELS
    
# Paths for the label encoder and model files
label_encoder_path = 'label_encoder.pkl'
zip_path = 'final_model_svc.pkl.zip'
model_filename = 'final_model_svc.pkl'
explainer_path = 'lime_explainer.dill'

@st.cache_resource
def load_label_encoder():
    # Load and cache the label encoder
    return joblib.load(label_encoder_path)

@st.cache_resource
def load_model():
    # Unzip only if the model file is not already extracted
    if not os.path.exists(model_filename):
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall()  # This will extract in the current directory
    
    # Load the extracted model file with joblib
    return joblib.load(model_filename)

@st.cache_resource
def load_explainer():
    # Load and cache the explainer
    with open(explainer_path, 'rb') as f:
        return dill.load(f)

# Load resources
label_encoder = load_label_encoder()
final_model = load_model()
explainer = load_explainer()

# After loading models and resources
st.write("After loading models and resources")
memory_usage()
    
# Convert SMILES to ECFP
def smiles_to_ecfp(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros((nBits,), dtype=int)
    ecfp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=radius, nBits=nBits)
    array = np.zeros((nBits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(ecfp, array)
    return array

@st.cache_data
def reaction_to_ecfps(reaction_smiles, radius=3, nBits=2048):
    reactants, products = reaction_smiles.split(">>")
    reactants_ecfp = smiles_to_ecfp(reactants, radius, nBits)
    products_ecfp = smiles_to_ecfp(products, radius, nBits)
    return np.concatenate([reactants_ecfp, products_ecfp])

def visualise_reaction(reaction_smiles):
    # Convert the SMILES string to an RDKit reaction object
    reaction = Chem.rdChemReactions.ReactionFromSmarts(reaction_smiles)

    # Draw the reaction and save as an image
    img = Draw.ReactionToImage(reaction)

    # Save the image to a BytesIO object
    img_buffer = BytesIO()
    img.save(img_buffer, format="PNG")
    img_buffer.seek(0)  # Go to the beginning of the BytesIO buffer

    return img_buffer


def smiles_from_bitinfo(mol, bit_info):
    bit_smiles = {}
    for bit, info in bit_info.items():
        for atom, rad in info:
            env = Chem.FindAtomEnvironmentOfRadiusN(mol, rad, atom)
            amap = {}
            submol=Chem.PathToSubmol(mol, env, atomMap=amap)
            smi = Chem.MolToSmiles(submol)
            bit_smiles[bit] = smi
    return bit_smiles


# Streamlit app setup
st.title("🔬 Reaction Class Predictor")

# Prompt user for input
st.subheader("Enter Reaction SMILES")
user_input = st.text_input("Example: `C1=CC=CC=C1>>C1=CC=C(C=C1)C(=O)O`", '')

if user_input:

    # Convert the input to ECFP
    reaction_ecfp = reaction_to_ecfps(user_input)

    # After reaction conversion
    st.write("After converting reaction to ECFP")
    memory_usage()
    
    # Model prediction
    prediction = final_model.predict([reaction_ecfp])
    predicted_class = label_encoder.inverse_transform(prediction)[0]
    predicted_proba = final_model.predict_proba([reaction_ecfp])[0]
    confidence = predicted_proba.max()

    # After model prediction
    st.write("After model prediction")
    memory_usage()

    # Display results in columns
    st.markdown("---")
    st.subheader("Prediction Results")
    col1, col2 = st.columns(2)
    col1.metric("Predicted Reaction Class", predicted_class)
    col2.metric("Confidence Level", f"{confidence:.2%}", delta_color="inverse")

    # Show confidence alert if low
    threshold = 0.7
    if confidence < threshold:
        # Sort probabilities in descending order and retrieve classes
        sorted_indices = np.argsort(predicted_proba)[::-1]
        
        # Find the first class different from the predicted class
        second_best_class_idx = next(
            (i for i in sorted_indices if i != prediction[0]), None
        )
        
        if second_best_class_idx is not None:
            second_best_label = label_encoder.inverse_transform([second_best_class_idx])[0]
            alt_class_prob = predicted_proba[second_best_class_idx]
            
            st.warning(
                f"The model is less confident in this prediction (confidence below {threshold * 100}%). "
                f"\nAlternative suggestion: **{second_best_label}** with probability: **{alt_class_prob:.2%}**"
            )
            
    # Visualize the reaction
    st.markdown("---")
    st.subheader("Reaction Visualization")
    img_buffer = visualise_reaction(user_input)
    st.image(img_buffer, caption='Reaction Structure', use_column_width=True)
    img_buffer.close()

    # After visualizing reaction
    st.write("After visualizing reaction")
    memory_usage()


    # GET REACTANT AND PRODUCT BIT INFO

    # Split the target reaction into reactants and products
    try:
        reactants, products = user_input.split(">>")
    except ValueError:
        st.error("Invalid input format. Please use the format: `Reactant>>Product`.")
        st.stop()
    
    # Convert the SMILES representation of the reactants and products into RDKit molecule objects
    r_mol = Chem.MolFromSmiles(reactants)
    p_mol = Chem.MolFromSmiles(products)

    # Check if the molecules were created successfully
    if r_mol is None:
        st.error(f"Could not parse reactant SMILES: `{reactants}`. Please check the format.")
        st.stop()
    if p_mol is None:
        st.error(f"Could not parse product SMILES: `{products}`. Please check the format.")
        st.stop()

    # Dictionaries to store bit information for reactants and products
    r_bi = {}
    p_bi = {}

    # Generate Morgan fingerprints for both reactants and products
    # Using radius=3 and nBits=2048
    fpr = AllChem.GetMorganFingerprintAsBitVect(r_mol, radius=3, bitInfo=r_bi, nBits=2048)
    fpp = AllChem.GetMorganFingerprintAsBitVect(p_mol, radius=3, bitInfo=p_bi, nBits=2048)

    updated_p_bi = {key + 2048: value for key, value in p_bi.items()}
    p_onbits = [x + 2048 for x in fpp.GetOnBits()]
    r_onbits = [x for x in fpr.GetOnBits()]

    reagent_smiles = smiles_from_bitinfo(r_mol, r_bi)
    product_smiles = smiles_from_bitinfo(p_mol, updated_p_bi)

    st.markdown("loading explainer")

    exp = explainer.explain_instance(np.asarray(reaction_ecfp), final_model.predict_proba, num_features=len(reaction_ecfp))
    st.markdown("explainer loaded")
    map = exp.as_map()[1]

    # Retrieve important bits and their corresponding scores from the map
    important_bits = [i[0] for i in map]
    importance_scores = [i[1] for i in map]

    sorted_map = sorted(map, key=lambda tup: tup[1], reverse=True)

    # After retrieving bit information
    st.write("After retrieving bit information")
    memory_usage()


    # Display bit-level importance with molecular structures
    st.markdown("---")
    st.subheader("Bit-Level Importance with Molecular Structures")

    # Sort and display in two columns for Reactants and Products
    reactant_bits = [(bit, importance) for bit, importance in sorted_map if bit in r_onbits]
    product_bits = [(bit, importance) for bit, importance in sorted_map if bit in p_onbits]

    def display_molecule(bit, importance, smiles, label):
        # Function to render molecule image with label and importance
        submol = Chem.MolFromSmiles(smiles)
        if submol:
            # Using HTML for custom styling to display bordered box without background color
            st.markdown(
            f'''
            <div style="
                border: 2px solid #ccc; 
                padding: 10px; 
                border-radius: 10px; 
                text-align: center;
                margin-bottom: 15px;
                transition: transform 0.2s;
            " 
            onmouseover="this.style.transform='scale(1.05)'" 
            onmouseout="this.style.transform='scale(1)'">
                <strong style="color: #333;">{label}</strong> - Importance: {importance:.3f} <br> Bit: {bit}
            </div>
            ''', unsafe_allow_html=True
            )
            st.image(Draw.MolToImage(submol, size=(120, 120)), caption=smiles, use_column_width=False)

    # Organize Reactants and Products side by side with improved column layout and scrollable sections
    col1, col2 = st.columns(2)

    # Display Reactants in the first column with scrolling enabled
    with col1:
        st.markdown("### Reactants")
        with st.container():  # Container to enable scrolling
            for bit, importance in reactant_bits:
                rsmiles = reagent_smiles.get(bit, None)
                if rsmiles:
                    display_molecule(bit, importance, rsmiles, "Reactant")
                    

    # Display Products in the second column with scrolling enabled
    with col2:
        st.markdown("### Products")
        with st.container():  # Container to enable scrolling
            for bit, importance in product_bits:
                psmiles = product_smiles.get(bit, None)
                if psmiles:
                    display_molecule(bit, importance, psmiles, "Product")

                
    # Clear variables to free memory
    del reactant_bits, product_bits, rsmiles, psmiles, reagent_smiles, product_smiles, map, sorted_map, exp
    
    # Divider and more info section
    st.markdown("---")
    with st.expander("🔍 Additional Information", expanded=False):
        st.write(
            """
            - **Model Description**: This model predicts reaction classes based on chemical reaction SMILES strings.
            - **Confidence Interpretation**: Higher confidence percentages suggest stronger predictions.
            - **Alternative Class Suggestion**: When confidence is below 70%, an alternative prediction may be shown.
            """
        )
