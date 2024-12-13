import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
 
# Charger le modèle
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("converted_model.h5")
 
model = load_model()
 
class_train = ["Normal", "Pneumonia"]  
 
# Fonction de prédiction
def predict(model, img):
    # Vérifier et convertir en 3 canaux si nécessaire
    if img.mode != "L":
        img = img.convert("L")
 
    # Préparation de l'image
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (150, 150))  # Redimensionner selon le modèle
    img_array = img_array / 255.0  # Normalisation (important si le modèle s'attend à des valeurs entre 0 et 1)
    img_array = tf.expand_dims(img_array, 0)  # Ajouter une dimension batch (forme: (1, 150, 150, 1))
 
    # Vérification de la forme
    st.write("Forme de l'image entrée dans le modèle :", img_array.shape)
 
    # Prédiction
    predictions = model.predict(img_array)
    predicted_class = class_train[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
 
    # Vérification de la confiance
    if confidence > 50:
        predicted_class = "Normal"
    else:
        predicted_class = "Pneumonia"
 
    return predicted_class, confidence
 
# Interface Streamlit
st.title("Détection de pneumonie")
col1, col2 = st.columns([3, 1])
with col1:
    uploaded_file = st.file_uploader("Charger une image de la radio", type=["jpg", "png"])

with col2:
    st.markdown(
        """
        <style>
        .stButton>button {
            margin-top: 1.6em;
            height: 3em;
            background-color: #4CAF50;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    prediction = st.button("Prédire",use_container_width=True)

 
if uploaded_file and prediction:
    image = Image.open(uploaded_file)
    # Prédire sur l'image chargée
    with st.spinner("Prédiction en cours..."):
        predicted_class, confidence = predict(model, image)
    
    if predicted_class == "Pneumonia":
        st.markdown(
            """
            <h3 style="background-color: rgba(255,0,0,0.3); text-align: center">
            Cette personne est atteinte de pneumonie.</h3>
            """,
            unsafe_allow_html=True
        )        
        st.image(image, use_container_width=True)
        st.success(f"Classe prédite : {predicted_class}")
        st.write(f"Confiance : {confidence}%")
    else:
        st.markdown(
            """
            <h3 style="background-color: rgba(0,255,0,0.3);text-align: center">
            Cette personne n'est pas atteinte de pneumonie.</h3>
            """,
            unsafe_allow_html=True
        )
        st.image(image, use_container_width=True)
        st.success(f"Classe prédite : {predicted_class}")
        st.write(f"Confiance : {confidence}%")

    
 
    # Option pour afficher la prédiction sur l'image
    #if st.checkbox("Afficher la prédiction sur l'image"):
        #plt.figure(figsize=(4, 4))
        #plt.imshow(image)
        #plt.title(f"{predicted_class} ({confidence}%)")
        #plt.axis("off")
        #st.pyplot(plt)
 
 






