import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
 
# Charger le modèle (ajustez le chemin selon vos fichiers)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("converted_model.h5")  # Remplacez par le chemin vers votre modèle sauvegardé
 
model = load_model()
 
# Classes (ajustez selon votre jeu de données)
class_train = ["Normal", "Pneumonia"]  # Remplacez avec les noms des classes
 
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
st.title("Interface de Prédiction d'Images")
st.write("Chargez une image pour obtenir la prédiction du modèle.")
 
uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "png", "jpeg"])
 
if uploaded_file is not None:
    # Charger et afficher l'image
    image = Image.open(uploaded_file)
    st.image(image, caption="Image chargée.", use_column_width=True)
 
    # Prédire sur l'image chargée
    with st.spinner("Prédiction en cours..."):
        predicted_class, confidence = predict(model, image)
 
    st.success(f"Classe prédite : {predicted_class}")
    st.write(f"Confiance : {confidence}%")
 
    # Option pour afficher la prédiction sur l'image
    if st.checkbox("Afficher la prédiction sur l'image"):
        plt.figure(figsize=(4, 4))
        plt.imshow(image)
        plt.title(f"{predicted_class} ({confidence}%)")
        plt.axis("off")
        st.pyplot(plt)