import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import tensorflow as tf
import matplotlib.pyplot as plt

# Charger le modèle
@st.cache_resource # Cache le chargement du modèle pour améliorer les performances
def load_model():
    return tf.keras.models.load_model("pneumonie1.h5")

model = load_model() 

st.title("DETECTION DE LA PNEUMONIE")

#picture = st.camera_input("Take a picture", disabled=False)
#if picture: 
    #st.image(picture)

fonctionnalities = ["prediction", "visualisation"]
choice = st.sidebar.selectbox("Menu", fonctionnalities)

if choice == "prediction":
    st.subheader("Faire une prédiction")
    data = st.file_uploader("Charger l'image", type=["jpg", "png"])
    
    if data is not None:
        image = Image.open(data)
        #st.image(image, caption='Image Chargée', use_column_width=True)
    predict = st.button(label='Predict')

    if predict and data is not None:
        def preprocess_image(image, target_size):
            image = image.convert("L")  # Convertir en niveaux de gris
            image = image.resize(target_size)  # Redimensionnement
            image = np.array(image)  # Conversion en tableau NumPy
            image = image / 255.0  # Normalisation
            image = np.expand_dims(image, axis=-1)  # Ajouter une dimension pour le canal (grayscale)
            image = np.expand_dims(image, axis=0)  # Ajouter une dimension batch
            return image
        
        processed_image = preprocess_image(image, target_size=(256,256))
        # Vérifiez la forme avant prédiction
        st.write(f"Shape of processed image: {processed_image.shape}")

        class_names = ["Normal", "Pneumonia"]
        # Faire la prédiction
        def predict(model, image):
            predictions = model.predict(image)
            predicted_class = class_names[np.argmax(predictions[0])]
            confidence = round(100 * (np.max(predictions[0])), 2)
            return predicted_class, confidence

         # Afficher le résultat
        predicted_class, confidence = predict(model, processed_image)
        st.write(f"**Résultat de la prédiction :** {predicted_class}")
        st.write(f"**Score de confiance :** {confidence}")

        # Visualiser la prédiction

        fig, ax = plt.subplots()

        #ax.imshow(np.squeeze(image[0]))  # Supprimer la dimension batch pour l'affichage
        #ax.imshow(np.array(image))
        ax.imshow(processed_image[0])
        ax.set_title(f"Prédiction : {predicted_class} ({confidence}%)")

        st.pyplot(fig)


elif choice == "visualisation":
    st.subheader("Statistiques générales")
