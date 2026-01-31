# app.py
import streamlit as st
from PIL import Image
import numpy as np
from model_loader import load_model, run_inference
from helpers import cv2_to_pil

# Configuration de la page
st.set_page_config(
    page_title="Détection d'Objets - YOLOv5",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.title("🔍 Détection d'Objets avec YOLOv5")
st.markdown("""
Ce projet utilise **YOLOv5s** (via PyTorch Hub) pour détecter des objets dans les images.
Modèle pré-entraîné sur le jeu de données **COCO** (80 classes).
""")

# Charger le modèle une seule fois (cache)
@st.cache_resource
def get_yolo_model():
    return load_model()

try:
    model = get_yolo_model()
except Exception as e:
    st.error(f"Erreur lors du chargement du modèle : {e}")
    st.stop()

# Sidebar
st.sidebar.header("⚙️ Paramètres")
conf_threshold = st.sidebar.slider(
    "Seuil de confiance", 
    min_value=0.1, 
    max_value=1.0, 
    value=0.25, 
    step=0.05,
    help="Seuls les objets avec une confiance ≥ ce seuil seront affichés."
)

# Upload ou image exemple
uploaded_file = st.file_uploader(
    "📤 Téléchargez une image (JPG, PNG)", 
    type=["jpg", "jpeg", "png"]
)

# Chemin des images exemples
example_images = {
    "Rue animée (voitures, piétons)": "assets/example1.jpg",
    "Bureau (écran, chaise, personne)": "assets/example2.jpg"
}

if uploaded_file is not None:
    input_image = Image.open(uploaded_file)
    st.image(input_image, caption="Image téléchargée", use_column_width=True)
else:
    st.info("ℹ️ Aucune image téléchargée. Sélectionnez une image exemple ci-dessous.")
    example_choice = st.selectbox("Choisissez une image exemple :", list(example_images.keys()))
    try:
        input_image = Image.open(example_images[example_choice])
        st.image(input_image, caption=f"Image exemple : {example_choice}", use_column_width=True)
    except FileNotFoundError:
        st.error(f"❌ Fichier manquant : {example_images[example_choice]}")
        st.stop()

# Bouton de détection
if st.button("🚀 Lancer la détection"):
    with st.spinner("⏳ Détection en cours..."):
        try:
            annotated_img_bgr, detections = run_inference(model, input_image, conf_threshold)
            annotated_img_pil = cv2_to_pil(annotated_img_bgr)

            st.subheader("✅ Résultat")
            st.image(annotated_img_pil, caption="Objets détectés", use_column_width=True)

            # Afficher le nombre d'objets
            st.success(f"**{len(detections)}** objet(s) détecté(s).")

            # Détails optionnels
            if st.checkbox("📊 Voir les détections brutes"):
                class_names = model.names
                st.write("**Format** : `[x1, y1, x2, y2, confiance, classe_id]`")
                for i, det in enumerate(detections):
                    x1, y1, x2, y2, conf, cls_id = det
                    cls_name = class_names[int(cls_id)]
                    st.text(f"{i+1}. {cls_name} | Conf: {conf:.2f} | Boîte: ({x1:.1f}, {y1:.1f}) → ({x2:.1f}, {y2:.1f})")

        except Exception as e:
            st.error(f"❌ Erreur pendant l'inférence : {e}")

# Footer
st.markdown("---")
st.caption("Projet de recherche – Détection d'objets | YOLOv5 + Streamlit")