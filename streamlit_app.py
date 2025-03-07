# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from main import predict_emotion, draw_face_landmarks, face_mesh  # Importer face_mesh depuis main.py
import logging

# Configurer les logs
logging.basicConfig(level=logging.DEBUG)

# Lire le fichier CSS
def load_css(file_name):
    with open(file_name, "r") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Charger le CSS
load_css("styles.css")

# Titre de l'application avec animation CSS et couleur blanche
st.markdown(
    """
    <div class="title">Reconnaissance d'émotions avec détection des contours du visage</div>
    """,
    unsafe_allow_html=True
)

# Section pour choisir entre la capture en temps réel et l'importation de photo
st.markdown(
    """
    <div class="card">
        <h2>Choisissez une option</h2>
        <p>Sélectionnez l'une des options ci-dessous pour commencer :</p>
    </div>
    """,
    unsafe_allow_html=True
)

# Options sous forme de cartes animées
col1, col2 = st.columns(2)

with col1:
    st.markdown(
        """
        <div class="card1">
            <div class="icon">📷</div>
            <h2>Capturer une photo en temps réel</h2>
            <p>Utilisez votre webcam pour capturer une photo en direct.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Capturer une photo"):
        st.session_state.option = "capture"

with col2:
    st.markdown(
        """
        <div class="card1">
            <div class="icon">📂</div>
            <h2>Importer une photo depuis votre PC</h2>
            <p>Téléversez une photo depuis votre ordinateur.</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    if st.button("Importer une photo"):
        st.session_state.option = "import"

# Gestion des options
if "option" not in st.session_state:
    st.session_state.option = None

if st.session_state.option == "capture":
    # Utiliser la webcam pour capturer des images en temps réel
    st.write("Veuillez activer votre webcam pour commencer la détection en temps réel.")
    picture = st.camera_input("Capturez une image")

    if picture:
        # Convertir l'image capturée en un tableau numpy
        frame = cv2.imdecode(np.frombuffer(picture.getvalue(), np.uint8), cv2.IMREAD_COLOR)
        if frame is None:
            st.error("Erreur : Impossible de lire l'image capturée.")
            st.stop()

        logging.debug(f"Taille de l'image : {frame.shape}")

        # Convertir l'image en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb_frame is None:
            st.error("Erreur : Impossible de convertir l'image en RGB.")
            st.stop()

        logging.debug(f"Taille de l'image RGB : {rgb_frame.shape}")

        # Détecter les points du visage avec MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            logging.debug("Visage détecté !")
            for face_landmarks in results.multi_face_landmarks:
                # Dessiner les contours du visage sur l'image
                frame_with_landmarks = draw_face_landmarks(frame)

                # Convertir l'image en RGB pour l'affichage dans Streamlit
                frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)

                # Afficher l'image avec les contours du visage
                st.image(frame_rgb, caption="Image capturée avec les contours du visage", use_container_width=True)

                # Prédire l'émotion et la confiance
                emotion, confidence = predict_emotion(face_landmarks, frame)
                if emotion:
                    # Afficher l'émotion prédite et la confiance
                    st.write(f"Émotion prédite : {emotion} (Confiance : {confidence * 100:.2f}%)")

                    # Ajouter un bouton de téléchargement pour l'image traitée
                    st.download_button(
                        label="Télécharger la photo traitée",
                        data=cv2.imencode('.jpg', frame_with_landmarks)[1].tobytes(),
                        file_name="photo_traitée.jpg",
                        mime="image/jpeg"
                    )
        else:
            logging.debug("Aucun visage détecté.")
            st.warning("Aucun visage détecté dans l'image capturée.")

elif st.session_state.option == "import":
    # Section pour importer une photo depuis le PC local
    st.write("Importez une photo depuis votre ordinateur pour la reconnaissance d'émotions.")
    uploaded_file = st.file_uploader("Choisissez une image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Lire l'image téléversée
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

        if frame is None:
            st.error("Erreur : Impossible de lire l'image téléversée.")
            st.stop()

        logging.debug(f"Taille de l'image importée : {frame.shape}")

        # Convertir l'image en RGB pour MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if rgb_frame is None:
            st.error("Erreur : Impossible de convertir l'image en RGB.")
            st.stop()

        logging.debug(f"Taille de l'image RGB : {rgb_frame.shape}")

        # Détecter les points du visage avec MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)
        if results.multi_face_landmarks:
            logging.debug("Visage détecté !")
            for face_landmarks in results.multi_face_landmarks:
                # Dessiner les contours du visage sur l'image
                frame_with_landmarks = draw_face_landmarks(frame)

                # Convertir l'image en RGB pour l'affichage dans Streamlit
                frame_rgb = cv2.cvtColor(frame_with_landmarks, cv2.COLOR_BGR2RGB)

                # Afficher l'image avec les contours du visage
                st.image(frame_rgb, caption="Image importée avec les contours du visage", use_container_width=True)

                # Prédire l'émotion et la confiance
                emotion, confidence = predict_emotion(face_landmarks, frame)
                if emotion:
                    # Afficher l'émotion prédite et la confiance
                    st.write(f"Émotion prédite : {emotion} (Confiance : {confidence * 100:.2f}%)")

                    # Ajouter un bouton de téléchargement pour l'image traitée
                    st.download_button(
                        label="Télécharger la photo traitée",
                        data=cv2.imencode('.jpg', frame_with_landmarks)[1].tobytes(),
                        file_name="photo_traitée.jpg",
                        mime="image/jpeg"
                    )
        else:
            logging.debug("Aucun visage détecté.")
            st.warning("Aucun visage détecté dans l'image importée.")