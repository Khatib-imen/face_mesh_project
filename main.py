# main.py
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Charger le modèle de reconnaissance d'émotions
emotion_model = tf.keras.models.load_model('emotion_model.h5')
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Fonction pour prédire l'émotion à partir des points du visage
def predict_emotion(face_landmarks, frame):
    """
    Prédit l'émotion à partir des points du visage détectés.
    Retourne l'émotion prédite et la confiance associée.
    """
    # Extraire les coordonnées des points du visage
    x_coords = [landmark.x * frame.shape[1] for landmark in face_landmarks.landmark]
    y_coords = [landmark.y * frame.shape[0] for landmark in face_landmarks.landmark]
    x_min, x_max = int(min(x_coords)), int(max(x_coords))
    y_min, y_max = int(min(y_coords)), int(max(y_coords))

    # Recadrer la région du visage
    face_roi = frame[y_min:y_max, x_min:x_max]
    if face_roi.size == 0:
        return None, None  # Si la région du visage est vide, retourner None

    # Prétraiter l'image pour le modèle de reconnaissance d'émotions
    face_roi = cv2.resize(face_roi, (48, 48))  # Redimensionner à 48x48 pixels
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)  # Convertir en niveaux de gris
    face_roi = np.expand_dims(face_roi, axis=-1)  # Ajouter une dimension pour le canal de couleur
    face_roi = np.expand_dims(face_roi, axis=0)  # Ajouter une dimension pour le batch
    face_roi = face_roi / 255.0  # Normaliser les valeurs des pixels entre 0 et 1

    # Prédire l'émotion avec le modèle
    predictions = emotion_model.predict(face_roi)
    emotion_index = np.argmax(predictions)  # Index de l'émotion prédite
    emotion = emotion_labels[emotion_index]  # Récupérer l'émotion prédite
    confidence = predictions[0][emotion_index]  # Récupérer la confiance de la prédiction

    return emotion, confidence

# Fonction pour dessiner les contours du visage
def draw_face_landmarks(frame):
    """
    Dessine les contours du visage sur l'image.
    """
    # Convertir l'image en RGB pour MediaPipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les points du visage avec MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Dessiner les contours du visage sur l'image
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_CONTOURS,  # Utiliser FACEMESH_CONTOURS pour les contours
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
    return frame