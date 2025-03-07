import cv2
import mediapipe as mp

# Initialiser MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True)

# Initialiser la webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Erreur : Impossible d'accéder à la webcam.")
else:
    print("Webcam détectée et accessible.")

# Boucle principale pour capturer et traiter les images en temps réel
while cap.isOpened():
    # Lire une image de la webcam
    ret, frame = cap.read()
    if not ret:
        print("Erreur : Impossible de lire l'image de la webcam.")
        break

    # Convertir l'image de BGR (format OpenCV) en RGB (format MediaPipe)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Détecter les points du visage avec MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)

    # Si des points du visage sont détectés
    if results.multi_face_landmarks:
        print("Visage détecté !")
        for face_landmarks in results.multi_face_landmarks:
            # Dessiner les points du visage sur l'image
            mp.solutions.drawing_utils.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION,  # Utiliser FACEMESH_TESSELATION
                landmark_drawing_spec=None,
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_contours_style()
            )
    else:
        print("Aucun visage détecté.")

    # Afficher l'image avec les points du visage
    cv2.imshow('Face Mesh', frame)

    # Quitter la boucle si la touche 'q' est pressée
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Libérer la webcam et fermer toutes les fenêtres OpenCV
cap.release()
cv2.destroyAllWindows()