import os
import cv2
import numpy as np

# Chemin vers le dataset organisé
dataset_dir = "dataset"

# Fonction pour prétraiter une image
def preprocess_image(image_path, target_size=(48, 48)):
    # Charger l'image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        print(f"Erreur : Impossible de charger l'image à partir de {image_path}")
        return None
    # Redimensionner l'image
    image = cv2.resize(image, target_size)
    # Normaliser les valeurs des pixels
    image = image / 255.0
    return image

# Prétraiter les ensembles d'entraînement et de test
def preprocess_dataset(dataset_dir, target_size=(48, 48)):
    images = []
    labels = []
    label_to_index = {emotion: i for i, emotion in enumerate(os.listdir(dataset_dir))}

    for emotion in os.listdir(dataset_dir):
        emotion_dir = os.path.join(dataset_dir, emotion)
        if os.path.isdir(emotion_dir):
            for img_name in os.listdir(emotion_dir):
                img_path = os.path.join(emotion_dir, img_name)
                print(f"Traitement de l'image : {img_path}")  # Afficher le chemin de l'image
                # Prétraiter l'image
                img = preprocess_image(img_path, target_size)
                if img is not None:  # Ignorer les images invalides
                    images.append(img)
                    labels.append(label_to_index[emotion])

    return np.array(images), np.array(labels)

# Prétraiter les données
X_train, y_train = preprocess_dataset(os.path.join(dataset_dir, "train"))
X_test, y_test = preprocess_dataset(os.path.join(dataset_dir, "test"))

# Sauvegarder les données prétraitées
np.save("X_train.npy", X_train)
np.save("y_train.npy", y_train)
np.save("X_test.npy", X_test)
np.save("y_test.npy", y_test)

print("Données prétraitées et sauvegardées avec succès !")