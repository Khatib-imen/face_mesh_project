# Reconnaissance d’Émotions en Temps Réel avec Détection des Contours du Visage

## Auteur
imen khatib

## Date
07/03/2025

## Résumé
**Objectif :**  
Ce projet vise à développer une application de reconnaissance d’émotions en temps réel en utilisant des techniques de vision par ordinateur et d’apprentissage profond.

**Résultats clés :**  
L’application peut détecter 7 émotions avec une précision moyenne de 65 % et affiche les résultats en temps réel via une interface web.

---

## Introduction
**Contexte :**  
La reconnaissance d’émotions est utilisée dans de nombreux domaines, comme la psychologie, le marketing, et les interfaces homme-machine.

**Objectifs :**  
L’objectif est de créer une application web qui détecte les émotions à partir d’une webcam ou d’une photo importée.

**Problématique :**  
Les défis incluent la détection précise des émotions sur des images de faible résolution.

---

## Méthodologie
**Architecture du modèle :**  
Un réseau de neurones convolutifs (CNN) a été utilisé pour la classification des émotions. Le modèle est composé de 2 couches convolutives, 2 couches de pooling, et 2 couches fully connected.

**Dataset :**  
Le modèle a été entraîné sur le dataset FER-2013, qui contient 35 886 images en niveaux de gris (48x48 pixels) réparties en 7 classes d’émotions.

**Prétraitement des données :**  
Les images ont été redimensionnées à 48x48 pixels, converties en niveaux de gris, et normalisées.

**Technologies utilisées :**  
- Python
- TensorFlow
- OpenCV
- MediaPipe
- Streamlit

---

## Résultats
**Performances du modèle :**  
Le modèle atteint une précision de 65 % sur le jeu de test. La matrice de confusion montre que les émotions 'Happy' et 'Sad' sont bien reconnues, tandis que 'Fear' et 'Disgust' sont plus difficiles à distinguer.

**Exemples de prédictions :**  
Voici des captures d’écran de l’application en action, montrant les émotions prédites et les contours du visage détectés.

**Limites :**  
Le modèle a des difficultés avec les expressions subtiles ou ambiguës, et la confiance des prédictions est parfois faible.

---

## Discussion
**Analyse des résultats :**  
La précision de 65 % est acceptable pour un premier prototype, mais elle pourrait être améliorée avec un modèle plus complexe ou un dataset plus riche.

**Comparaison avec d’autres travaux :**  
D’autres modèles basés sur des architectures plus avancées (comme ResNet ou Vision Transformers) atteignent des précisions supérieures à 80 %.

**Améliorations possibles :**  
- Utiliser un modèle pré-entraîné
- Augmenter le dataset
- Ajouter des techniques de post-traitement

---

## Conclusion
**Bilan du projet :**  
Ce projet a permis de développer une application fonctionnelle de reconnaissance d’émotions en temps réel, avec une interface web intuitive.

**Perspectives :**  
Cette application pourrait être utilisée dans des systèmes de surveillance émotionnelle ou des interfaces interactives.

---

## Annexes

**Captures d’écran :**  
![image](https://github.com/user-attachments/assets/3745b6b6-75b2-4b72-a090-031f65f7bf5b)
![image](https://github.com/user-attachments/assets/6566b8b3-ef1c-4126-9a62-bdbb4520c5a0)


---

## Bibliographie
- FER-2013 Dataset : [Lien vers le dataset](https://www.kaggle.com/datasets/msambare/fer2013)
