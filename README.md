# Générateur de Notes DSI — Agent IA connecté à Gmail

Application qui se connecte à une boîte Gmail, analyse un email sélectionné (et sa pièce jointe éventuelle), puis génère automatiquement une note administrative au format Word (.docx), avec l'en-tête et le pied de page officiels de la DSI.

## Fonctionnalités

- Connexion sécurisée à Gmail (OAuth2, lecture seule)
- Classification automatique du type de document joint (Facture, Courrier officiel, Formulaire, Note interne) via un modèle CNN (ResNet18)
- Extraction et rédaction automatique du contenu de la note via un système multi-agents (CrewAI + Ollama)
- Génération d'un document Word structuré, prêt à télécharger
- Interface web simple via Streamlit

## Structure du projet

```
projet_notes_dsi/
├── app.py                   # Interface Streamlit (point d'entrée)
├── agents.py                # Définition des agents CrewAI
├── tasks.py                 # Définition des tâches CrewAI
├── gmail_service.py         # Connexion et lecture de la boîte Gmail
├── document_classifier.py   # Chargement et utilisation du modèle CNN
├── train_classifier.py      # Script d'entraînement du modèle CNN
├── document_classifier.pt   # Modèle CNN entraîné
├── utils.py                 # Génération du document Word + parsing JSON
├── logo1.jpg / logo2.jpg    # Logos utilisés dans l'en-tête/pied de page
├── requirements.txt         # Dépendances Python
├── confusion_matrix.png     # Résultat d'évaluation du modèle
├── loss_curve.png           # Courbe d'entraînement du modèle
```

## Prérequis

- Windows avec accès Internet
- Python 3.10 ou plus récent
- [Ollama](https://ollama.com/download) installé, avec le modèle `mistral` téléchargé
- Un compte Gmail personnel

## Installation

### 1. Créer vos identifiants Google Cloud

Chaque utilisateur doit générer son propre fichier `credentials.json` :

1. Aller sur [console.cloud.google.com](https://console.cloud.google.com/) et créer un projet.
2. Activer la **Gmail API**.
3. Configurer l'écran de consentement OAuth (type Externe) et s'ajouter soi-même comme utilisateur de test.
4. Créer un identifiant OAuth de type **Application de bureau**.
5. Télécharger le fichier JSON, le renommer en `credentials.json`, et le placer à la racine du projet.

⚠️ Ce fichier est personnel et confidentiel : ne jamais le partager ni le publier.

### 2. Installer le projet

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Installer et lancer Ollama

```bash
ollama pull mistral
```

Vérifier qu'Ollama tourne en arrière-plan (icône dans la barre des tâches).

## Lancer l'application

```bash
streamlit run app.py
```

L'application s'ouvre automatiquement dans le navigateur à l'adresse `http://localhost:8501`.

À la première utilisation, une fenêtre Google s'ouvre pour autoriser l'accès en lecture seule à votre Gmail. Un fichier `token.json` est alors créé automatiquement pour mémoriser la connexion.

## Utilisation

1. Cliquer sur **Charger les emails**.
2. Sélectionner un email dans la liste.
3. Cliquer sur **Générer la note DSI**.
4. Relire les champs générés (Cadre, Objet, Participants, Descriptif, Prochaine Action).
5. Télécharger la note au format Word.

## Sécurité

- L'accès à Gmail est en lecture seule uniquement (aucun envoi, suppression ou modification possible).
- Les fichiers `credentials.json` et `token.json` sont personnels et ne doivent jamais être partagés.
- Le traitement du texte est effectué localement via Ollama : aucune donnée n'est envoyée à un service cloud externe.

## Technologies utilisées

| Composant | Technologie |
|---|---|
| Interface | Streamlit |
| Messagerie | Gmail API (OAuth2) |
| Orchestration IA | CrewAI |
| Modèle de langage | Ollama (Mistral, local) |
| Classification de documents | CNN ResNet18 (PyTorch) |
| Génération du document | python-docx |

## Auteur

Manal SAWLI — Stage DSI, Ministère de l'Agriculture, de la Pêche Maritime, du Développement Rural et des Eaux et Forêts
