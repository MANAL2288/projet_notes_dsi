"""
Charge le CNN entraîné (document_classifier.pt) et l'expose comme
un outil CrewAI utilisable par un agent.

Gère aussi la conversion PDF -> image (première page) puisque les
pièces jointes Gmail arrivent souvent en PDF.
"""

import os
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from crewai.tools import tool

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "document_classifier.pt")

_model = None
_class_names = None

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _load_model():
    """Charge le modèle une seule fois (lazy loading), pour ne pas
    recharger les poids à chaque appel."""
    global _model, _class_names

    if _model is not None:
        return _model, _class_names

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Modèle introuvable à {MODEL_PATH}. "
            "Lance d'abord train_classifier.py."
        )

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
    class_names = checkpoint["class_names"]
    dropout = checkpoint.get("dropout", 0.4)

    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(128, len(class_names))
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    _model = model
    _class_names = class_names
    return _model, _class_names


def _pdf_to_image(pdf_path):
    """Convertit la première page d'un PDF en image PIL."""
    from pdf2image import convert_from_path
    pages = convert_from_path(pdf_path, first_page=1, last_page=1, dpi=150)
    return pages[0].convert("RGB")


def classify_document_file(file_path: str) -> dict:
    """
    Classifie un fichier (PDF ou image) et retourne :
    {"label": "Facture", "confidence": 0.87}
    """
    model, class_names = _load_model()

    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        img = _pdf_to_image(file_path)
    else:
        img = Image.open(file_path).convert("RGB")

    tensor = _transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs, dim=1)[0]
        pred_idx = int(torch.argmax(probs).item())
        confidence = float(probs[pred_idx].item())

    return {
        "label": class_names[pred_idx],
        "confidence": round(confidence, 3)
    }


# ─── OUTIL CREWAI ──────────────────────────────────────────────────────────
@tool("Classificateur de type de document")
def classify_document_tool(file_path: str) -> str:
    """
    Prend le chemin d'un fichier (PDF ou image) correspondant à une pièce
    jointe d'email, et retourne le type de document détecté par un CNN
    entraîné (ex: Facture, Courrier officiel, Formulaire, Note interne)
    ainsi que le niveau de confiance.
    """
    try:
        result = classify_document_file(file_path)
        return f"Type de document détecté : {result['label']} (confiance : {result['confidence']})"
    except Exception as e:
        return f"Erreur lors de la classification : {e}"