"""
Script tout-en-un : cree un petit dataset de documents synthetiques
localement (aucun telechargement), puis entraine un CNN dessus.

Lancer avec : py train_classifier.py
Sortie : document_classifier.pt (modele entraine)
         confusion_matrix.png, loss_curve.png
"""

import os
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pyplot as plt

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ["Facture", "Courrier officiel", "Formulaire", "Note interne"]
LABEL_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}
DATA_DIR = "synthetic_dataset"
IMAGES_PER_CLASS = 120
IMG_SIZE = 224
BATCH_SIZE = 16
EPOCHS = 6
DROPOUT = 0.4


def generate_image(doc_type, idx):
    img = Image.new("RGB", (600, 800), color="white")
    draw = ImageDraw.Draw(img)
    try:
        font_title = ImageFont.truetype("arial.ttf", 26)
        font_text = ImageFont.truetype("arial.ttf", 16)
    except Exception:
        font_title = ImageFont.load_default()
        font_text = ImageFont.load_default()

    if doc_type == "Facture":
        draw.text((30, 30), "FACTURE", font=font_title, fill="black")
        draw.text((30, 80), f"N FAC-{1000+idx}", font=font_text, fill="black")
        draw.rectangle([30, 150, 570, 400], outline="black", width=2)
        for y in range(170, 380, 40):
            draw.line([(30, y), (570, y)], fill="gray", width=1)
        draw.text((30, 420), f"Total TTC : {random.randint(100,5000)} MAD", font=font_text, fill="black")

    elif doc_type == "Courrier officiel":
        draw.text((30, 30), "OBJET : Correspondance officielle", font=font_title, fill="black")
        draw.text((30, 100), "Madame, Monsieur,", font=font_text, fill="black")
        for y in range(140, 500, 30):
            draw.line([(30, y), (random.randint(400, 570), y)], fill="black", width=1)
        draw.text((30, 550), "Veuillez agreer, Madame, Monsieur,", font=font_text, fill="black")
        draw.text((30, 580), "l'expression de mes salutations distinguees.", font=font_text, fill="black")

    elif doc_type == "Formulaire":
        draw.text((30, 30), "FORMULAIRE", font=font_title, fill="black")
        for i, label in enumerate(["Nom :", "Prenom :", "Date :", "Adresse :", "Signature :"]):
            y = 100 + i * 80
            draw.text((30, y), label, font=font_text, fill="black")
            draw.rectangle([180, y - 5, 570, y + 30], outline="black", width=1)

    elif doc_type == "Note interne":
        draw.text((30, 30), "NOTE INTERNE", font=font_title, fill="black")
        draw.text((30, 80), "De : Direction", font=font_text, fill="black")
        draw.text((30, 110), "A : Service concerne", font=font_text, fill="black")
        for y in range(160, 350, 25):
            draw.line([(30, y), (random.randint(300, 570), y)], fill="black", width=1)

    pixels = np.array(img)
    noise = np.random.randint(0, 15, pixels.shape, dtype="uint8")
    pixels = np.clip(pixels.astype(int) - noise, 0, 255).astype("uint8")
    return Image.fromarray(pixels)


def build_synthetic_dataset():
    if os.path.exists(DATA_DIR):
        shutil.rmtree(DATA_DIR)
    for cls in CLASS_NAMES:
        os.makedirs(os.path.join(DATA_DIR, cls), exist_ok=True)
        for i in range(IMAGES_PER_CLASS):
            img = generate_image(cls, i)
            img.save(os.path.join(DATA_DIR, cls, f"{cls}_{i}.png"))
    print(f"Dataset synthetique cree dans ./{DATA_DIR} ({len(CLASS_NAMES)} classes x {IMAGES_PER_CLASS} images)")


class FolderDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.samples = []
        self.transform = transform
        for cls in CLASS_NAMES:
            folder = os.path.join(root_dir, cls)
            for fname in os.listdir(folder):
                self.samples.append((os.path.join(folder, fname), LABEL_TO_IDX[cls]))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label


def build_model(num_classes, dropout=DROPOUT):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for param in model.parameters():
        param.requires_grad = False
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout),
        nn.Linear(128, num_classes)
    )
    return model.to(DEVICE)


def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * imgs.size(0)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    all_preds, all_labels = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * imgs.size(0)
        preds = outputs.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    return total_loss / len(loader.dataset), accuracy_score(all_labels, all_preds), all_labels, all_preds


def main():
    print(f"Device : {DEVICE}")
    build_synthetic_dataset()

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = FolderDataset(DATA_DIR, transform)
    n_val = int(0.2 * len(full_dataset))
    n_train = len(full_dataset) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = build_model(len(CLASS_NAMES))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.fc.parameters(), lr=1e-4)

    history = {"train_loss": [], "val_loss": [], "val_acc": []}

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion)
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        print(f"Epoch {epoch}/{EPOCHS} - train_loss: {train_loss:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "class_names": CLASS_NAMES,
        "dropout": DROPOUT,
    }, "document_classifier.pt")
    print("Modele sauvegarde -> document_classifier.pt")

    _, _, y_true, y_pred = evaluate(model, val_loader, criterion)
    print("\n=== Rapport de classification ===")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Matrice de confusion")
    plt.colorbar()
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")

    plt.figure()
    plt.plot(history["train_loss"], label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
    plt.legend()
    plt.title("Courbe de perte")
    plt.savefig("loss_curve.png")

    print("Fichiers generes : document_classifier.pt, confusion_matrix.png, loss_curve.png")


if __name__ == "__main__":
    main()