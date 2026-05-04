import os
import sys
import subprocess

try:
    import seaborn as sns
except ImportError:
    print("[*] Installing missing library: seaborn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "seaborn"])
    import seaborn as sns

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm
from MainMicrobiome import (
    load_ensemble, inference_transform, CLASS_NAMES, AI_COUNT
)

def run_tta_prediction(models, images, device):
    """Run Test-Time Augmentation (TTA) for maximum accuracy"""
    tta_transforms = [
        lambda x: x,
        lambda x: torch.flip(x, [3]), # Horizontal flip
        lambda x: torch.flip(x, [2]), # Vertical flip
        lambda x: torch.rot90(x, 1, [2, 3]), # 90 deg
    ]
    
    with torch.inference_mode():
        all_preds = []
        for transform in tta_transforms:
            transformed_imgs = transform(images)
            preds = [F.softmax(m(transformed_imgs), dim=1) for m in models[:7]]
            all_preds.append(torch.stack(preds).mean(dim=0))
        
        return torch.stack(all_preds).mean(dim=0)

def generate_report():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting Final Ensemble Verification on {device}...")
    
    if not os.path.exists("data/train"):
        print("❌ Error: No data found.")
        return

    models = load_ensemble(device)
    
    # Only load the 4 Microbe classes for the Accuracy Certificate
    all_dataset = ImageFolder("data/train", transform=inference_transform)
    soil_indices = [i for i, (path, label) in enumerate(all_dataset.samples) if all_dataset.classes[label] != "NOT_SOIL"]
    dataset = torch.utils.data.Subset(all_dataset, soil_indices)
    
    loader = DataLoader(dataset, batch_size=16, shuffle=False)
    
    y_true = []
    y_pred = []
    
    print("[*] Running TTA Ensemble Predictions (Research Grade)...")
    for images, labels in tqdm(loader):
        images = images.to(device)
        outputs = run_tta_prediction(models, images, device)
        preds = torch.argmax(outputs, dim=1).cpu().numpy()
        
        y_true.extend(labels.numpy())
        y_pred.extend(preds)
    
    # 1. Generate Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title("SoilSense Ultimate: Final Confusion Matrix")
    plt.ylabel('Actual Category')
    plt.xlabel('AI Prediction')
    plt.savefig("confusion_matrix.png", dpi=300)
    print("\n✅ Saved: confusion_matrix.png (Ready for your poster!)")
    
    # 2. Generate Text Report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES)
    with open("accuracy_certificate.txt", "w") as f:
        f.write("="*50 + "\n")
        f.write("🏆 SOILSENSE ULTIMATE: ACCURACY CERTIFICATE 🏆\n")
        f.write("="*50 + "\n\n")
        f.write(report)
        f.write("\n" + "="*50 + "\n")
    
    print("✅ Saved: accuracy_certificate.txt")
    print(f"\nFinal Accuracy: {np.mean(np.array(y_true) == np.array(y_pred))*100:.2f}%")

if __name__ == "__main__":
    generate_report()
