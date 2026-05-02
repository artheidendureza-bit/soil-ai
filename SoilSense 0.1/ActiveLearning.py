import os
import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from PIL import Image
from tqdm import tqdm

from MainMicrobiome import (
    AI_COUNT, CLASS_NAMES, get_model_variant, get_model_path, get_optimizer,
    get_train_transform, inference_transform, mixup_data, cutmix_data, mixup_criterion
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HARD_MINING_FILES = [
    "wrong_predictions.csv",
    "uncertain_predictions.csv",
    "ensemble_disagreements.csv"
]
REMEDIAL_EPOCHS = 10
RANDOM_TRAIN_FRACTION = 0.20  

GLOBAL_IMAGE_CACHE = {}

def preload_to_ram(samples):
    """Reuse the high-speed RAM preloading logic."""
    if not GLOBAL_IMAGE_CACHE:
        print(f"[*] Preloading {len(samples)} remedial images into RAM...")
        for path, _ in tqdm(samples, desc="Preloading RAM"):
            if path not in GLOBAL_IMAGE_CACHE:
                try:
                    from PIL import Image
                    img = Image.open(path).convert('RGB')
                    img = img.resize((128, 128), Image.Resampling.BILINEAR)
                    GLOBAL_IMAGE_CACHE[path] = img.copy()
                except Exception:
                    continue

class RemedialDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        preload_to_ram(self.samples)
        # Only keep samples that loaded successfully
        self.samples = [(p, t) for p, t in self.samples if p in GLOBAL_IMAGE_CACHE]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        sample = GLOBAL_IMAGE_CACHE[path]
        if self.transform:
            sample = self.transform(sample)
        return sample, target

def extract_hard_samples():
    hard_samples_dict = {} 
    
    for filename in HARD_MINING_FILES:
        if not os.path.exists(filename):
            print(f"[-] Missing {filename}, skipping...")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                path = row['image_path']
                label_str = row['actual']
                if label_str in CLASS_NAMES:
                    label_idx = CLASS_NAMES.index(label_str)
                    hard_samples_dict[path] = label_idx
                    
    hard_samples = list(hard_samples_dict.items())
    print(f"[*] Found {len(hard_samples)} unique hard examples to study.")
    return hard_samples

def active_learning_loop():
    print("=" * 60)
    print("🚀 INITIATING ACTIVE LEARNING REMEDIAL PIPELINE")
    print("=" * 60)
    
    hard_samples = extract_hard_samples()
    if not hard_samples:
        print("[!] No hard examples found. The AI is already perfect, or you haven't run TestModel.py yet.")
        return

    if not os.path.exists("data/train"):
        print("[!] ERROR: data/train not found.")
        return
        
    base_train_dataset = ImageFolder("data/train")
    total_train = len(base_train_dataset)
    
    mix_in_count = int(total_train * RANDOM_TRAIN_FRACTION)
    normal_indices = random.sample(range(total_train), mix_in_count)
    print(f"[*] Mixing in {mix_in_count} normal examples to prevent catastrophic forgetting.")

    for model_idx in range(AI_COUNT):
        model_path = get_model_path(model_idx)
        if not os.path.exists(model_path):
            print(f"[!] AI {model_idx+1} not found, skipping...")
            continue
            
        print(f"\n--- Retraining AI {model_idx+1} ---")
        model = get_model_variant(model_idx).to(DEVICE, memory_format=torch.channels_last)
        
        model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        
        transform = get_train_transform(model_idx)
        
        # Mix hard samples with some random normal samples to prevent forgetting
        normal_samples = [base_train_dataset.samples[i] for i in normal_indices]
        remedial_samples = hard_samples + normal_samples
        
        remedial_dataset = RemedialDataset(remedial_samples, transform=transform)
        loader = DataLoader(remedial_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        model.train()
        for epoch in range(REMEDIAL_EPOCHS):
            total_loss = 0.0
            loop = tqdm(loader, desc=f"AI {model_idx+1} - Remedial Epoch {epoch+1}/{REMEDIAL_EPOCHS}", unit="batch")
            
            for images, labels in loop:
                images, labels = images.to(DEVICE, memory_format=torch.channels_last), labels.to(DEVICE)
                optimizer.zero_grad()
                
                import numpy as np
                r = np.random.rand()
                if r < 0.15:
                    mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=1.0)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                elif r < 0.30:
                    mixed_images, labels_a, labels_b, lam = cutmix_data(images, labels, alpha=1.0)
                    outputs = model(mixed_images)
                    loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
                else:
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")
                
        torch.save(model.state_dict(), model_path)
        if model_idx == 0:
            torch.save(model.state_dict(), "microbiome.pth")
        print(f"[*] AI {model_idx+1} successfully learned from its mistakes!")

    print("\n" + "=" * 60)
    print("✅ ACTIVE LEARNING COMPLETE! The ensemble is now smarter.")
    print("=" * 60)

if __name__ == "__main__":
    active_learning_loop()
