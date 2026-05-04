import os
import sys
import json
import torch
import optuna
import gc
from torch.utils.data import DataLoader
from tqdm import tqdm

from MainMicrobiome import (
    MicrobiomeBrain, CachedImageDataset, get_train_transform,
    inference_transform, CLASS_NAMES, SAM, FocalLoss
)

def objective(trial):
    print(f"\n[⚡] Starting Trial #{trial.number}...")
    # 1. Hyperparameter Search Space
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    dropout = trial.suggest_float("dropout", 0.2, 0.6)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. Data Setup (Using high-speed cache - Soil Only)
    try:
        from torchvision.datasets import ImageFolder
        full_ds = ImageFolder("data/train")
        soil_indices = [i for i, (p, t) in enumerate(full_ds.samples) if full_ds.classes[t] != "NOT_SOIL"]
        # Use simple label mapping for the 4 ensemble classes
        samples = [full_ds.samples[i] for i in soil_indices]
        train_dataset = CachedImageDataset(samples, transform=get_train_transform(0))
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    except Exception as e:
        print(f"[!] Data Error: {e}")
        return 0.0
    
    # 3. Model Setup (Championship Specs)
    print(f"[*] Moving {115 if trial.number > 5 else 32}M Parameter Brain to {device}...")
    model = MicrobiomeBrain(base_channels=32, d1=dropout).to(device)
    optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=weight_decay)
    criterion = FocalLoss()
    
    # 4. Training Loop (Small burst for tuning)
    model.train()
    total_acc = 0
    for epoch in range(3):
        correct, total = 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            # SAM Step 1
            optimizer.zero_grad()
            outputs = model(images)
            criterion(outputs, labels).backward()
            optimizer.first_step(zero_grad=True)
            
            # SAM Step 2
            criterion(model(images), labels).backward()
            optimizer.second_step(zero_grad=True)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        total_acc = 100 * correct / total
        print(f"   - Epoch {epoch+1}/3: Accuracy {total_acc:.2f}%")
        trial.report(total_acc, epoch)
        if trial.should_prune():
            # Clean up before pruning
            del model, optimizer, train_loader
            gc.collect()
            if torch.cuda.is_available(): torch.cuda.empty_cache()
            raise optuna.exceptions.TrialPruned()
            
    # 5. Final Cleanup
    del model, optimizer, train_loader
    gc.collect()
    if torch.cuda.is_available(): torch.cuda.empty_cache()
            
    return total_acc

if __name__ == "__main__":
    print("="*50)
    print("🚀 SOILSENSE ULTIMATE HYPER-AUTO-TUNER")
    print("="*50)
    
    if not os.path.exists("data/train"):
        print("❌ ERROR: No training data found. Run SetupMicrobiomeData.py first!")
        sys.exit(1)

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=10)
    
    print("\n" + "="*50)
    print("✅ OPTIMIZATION COMPLETE")
    print(f"🏆 Best Accuracy: {study.best_value:.2f}%")
    
    # SAVE PARAMS (This is the magic fix for MasterScript!)
    with open("best_params.json", "w") as f:
        json.dump(study.best_params, f, indent=4)
    
    print("✅ Saved: best_params.json")
    print("="*50)