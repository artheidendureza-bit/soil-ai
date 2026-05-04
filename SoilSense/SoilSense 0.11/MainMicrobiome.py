import os
import sys
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image, ImageFile
from tqdm import tqdm
import numpy as np

# --- IMPORT CORE AI FROM ENGINE ---
from MicrobiomeEngine import (
    SAM, FocalLoss, MicrobiomeBrain, SentinelBrain,
    get_model_variant, mixup_data, mixup_criterion, cutmix_data,
    generate_cam, dream_feature, sentinel_check
)

# --- RE-EXPORT FOR BACKWARD COMPATIBILITY ---
__all__ = [
    'SAM', 'FocalLoss', 'MicrobiomeBrain', 'SentinelBrain',
    'get_model_variant', 'mixup_data', 'mixup_criterion', 'cutmix_data',
    'generate_cam', 'dream_feature', 'sentinel_check',
    'AI_COUNT', 'CLASS_NAMES', 'load_ensemble', 'inference_transform',
    'CachedImageDataset', 'get_train_transform', 'get_optimizer', 'get_confidence'
]

# --- CONSTANTS ---
ImageFile.LOAD_TRUNCATED_IMAGES = True
AI_COUNT = 8
MODEL_FILE_TEMPLATE = "microbiome_{}.pth"
CLASS_NAMES = ["DIRT", "GRASS", "LEAF", "MIX"]
VAL_FRACTION = 0.20
EARLY_STOPPING_PATIENCE = 15
torch.backends.cudnn.benchmark = True

GLOBAL_IMAGE_CACHE = {}

# --- DATA HANDLING ---
class CachedImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, transform=None):
        if isinstance(samples, str):
            from torchvision.datasets import ImageFolder
            samples = ImageFolder(samples).samples
        self.samples = samples
        self.transform = transform
        if not GLOBAL_IMAGE_CACHE:
            print(f"[*] Initializing Global RAM Cache ({len(samples)} images)...")
            for item in tqdm(samples, desc="Preloading RAM"):
                path, target = item
                if isinstance(path, str):
                    if path not in GLOBAL_IMAGE_CACHE:
                        try:
                            img = Image.open(path).convert('RGB').resize((128, 128), Image.Resampling.BILINEAR)
                            GLOBAL_IMAGE_CACHE[path] = img
                        except Exception:
                            if os.path.exists(path): os.remove(path)
                else:
                    # It's already a PIL image or tensor, just use a dummy key
                    pass
        
        # If samples are already (Image, Target), we use them directly
        self.samples = samples

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, target = self.samples[idx]
        if isinstance(path, str):
            sample = GLOBAL_IMAGE_CACHE[path]
        else:
            sample = path # Already an image
        if self.transform: sample = self.transform(sample)
        return sample, target

class SentinelDataset(torch.utils.data.Dataset):
    """Wraps a dataset to convert it into a binary 'Soil vs Not Soil' task."""
    def __init__(self, dataset, not_soil_idx):
        self.dataset = dataset
        self.not_soil_idx = not_soil_idx

    def __len__(self): return len(self.dataset)
    def __getitem__(self, idx):
        img, target = self.dataset[idx]
        # Soil = 0, Not Soil = 1
        new_target = 1 if target == self.not_soil_idx else 0
        return img, new_target

def get_train_transform(index):
    common = [transforms.RandomSolarize(128, p=0.2), transforms.RandomEqualize(p=0.2), transforms.ColorJitter(hue=0.1), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    recipes = [
        [transforms.RandomHorizontalFlip(p=0.5), transforms.RandomRotation(10)],
        [transforms.RandomRotation(30), transforms.ColorJitter(brightness=0.2, contrast=0.2)],
        [transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)), transforms.RandomGrayscale(p=0.2)],
        [transforms.RandomResizedCrop(128, scale=(0.8, 1.0)), transforms.RandomHorizontalFlip(p=0.5)],
        [transforms.RandomPerspective(distortion_scale=0.2, p=0.5)],
        [transforms.RandomRotation(45), transforms.RandomAffine(0, translate=(0.2, 0.2))],
        [transforms.ColorJitter(0.3, 0.3, 0.3, 0.2), transforms.RandomHorizontalFlip(p=0.5)]
    ]
    return transforms.Compose(recipes[index % 7] + common)


def inference_transform(img):
    return transforms.Compose([transforms.Resize((128, 128)), transforms.ToTensor(), transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])(img)

# --- ENSEMBLE MGMT ---
def get_model_path(index): return MODEL_FILE_TEMPLATE.format(index)

def load_ensemble(device):
    models = []
    for i in range(AI_COUNT):
        m = get_model_variant(i).to(device)
        path = get_model_path(i)
        if os.path.exists(path):
            try:
                m.load_state_dict(torch.load(path, map_location=device, weights_only=True))
            except Exception as e:
                print(f"[!] AI {i+1} has an architectural mismatch. It will need to be re-trained.")
        m.eval(); models.append(m)
    return models

def get_optimizer(params, lr, weight_decay):
    return SAM(params, torch.optim.AdamW, lr=lr, weight_decay=weight_decay)

def get_confidence(outputs):
    return torch.max(F.softmax(outputs, dim=1)).item()

# --- THE CHAMPIONSHIP PIPELINE ---
def train_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists("data/train"): return
    
    full_imagefolder = ImageFolder("data/train")
    # Separate indices for Soil vs Not Soil
    soil_indices = [i for i, (p, t) in enumerate(full_imagefolder.samples) if full_imagefolder.classes[t] != "NOT_SOIL"]
    ns_indices = [i for i, (p, t) in enumerate(full_imagefolder.samples) if full_imagefolder.classes[t] == "NOT_SOIL"]

    for i in range(AI_COUNT):
        model_path = get_model_path(i)
        if os.path.exists(model_path): continue

        print(f"\n[NEURAL] TRAINING AI {i+1}/{AI_COUNT}")
        
        # Determine Dataset based on Brain Role
        if i < 7:
            # Ensemble Brains only see Soil
            subset = torch.utils.data.Subset(full_imagefolder, soil_indices)
            class SoilOnlyDataset(torch.utils.data.Dataset):
                def __init__(self, subset): 
                    self.subset = subset
                    # Pre-extract samples to avoid heavy ImageFolder __getitem__ in RAM
                    self.samples = [subset.dataset.samples[i] for i in subset.indices]
                def __len__(self): return len(self.samples)
                def __getitem__(self, idx):
                    path, target = self.samples[idx]
                    return path, target # Return path for CachedImageDataset to handle
            base_dataset = SoilOnlyDataset(subset)
        else:
            # Sentinel Brain sees everything
            if not os.path.exists(os.path.join("data/train", "NOT_SOIL")):
                print("[*] NOT_SOIL folder not found. Skipping Sentinel training.")
                continue
            
            class SentinelWrapper(torch.utils.data.Dataset):
                def __init__(self, full_ds): 
                    self.full_ds = full_ds
                    self.ns_idx = full_ds.classes.index("NOT_SOIL")
                def __len__(self): return len(self.full_ds)
                def __getitem__(self, idx):
                    path, target = self.full_ds.samples[idx]
                    return path, target # Path for CachedImageDataset
            base_dataset = SentinelWrapper(full_imagefolder)
        
        lr, dropout, wd = 1e-3, 0.3, 1e-4
        if os.path.exists("best_params.json"):
            try:
                import json
                with open("best_params.json", "r") as f:
                    p = json.load(f)
                    lr, dropout, wd = p.get("lr", lr), p.get("dropout", dropout), p.get("weight_decay", wd)
            except: pass

        model = get_model_variant(i).to(device)
        # Apply tuned dropout if it's a MicrobiomeBrain (Sequential classifier)
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Sequential):
            for layer in model.modules():
                if isinstance(layer, nn.Dropout): layer.p = dropout
        
        # 80/20 Split
        indices = list(range(len(base_dataset)))
        random.seed(42); random.shuffle(indices)
        split = int(len(indices) * (1 - VAL_FRACTION))
        train_idx, val_idx = indices[:split], indices[split:]
        
        # Pass the pre-filtered samples directly for maximum speed
        samples = base_dataset.samples if hasattr(base_dataset, "samples") else [base_dataset[idx] for idx in range(len(base_dataset))]
        train_ds = CachedImageDataset(samples, transform=get_train_transform(i))
        val_ds = CachedImageDataset(samples, transform=inference_transform)
        
        train_loader = DataLoader(train_ds, batch_size=32, sampler=SubsetRandomSampler(train_idx))
        val_loader = DataLoader(val_ds, batch_size=32, sampler=SubsetRandomSampler(val_idx))
        full_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
        
        # Binary training for Sentinel (AI 8)
        if i == 7:
            ns_idx = full_imagefolder.classes.index("NOT_SOIL")
            train_loader = DataLoader(SentinelDataset(train_ds, ns_idx), batch_size=32, sampler=SubsetRandomSampler(train_idx))
            val_loader = DataLoader(SentinelDataset(val_ds, ns_idx), batch_size=32, sampler=SubsetRandomSampler(val_idx))
            full_loader = DataLoader(SentinelDataset(train_ds, ns_idx), batch_size=32, shuffle=True)
        
        optimizer = SAM(model.parameters(), torch.optim.AdamW, lr=lr, weight_decay=wd)
        criterion = FocalLoss()
        
        # SWA Setup
        swa_model = AveragedModel(model)
        swa_start = 35
        swa_scheduler = SWALR(optimizer, swa_lr=lr/10.0)

        best_acc = 0; patience = 0
        for epoch in range(50):
            model.train()
            loop = tqdm(train_loader, leave=False, desc=f"AI {i+1} Epoch {epoch+1}")
            for imgs, labels in loop:
                imgs, labels = imgs.to(device), labels.to(device)
                
                optimizer.zero_grad()
                if random.random() < 0.2:
                    m_imgs, la, lb, lam = mixup_data(imgs, labels)
                    mixup_criterion(criterion, model(m_imgs), la, lb, lam).backward()
                else:
                    criterion(model(imgs), labels).backward()
                
                optimizer.first_step(zero_grad=True)
                criterion(model(imgs), labels).backward()
                optimizer.second_step(zero_grad=True)
            
            if epoch >= swa_start:
                swa_model.update_parameters(model); swa_scheduler.step()
            
            model.eval(); correct, total = 0, 0
            with torch.inference_mode():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    correct += (model(imgs).argmax(1) == labels).sum().item(); total += labels.size(0)
            
            acc = correct / total
            if acc > best_acc:
                best_acc = acc; best_state = model.state_dict().copy(); patience = 0
            else: patience += 1
            if patience >= EARLY_STOPPING_PATIENCE: break
        
        # --- SWA Finalization ---
        if epoch >= swa_start:
            print(f"[*] Updating SWA Batch Normalization for AI {i+1}...")
            update_bn(train_loader, swa_model, device=device)
            swa_model.eval()
            sw_correct, sw_total = 0, 0
            with torch.inference_mode():
                for imgs, labels in val_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    sw_correct += (swa_model(imgs).argmax(1) == labels).sum().item(); sw_total += labels.size(0)
            sw_acc = sw_correct / sw_total
            if sw_acc > best_acc:
                print(f"[*] SWA Model outperformed Best Epoch! ({sw_acc*100:.2f}% vs {best_acc*100:.2f}%)")
                best_acc = sw_acc; best_state = swa_model.module.state_dict().copy()
        
        # Final Victory Lap (100% Data)
        print(f"[*] Final Victory Lap for AI {i+1}...")
        model.load_state_dict(best_state)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        for _ in range(5):
            model.train()
            for imgs, labels in full_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad(); criterion(model(imgs), labels).backward(); optimizer.step()
        
        torch.save(model.state_dict(), model_path)
        print(f"[SUCCESS] AI {i+1} SAVED (Val Acc: {best_acc*100:.2f}%)")

if __name__ == "__main__":
    train_model()
    
    # --- AUTO-EVOLUTION LOOP ---
    def run_next(name):
        print(f"\n🚀 [AUTO] STARTING: {name}")
        import subprocess
        subprocess.run([sys.executable, name])

    run_next("TestModel.py")
    if os.path.exists("uncertain_predictions.csv"):
        run_next("ActiveLearning.py")
    run_next("GenerateMetrics.py")
    print("\n✅ ALL SYSTEMS READY. Launching Dashboard...")
    run_next("app.py")