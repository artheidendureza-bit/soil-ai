import os
import sys
import subprocess

try:
    import sklearn
except ImportError:
    print("[*] Automatically installing missing module: scikit-learn...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "scikit-learn"])

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.swa_utils import AveragedModel, SWALR
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
AI_COUNT = 7
MODEL_FILE_TEMPLATE = "microbiome_{}.pth"
CLASS_NAMES = ["DIRT", "GRASS", "LEAF", "MIX"]
VAL_FRACTION = 0.10
EARLY_STOPPING_PATIENCE = 10
K_FOLD_SPLITS = 5
torch.backends.cudnn.benchmark = True

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.SiLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)
    def forward(self, x):
        x = x * self.ca(x)
        x = x * self.sa(x)
        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, 
                                   stride=stride, padding=dilation, dilation=dilation, 
                                   groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pointwise(self.depthwise(x))

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dilation=1, drop_path_prob=0.0):
        super().__init__()
        self.conv1 = DepthwiseSeparableConv(in_channels, out_channels, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.SiLU()
        self.conv2 = DepthwiseSeparableConv(out_channels, out_channels, stride=1, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.cbam = CBAM(out_channels, ratio=max(2, out_channels // 8))
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0. else nn.Identity()
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out = self.shortcut(x) + self.drop_path(out)
        out = nn.SiLU()(out)
        return out

class MicrobiomeBrain(nn.Module):
    def __init__(self, num_classes=4, base_channels=32, dropout1=0.3, dropout2=0.2, drop_path_rate=0.2):
        super().__init__()
        self.num_classes = num_classes
        self.base_channels = base_channels
        
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.act = nn.SiLU()
        
        self.stage_blocks = [2, 2, 3, 2, 1]
        total_blocks = sum(self.stage_blocks)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, total_blocks)]
        
        self.layer1 = self._make_layer(base_channels, base_channels * 2, self.stage_blocks[0], stride=2, dpr_list=dpr[0:2])
        self.layer2 = self._make_layer(base_channels * 2, base_channels * 4, self.stage_blocks[1], stride=2, dpr_list=dpr[2:4])
        self.layer3 = self._make_layer(base_channels * 4, base_channels * 8, self.stage_blocks[2], stride=2, dilation=1, dpr_list=dpr[4:7])
        self.layer4 = self._make_layer(base_channels * 8, base_channels * 16, self.stage_blocks[3], stride=2, dilation=1, dpr_list=dpr[7:9])
        self.layer5 = self._make_layer(base_channels * 16, base_channels * 16, self.stage_blocks[4], stride=1, dilation=2, dpr_list=dpr[9:10])
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout1),
            nn.Linear(base_channels * 16, base_channels * 8),
            nn.SiLU(),
            nn.Dropout(dropout2),
            nn.Linear(base_channels * 8, num_classes)
        )
        self.feature_maps = None
    
    def get_feature_maps(self, x):
        x = self.act(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        return x

    def forward(self, x):
        x = self.get_feature_maps(x)
        self.feature_maps = x  
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

    def _make_layer(self, in_channels, out_channels, blocks, stride, dilation=1, dpr_list=None):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride, dilation=dilation, drop_path_prob=dpr_list[0]))
        for i in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, stride=1, dilation=dilation, drop_path_prob=dpr_list[i]))
        return nn.Sequential(*layers)


def get_model_variant(index):
    if index == 0:
        return MicrobiomeBrain(base_channels=64, dropout1=0.3, dropout2=0.2)
    if index == 1:
        return MicrobiomeBrain(base_channels=64, dropout1=0.25, dropout2=0.1)
    if index == 2:
        return MicrobiomeBrain(base_channels=96, dropout1=0.4, dropout2=0.3)
    if index == 3:
        return MicrobiomeBrain(base_channels=48, dropout1=0.2, dropout2=0.1)
    if index == 4:
        return MicrobiomeBrain(base_channels=80, dropout1=0.35, dropout2=0.25)
    if index == 5:
        return MicrobiomeBrain(base_channels=112, dropout1=0.45, dropout2=0.3)
    return MicrobiomeBrain(base_channels=128, dropout1=0.5, dropout2=0.4)


def get_train_transform(index):
    common_aug = [
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomPosterize(bits=4, p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.ColorJitter(hue=0.1)
    ]
    base_post = [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.2, scale=(0.02, 0.1), ratio=(0.3, 3.3))
    ]
    if index == 0:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            *common_aug, *base_post
        ])
    if index == 1:
        return transforms.Compose([
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, hue=0.1),
            *common_aug[1:], *base_post
        ])
    if index == 2:
        return transforms.Compose([
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),
            transforms.RandomGrayscale(p=0.2),
            *common_aug, *base_post
        ])
    if index == 3:
        return transforms.Compose([
            transforms.RandomResizedCrop(128, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            *common_aug, *base_post
        ])
    if index == 4:
        return transforms.Compose([
            transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
            *common_aug, *base_post
        ])
    if index == 5:
        return transforms.Compose([
            transforms.RandomRotation(45),
            transforms.RandomAffine(degrees=0, translate=(0.2, 0.2)),
            *common_aug, *base_post
        ])
    return transforms.Compose([
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
        transforms.RandomHorizontalFlip(p=0.5),
        *common_aug, *base_post
    ])


def mixup_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    y_a, y_b = y, y[index]
    bbx1, bby1, bbx2, bby2 = rand_bbox(x.size(), lam)
    x[:, :, bbx1:bbx2, bby1:bby2] = x[index, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (x.size()[-1] * x.size()[-2]))
    return x, y_a, y_b, lam

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int64(W * cut_rat)
    cut_h = np.int64(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)
    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)
    return bbx1, bby1, bbx2, bby2

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

inference_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def get_model_path(index):
    return MODEL_FILE_TEMPLATE.format(index)


def get_train_val_indices(dataset_size, val_fraction=VAL_FRACTION, seed=None):
    indices = list(range(dataset_size))
    if seed is not None:
        random.seed(seed)
    random.shuffle(indices)
    split = int(dataset_size * (1 - val_fraction))
    train_indices = indices[:split]
    val_indices = indices[split:]
    return train_indices, val_indices


def ensemble_files_exist():
    return all(os.path.exists(get_model_path(i)) for i in range(AI_COUNT))


def are_saved_models_compatible():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i in range(AI_COUNT):
        model_path = get_model_path(i)
        if not os.path.exists(model_path):
            return False
        model = get_model_variant(i)
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception:
            return False
    return True


def get_optimizer(model, index):
    if index == 0:
        return torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    if index == 1:
        return torch.optim.SGD(model.parameters(), lr=3e-3, momentum=0.9, weight_decay=1e-4)
    if index == 2:
        return torch.optim.Adam(model.parameters(), lr=8e-4, weight_decay=1e-5)
    if index == 3:
        return torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-3)
    return torch.optim.SGD(model.parameters(), lr=2e-3, momentum=0.95, weight_decay=2e-4)


GLOBAL_IMAGE_CACHE = {}

class CachedImageDataset(torch.utils.data.Dataset):
    def __init__(self, samples, class_to_idx, transform=None):
        self.class_to_idx = class_to_idx
        self.transform = transform
        
        # Preload once and share across all AIs
        if not GLOBAL_IMAGE_CACHE:
            print(f"[*] Initializing Global RAM Cache ({len(samples)} images)...")
            for path, _ in tqdm(samples, desc="Preloading RAM"):
                if path not in GLOBAL_IMAGE_CACHE:
                    try:
                        img = Image.open(path).convert('RGB')
                        img = img.resize((128, 128), Image.Resampling.BILINEAR)
                        GLOBAL_IMAGE_CACHE[path] = img.copy()
                    except Exception:
                        if os.path.exists(path):
                            try: os.remove(path)
                            except: pass
        
        # Only keep samples that are actually in the cache (skips corrupted ones)
        self.samples = [(p, t) for p, t in samples if p in GLOBAL_IMAGE_CACHE]

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = GLOBAL_IMAGE_CACHE[path]
        if self.transform is not None:
            sample = self.transform(sample)
        return sample, target

    def __len__(self):
        return len(self.samples)

def train_model():
    data_path = "data/train"
    if not os.path.exists(data_path):
        print(f"[*] ERROR: I can't find the folder: {os.path.abspath(data_path)}")
        print("Please make sure your 'data' folder is in the same place as this script!")
        return

    total_dataset = ImageFolder(data_path, transform=get_train_transform(0))
    total_images = len(total_dataset)
    if total_images == 0:
        print("[*] No training images found in data/train.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting ensemble training with {AI_COUNT} AIs on {device}... Images found: {total_images}")
    print(f"[*] Using 80/20 Production Split (Fast Mode)")

    # Walk directory ONCE
    base_dataset = ImageFolder(data_path)

    for model_idx in range(AI_COUNT):
        torch.manual_seed(42 + model_idx)
        train_dataset = CachedImageDataset(base_dataset.samples, base_dataset.class_to_idx, transform=get_train_transform(model_idx))
        val_dataset = CachedImageDataset(base_dataset.samples, base_dataset.class_to_idx, transform=inference_transform)

        # Consistent 80/20 Split
        indices = list(range(len(train_dataset)))
        random.seed(42)
        random.shuffle(indices)
        split_idx = int(0.8 * len(indices))
        train_indices = indices[:split_idx]
        val_indices = indices[split_idx:]

        print(f"\n--- Training AI {model_idx+1} (80/20 Split) ---")
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=SubsetRandomSampler(train_indices), num_workers=0, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=32, sampler=SubsetRandomSampler(val_indices), num_workers=0, pin_memory=True)
        full_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)

        model = get_model_variant(model_idx).to(device, memory_format=torch.channels_last)
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = get_optimizer(model, model_idx)
        max_lrs = [group['lr'] * 5.0 for group in optimizer.param_groups]
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, steps_per_epoch=len(train_loader), epochs=50, pct_start=0.3)

        swa_model = AveragedModel(model)
        swa_start = int(50 * 0.75)
        swa_scheduler = SWALR(optimizer, swa_lr=0.001)

        best_val_acc = -1.0
        best_epoch = 0
        epochs_no_improve = 0
        best_model_state = None
        best_val_acc_overall = 0.0
        best_model_state_overall = None

        for epoch in range(50):
            model.train()
            total_loss = 0.0
            loop = tqdm(train_loader, desc=f"AI {model_idx+1} - Epoch {epoch+1}/50", unit="batch")
            for images, labels in loop:
                images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                if epoch < swa_start:
                    scheduler.step()
                total_loss += loss.item()
                loop.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)

            if epoch >= swa_start:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            model.eval()
            val_correct = 0
            val_total = 0
            with torch.inference_mode():
                for images, labels in val_loader:
                    images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
                    outputs = model(images)
                    preds = torch.argmax(outputs, dim=1)
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)
            val_acc = val_correct / val_total if val_total > 0 else 0.0

            if (epoch + 1) % 5 == 0:
                print(f"[*] AI {model_idx+1} epoch {epoch+1} - Loss: {avg_loss:.4f} | Val acc: {val_acc*100:.2f}%")

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                epochs_no_improve = 0
                best_model_state = model.state_dict().copy()
            else:
                epochs_no_improve += 1
                if epoch > 5 and epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                    print(f"[*] Early stopping AI {model_idx+1} at epoch {epoch+1} (Patience {epochs_no_improve}/{EARLY_STOPPING_PATIENCE})")
                    break
            
        print(f"[*] Updating SWA Batch Normalization for AI {model_idx+1}...")
        torch.optim.swa_utils.update_bn(train_loader, swa_model, device=device)
        swa_model.eval()
        val_correct = 0
        val_total = 0
        with torch.inference_mode():
            for images, labels in val_loader:
                images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
                outputs = swa_model(images)
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        
        if best_val_acc > best_val_acc_overall:
            best_val_acc_overall = best_val_acc
            best_model_state_overall = best_model_state
        
        print(f"[*] Best validation accuracy: {best_val_acc_overall*100:.2f}%")
        
        # --- PHASE 2: Victory Lap on Full Dataset (100%) ---
        if best_model_state_overall is not None:
            print(f"[*] Performing 5-epoch Victory Lap on 100% dataset for AI {model_idx+1}...")
            model.load_state_dict(best_model_state_overall)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) 
            
            for v_epoch in range(5):
                model.train()
                for images, labels in full_loader:
                    images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(images), labels)
                    loss.backward()
                    optimizer.step()
            
            # Save the final model
            best_model_path = get_model_path(model_idx)
            torch.save(model.state_dict(), best_model_path)
            if model_idx == 0:
                torch.save(model.state_dict(), "microbiome.pth")
            print(f"[*] Saved Final AI {model_idx+1} model (Acc: {best_val_acc_overall*100:.2f}%)")

    print("\n[*] Ensemble training complete. All models saved.")


def load_ensemble(device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    models = []
    for i in range(AI_COUNT):
        model_path = get_model_path(i)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Missing model file: {model_path}")
        model = get_model_variant(i)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device, memory_format=torch.channels_last)
        model.eval()
        models.append(model)
    return models

def ensemble_predict(models, tensor):
    outputs = []
    with torch.inference_mode():
        for model in models:
            outputs.append(F.softmax(model(tensor), dim=1))
    return torch.mean(torch.stack(outputs), dim=0)


def generate_cam(model, input_tensor, class_idx=None):
    """Generate Class Activation Map (CAM) for interpretability"""
    model.eval()
    fmap = model.get_feature_maps(input_tensor)  # [B, C, H, W]
    
    # Get weights from classifier
    classifier = model.classifier
    linear_layer = None
    for layer in classifier:
        if isinstance(layer, nn.Linear):
            linear_layer = layer
            break
    
    if linear_layer is None:
        return None
    
    weights = linear_layer.weight  # [num_classes, num_features]
    bias = linear_layer.bias
    
    if class_idx is None:
        output = model(input_tensor)
        class_idx = output.argmax(dim=1).item()
    
    cam = (weights[class_idx].unsqueeze(-1).unsqueeze(-1) * fmap).sum(dim=1)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-5)
    
    return cam.squeeze().cpu().detach().numpy(), class_idx


def get_confidence(model, input_tensor):
    """Get prediction confidence score"""
    model.eval()
    with torch.inference_mode():
        output = F.softmax(model(input_tensor), dim=1)
        confidence, pred_idx = torch.max(output, dim=1)
    return confidence.item(), pred_idx.item()


def classify_sample(image_path):
    if not os.path.exists(image_path):
        print("Image not found")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models = load_ensemble(device)
    img = Image.open(image_path).convert("RGB")
    tensor = inference_transform(img).unsqueeze(0).to(device, memory_format=torch.channels_last)

    predictions = []
    with torch.inference_mode():
        for model in models:
            output = model(tensor)
            label = CLASS_NAMES[torch.argmax(output, dim=1).item()]
            predictions.append(label)

    avg_probs = ensemble_predict(models, tensor).squeeze().cpu().numpy()
    final_prediction = CLASS_NAMES[int(avg_probs.argmax())]

    print(f"AI votes: {predictions}")
    print(f"Final consensus: {final_prediction}")

    plt.bar(CLASS_NAMES, avg_probs)
    plt.title("Ensemble AI Confidence")
    plt.ylabel("Confidence")
    plt.ylim(0, 1)
    plt.show()

if __name__ == "__main__":
    if not ensemble_files_exist() or not are_saved_models_compatible():
        print("Saved model files are missing or incompatible with the current architecture.")
        print(f"Starting training for {AI_COUNT} AIs...")
        for i in range(AI_COUNT):
            model_path = get_model_path(i)
            if os.path.exists(model_path):
                os.remove(model_path)
        if os.path.exists("microbiome.pth"):
            os.remove("microbiome.pth")
        train_model()
    else:
        print(f"Found {AI_COUNT} trained AIs. Ready for operations.")

    print("\n--- Testing on a Random Sample ---")
    all_images = []
    for root, dirs, files in os.walk("data/train"):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                all_images.append(os.path.join(root, file))

    import random
    if all_images:
        random_image = random.choice(all_images)
        print(f"Selected image: {random_image}")
        classify_sample(random_image)
    else:
        print("No images found to test!")