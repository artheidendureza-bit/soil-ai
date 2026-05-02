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
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, SubsetRandomSampler
from PIL import Image, ImageFile
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import KFold
import numpy as np

ImageFile.LOAD_TRUNCATED_IMAGES = True
AI_COUNT = 5
MODEL_FILE_TEMPLATE = "microbiome_{}.pth"
CLASS_NAMES = ["DIRT", "GRASS", "LEAF", "MIX"]
VAL_FRACTION = 0.15
EARLY_STOPPING_PATIENCE = 3
K_FOLD_SPLITS = 3
torch.backends.cudnn.benchmark = True

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
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
        
        # Initial convolution
        self.conv1 = nn.Conv2d(3, base_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.act = nn.SiLU()
        
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, 5)]
        
        # 5 Residual Blocks (Progressive widening, some dilated)
        self.layer1 = ResidualBlock(base_channels, base_channels * 2, stride=2, drop_path_prob=dpr[0])
        self.layer2 = ResidualBlock(base_channels * 2, base_channels * 4, stride=2, drop_path_prob=dpr[1])
        self.layer3 = ResidualBlock(base_channels * 4, base_channels * 8, stride=2, dilation=1, drop_path_prob=dpr[2])
        self.layer4 = ResidualBlock(base_channels * 8, base_channels * 16, stride=2, dilation=1, drop_path_prob=dpr[3])
        self.layer5 = ResidualBlock(base_channels * 16, base_channels * 16, stride=1, dilation=2, drop_path_prob=dpr[4])
        
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
        self.feature_maps = x  # Store for CAM
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_model_variant(index):
    if index == 0:
        return MicrobiomeBrain(base_channels=24, dropout1=0.3, dropout2=0.2)
    if index == 1:
        return MicrobiomeBrain(base_channels=24, dropout1=0.25, dropout2=0.1)
    if index == 2:
        return MicrobiomeBrain(base_channels=28, dropout1=0.4, dropout2=0.3)
    if index == 3:
        return MicrobiomeBrain(base_channels=20, dropout1=0.2, dropout2=0.1)
    return MicrobiomeBrain(base_channels=26, dropout1=0.35, dropout2=0.25)


def get_train_transform(index):
    common_aug = [
        transforms.RandomSolarize(threshold=128, p=0.2),
        transforms.RandomEqualize(p=0.2),
        transforms.RandomPosterize(bits=4, p=0.2),
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.2),
        transforms.ColorJitter(hue=0.1)
    ]
    base_post = [
        transforms.Resize((32, 32)),
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
            transforms.RandomResizedCrop(32, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            *common_aug, *base_post
        ])
    return transforms.Compose([
        transforms.RandomPerspective(distortion_scale=0.2, p=0.5),
        *common_aug, *base_post
    ])


inference_transform = transforms.Compose([
    transforms.Resize((32, 32)),
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


def train_model():
    data_path = "data/train"
    if not os.path.exists(data_path):
        print(f"[*] ERROR: I can't find the folder: {os.path.abspath(data_path)}")
        print("Please make sure your 'data' folder is in the same place as this script!")
        return

    total_dataset = ImageFolder(data_path, transform=get_train_transform(0))
    total_images = len(total_dataset)
    if total_images == 0:
        print("[*] No training images found in data/train. Add images in subfolders: DIRT, GRASS, LEAF, MIX.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Starting ensemble training with {AI_COUNT} AIs on {device}... Images found: {total_images}")
    print(f"[*] Using {K_FOLD_SPLITS}-Fold Cross-Validation")

    for model_idx in range(AI_COUNT):
        torch.manual_seed(42 + model_idx)
        train_dataset = ImageFolder(data_path, transform=get_train_transform(model_idx))
        val_dataset = ImageFolder(data_path, transform=inference_transform)  # NO augmentations for validation
        if len(train_dataset) == 0:
            print("[*] No training images found in data/train. Add images in subfolders: DIRT, GRASS, LEAF, MIX.")
            return

        # K-Fold Cross-Validation
        kfold = KFold(n_splits=K_FOLD_SPLITS, shuffle=True, random_state=42 + model_idx)
        fold_accuracies = []
        best_val_acc_overall = 0.0
        best_epoch_overall = 0
        best_model_state_overall = None
        
        indices = list(range(len(train_dataset)))
        fold_idx = 0
        
        for train_indices, val_indices in kfold.split(indices):
            fold_idx += 1
            print(f"\n--- AI {model_idx+1} Fold {fold_idx}/{K_FOLD_SPLITS} ---")
            
            train_loader = DataLoader(train_dataset, batch_size=32, sampler=SubsetRandomSampler(train_indices.tolist()), num_workers=1, pin_memory=True)
            val_loader = DataLoader(val_dataset, batch_size=32, sampler=SubsetRandomSampler(val_indices.tolist()), num_workers=1, pin_memory=True)

            model = get_model_variant(model_idx).to(device, memory_format=torch.channels_last)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
            optimizer = get_optimizer(model, model_idx)
            max_lrs = [group['lr'] * 5.0 for group in optimizer.param_groups]
            scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=max_lrs, steps_per_epoch=len(train_loader), epochs=20, pct_start=0.3)

            best_val_acc = 0.0
            best_epoch = 0
            epochs_no_improve = 0
            best_model_state = None

            for epoch in range(20):
                model.train()
                total_loss = 0.0
                loop = tqdm(train_loader, desc=f"AI {model_idx+1} F{fold_idx} - Epoch {epoch+1}/20", unit="batch")
                for images, labels in loop:
                    images, labels = images.to(device, memory_format=torch.channels_last), labels.to(device)
                    
                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step() # OneCycleLR is stepped per batch
                    total_loss += loss.item()
                    loop.set_postfix(loss=f"{loss.item():.4f}")

                avg_loss = total_loss / len(train_loader)

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
                    print(f"[*] AI {model_idx+1} F{fold_idx} epoch {epoch+1} - Loss: {avg_loss:.4f} | Val acc: {val_acc*100:.2f}%")

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    best_epoch = epoch + 1
                    epochs_no_improve = 0
                    best_model_state = model.state_dict().copy()
                else:
                    epochs_no_improve += 1
                    if epochs_no_improve >= EARLY_STOPPING_PATIENCE:
                        print(f"[*] Early stopping AI {model_idx+1} F{fold_idx} at epoch {epoch+1}.")
                        break
            
            fold_accuracies.append(best_val_acc)
            if best_val_acc > best_val_acc_overall:
                best_val_acc_overall = best_val_acc
                best_epoch_overall = best_epoch
                best_model_state_overall = best_model_state
            
            print(f"[*] Fold {fold_idx} best accuracy: {best_val_acc*100:.2f}%")
        
        # Save the best model found across all folds
        if best_model_state_overall is not None:
            best_model_path = get_model_path(model_idx)
            torch.save(best_model_state_overall, best_model_path)
            if model_idx == 0:
                torch.save(best_model_state_overall, "microbiome.pth")
            print(f"[*] Saved best AI {model_idx+1} model (acc: {best_val_acc_overall*100:.2f}%)")
        
        avg_fold_acc = np.mean(fold_accuracies)
        print(f"[*] AI {model_idx+1} K-Fold Average Accuracy: {avg_fold_acc*100:.2f}% (Best: {best_val_acc_overall*100:.2f}%)")

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


print("Microbiome AI Logic Engine: ONLINE")

if __name__ == "__main__":
    if not ensemble_files_exist() or not are_saved_models_compatible():
        print("Saved model files are missing or incompatible with the current architecture.")
        print("Starting training for 5 AIs...")
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
