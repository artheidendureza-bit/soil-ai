import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from PIL import Image, ImageOps

# --- SAM OPTIMIZER ---
class SAM(torch.optim.Optimizer):
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super(SAM, self).__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups

    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            for p in group["params"]:
                if p.grad is None: continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)
        if zero_grad: self.zero_grad()

    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or "old_p" not in self.state[p]: continue
                p.data = self.state[p]["old_p"]
        self.base_optimizer.step()
        if zero_grad: self.zero_grad()

    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device
        norm = torch.norm(torch.stack([
            ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(p=2).to(shared_device)
            for group in self.param_groups for p in group["params"] if p.grad is not None
        ]), p=2)
        return norm

    def step(self, closure=None): raise NotImplementedError("SAM requires 2 steps.")

# --- FOCAL LOSS ---
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean', smoothing=0.1):
        super().__init__()
        self.alpha, self.gamma, self.reduction, self.smoothing = alpha, gamma, reduction, smoothing

    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        with torch.no_grad():
            smoothed_labels = torch.full_like(inputs, self.smoothing / (num_classes - 1))
            smoothed_labels.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        ce_loss = F.binary_cross_entropy_with_logits(inputs, smoothed_labels, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == 'mean' else focal_loss.sum()

# --- STOCHASTIC DEPTH ---
class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x):
        if self.drop_prob == 0. or not self.training: return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_tensor = torch.floor(random_tensor)
        return x.div(keep_prob) * binary_tensor

# --- ATTENTION & BLOCKS ---
class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False), nn.SiLU(),
                               nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        return x * self.sigmoid(self.fc(self.avg_pool(x)) + self.fc(self.max_pool(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim*4), nn.SiLU(), nn.Linear(dim*4, dim))
    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2)
        x_flat = x_flat + self.attn(self.norm1(x_flat), self.norm1(x_flat), self.norm1(x_flat))[0]
        x_flat = x_flat + self.mlp(self.norm2(x_flat))
        return x_flat.transpose(1, 2).reshape(B, C, H, W)

class ResidualBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dpr=0.0):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.act = nn.SiLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.cbam = CBAM(out_c)
        self.dpr = dpr
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1, stride=stride, bias=False), nn.BatchNorm2d(out_c))
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.cbam(self.bn2(self.conv2(self.act(self.bn1(self.conv1(x))))))
        if self.training and self.dpr > 0 and random.random() <= self.dpr: return identity
        return self.act(identity + out)

# --- FUSION & BRAINS ---
class MultiScaleFusion(nn.Module):
    def __init__(self, in_c):
        super().__init__()
        self.pool2, self.pool4 = nn.AvgPool2d(2), nn.AvgPool2d(4)
        self.conv = nn.Conv2d(in_c * 3, in_c, 1)
    def forward(self, x):
        x2 = F.interpolate(self.pool2(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        x4 = F.interpolate(self.pool4(x), size=x.shape[2:], mode='bilinear', align_corners=False)
        return F.silu(self.conv(torch.cat([x, x2, x4], dim=1)))

class MicrobiomeBrain(nn.Module):
    def __init__(self, base_channels=32, d1=0.3, d2=0.2, dpr=0.2):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv2d(3, base_channels, 3, padding=1), nn.BatchNorm2d(base_channels), nn.SiLU())
        self.layer1 = ResidualBlock(base_channels, base_channels*2, stride=2, dpr=dpr*0.2)
        self.layer2 = ResidualBlock(base_channels*2, base_channels*4, stride=2, dpr=dpr*0.4)
        self.layer3 = ResidualBlock(base_channels*4, base_channels*8, stride=2, dpr=dpr*0.6)
        self.layer4 = ResidualBlock(base_channels*8, base_channels*16, stride=2, dpr=dpr)
        self.fusion = MultiScaleFusion(base_channels*16)
        self.transformer = TransformerBlock(base_channels*16) if dpr > 0.3 else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(nn.Dropout(d1), nn.Linear(base_channels*16, base_channels*8), nn.SiLU(), nn.Dropout(d2), nn.Linear(base_channels*8, 4))
    def forward(self, x):
        x = self.transformer(self.fusion(self.layer4(self.layer3(self.layer2(self.layer1(self.stem(x)))))))
        self.feature_maps = x
        return self.classifier(self.avgpool(x).view(x.size(0), -1))
    def get_feature_maps(self, x): self.forward(x); return self.feature_maps

class SentinelBrain(nn.Module):
    def __init__(self, base_channels=32):
        super().__init__()
        self.feat = nn.Sequential(nn.Conv2d(3, base_channels, 3, padding=1), nn.SiLU(), nn.MaxPool2d(2),
                                 nn.Conv2d(base_channels, base_channels*2, 3), nn.SiLU(), nn.AdaptiveAvgPool2d(1))
        self.classifier = nn.Linear(base_channels*2, 2)
    def forward(self, x): return self.classifier(self.feat(x).view(x.size(0), -1))

# --- HELPERS ---
def mixup_data(x, y, alpha=1.0):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    idx = torch.randperm(x.size(0)).to(x.device)
    return lam * x + (1 - lam) * x[idx, :], y, y[idx], lam

def cutmix_data(x, y, alpha=1.0):
    if alpha > 0: lam = np.random.beta(alpha, alpha)
    else: lam = 1
    idx = torch.randperm(x.size(0)).to(x.device)
    
    W, H = x.size(2), x.size(3)
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    x[:, :, bbx1:bbx2, bby1:bby2] = x[idx, :, bbx1:bbx2, bby1:bby2]
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (W * H))
    return x, y, y[idx], lam

def mixup_criterion(crit, pred, y_a, y_b, lam): return lam * crit(pred, y_a) + (1 - lam) * crit(pred, y_b)

def generate_cam(model, input_tensor, class_idx=None):
    model.eval()
    fmap = model.get_feature_maps(input_tensor)
    # Use the first linear layer weights which match the feature map channels
    weights = model.classifier[1].weight 
    if class_idx is None: class_idx = model(input_tensor).argmax(dim=1).item()
    # Since we have a hidden layer, we approximate by using the first layer's impact
    cam = F.relu((weights[0].unsqueeze(-1).unsqueeze(-1) * fmap).sum(dim=1))
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-5), class_idx

def dream_feature(model, class_idx):
    model.eval(); img = torch.randn((1,3,128,128), requires_grad=True, device=next(model.parameters()).device)
    opt = torch.optim.Adam([img], lr=0.1)
    for _ in range(30): opt.zero_grad(); (-model(img)[0, class_idx]).backward(); opt.step(); img.data.clamp_(0,1)
    return (img.detach().cpu().squeeze().permute(1,2,0).numpy()*255).astype(np.uint8)

def sentinel_check(model, tensor):
    with torch.inference_mode(): out = F.softmax(model(tensor), dim=1); return torch.argmax(out, dim=1).item() == 0, out[0, 0].item()

def get_model_variant(index):
    # Balanced Championship Configs (High power, High stability)
    configs = [(32,0.3,0.2,0.2), (32,0.25,0.1,0.2), (48,0.4,0.3,0.3), (32,0.2,0.1,0.2), (48,0.35,0.25,0.3), (64,0.45,0.3,0.4), (64,0.5,0.4,0.5)]
    return MicrobiomeBrain(*configs[index]) if index < 7 else SentinelBrain()
