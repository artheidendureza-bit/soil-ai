import os
import sys
import subprocess

try:
    import cv2
except ImportError:
    print("[*] Automatically installing missing module: opencv-python...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "opencv-python"])

try:
    import tqdm
except ImportError:
    print("[*] Automatically installing missing module: tqdm...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tqdm"])

import csv
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image, ImageEnhance, ImageOps
import os
import random
import numpy as np
import matplotlib.pyplot as plt

from MainMicrobiome import load_ensemble, CLASS_NAMES, inference_transform, generate_cam, get_confidence, AI_COUNT

transform = inference_transform

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
models = load_ensemble(device)
classes = CLASS_NAMES

CONFIDENCE_THRESHOLD = 0.5

print("--- Batch Test Results (with Confidence Thresholding & CAM) ---")


tta_transforms = [
    lambda img: img,
    lambda img: ImageOps.mirror(img),  
    lambda img: ImageOps.flip(img),  
    lambda img: img.rotate(10, resample=Image.BICUBIC),  
    lambda img: img.rotate(-10, resample=Image.BICUBIC),  
    lambda img: ImageEnhance.Brightness(img).enhance(0.9),  
    lambda img: ImageEnhance.Brightness(img).enhance(1.1),  
]

all_images = []
for root, dirs, files in os.walk("data/test"):
    if "NOT_SOIL" in root: continue # Skip sentinel data for main testing
    for file in files:
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            all_images.append((os.path.join(root, file), root.split(os.sep)[-1]))

import matplotlib.pyplot as plt

class_correct = {c: 0 for c in classes}
class_total = {c: 0 for c in classes}
confusion = {c: {d: 0 for d in classes} for c in classes}
wrong_predictions = []
uncertain_predictions = []
ensemble_disagreements = []

if len(all_images) == 0:
    raise RuntimeError("No test images found in data/test")

random.shuffle(all_images)
samples = all_images[:500]

correct = 0
for img_path, actual_label in samples:
    img = Image.open(img_path).convert("RGB")

    all_outputs = []
    all_confidences = []
    all_individual_predictions = []
    for tta_aug in tta_transforms:
        aug_img = tta_aug(img)
        tensor = transform(aug_img).unsqueeze(0).to(device, memory_format=torch.channels_last)
        with torch.inference_mode():
            outputs = [F.softmax(model(tensor), dim=1) for model in models[:7]]
            avg_probs = torch.mean(torch.stack(outputs), dim=0)
            all_outputs.append(avg_probs)
            all_confidences.append(torch.max(avg_probs).item())
            # Track individual model predictions for disagreement analysis
            individual_preds = [classes[torch.argmax(output, dim=1).item()] for output in outputs]
            all_individual_predictions.append(individual_preds)

    final_avg_probs = torch.mean(torch.stack(all_outputs), dim=0)
    final_confidence = torch.max(final_avg_probs).item()
    pred_idx = torch.argmax(final_avg_probs, dim=1).item()
    pred_label = classes[pred_idx]
    votes = [classes[torch.argmax(output, dim=1).item()] for output in all_outputs]

    # Calculate ensemble agreement
    all_model_votes = []
    for tta_preds in all_individual_predictions:
        all_model_votes.extend(tta_preds)
    unique_predictions = set(all_model_votes)
    disagreement_score = len(unique_predictions) / AI_COUNT if len(unique_predictions) > 1 else 0

    class_total[actual_label] += 1
    
    # Apply confidence threshold
    is_uncertain = final_confidence < CONFIDENCE_THRESHOLD
    display_label = "UNCERTAIN" if is_uncertain else pred_label
    
    confusion[actual_label][pred_label] += 1
    
    if actual_label == pred_label and not is_uncertain:
        correct += 1
        class_correct[actual_label] += 1
    elif is_uncertain:
        uncertain_predictions.append((img_path, actual_label, pred_label, final_confidence, votes))
    else:
        wrong_predictions.append((img_path, actual_label, pred_label, votes))
    
    # Track disagreement
    if disagreement_score > 0:
        ensemble_disagreements.append((img_path, actual_label, pred_label, disagreement_score, unique_predictions))
    
    confidence_pct = final_confidence * 100
    disagreement_indicator = " ⚠️ ENSEMBLE DISAGREEMENT" if disagreement_score > 0 else ""
    print(f"Actual: {actual_label:6} | Predicted: {display_label:9} | Confidence: {confidence_pct:5.1f}% | Votes: {votes}{disagreement_indicator}")
    
total = len(samples)
print(f"\nTotal Accuracy: {correct}/{total}")
print(f"Accuracy: {(correct/total)*100:.1f}%")
print(f"Uncertain predictions (confidence < {CONFIDENCE_THRESHOLD*100:.0f}%): {len(uncertain_predictions)}")
print(f"Wrong predictions: {len(wrong_predictions)}")
print(f"Ensemble disagreements (models voting differently): {len(ensemble_disagreements)}\n")

wrong_file = "wrong_predictions.csv"
with open(wrong_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "actual", "predicted", "votes"])
    for row in wrong_predictions:
        writer.writerow(row)
print(f"Saved {len(wrong_predictions)} wrong predictions to {wrong_file}")

uncertain_file = "uncertain_predictions.csv"
with open(uncertain_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "actual", "predicted", "confidence", "votes"])
    for row in uncertain_predictions:
        writer.writerow(row)
print(f"Saved {len(uncertain_predictions)} uncertain predictions to {uncertain_file}")

disagreement_file = "ensemble_disagreements.csv"
with open(disagreement_file, "w", newline="", encoding="utf-8") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(["image_path", "actual", "predicted", "disagreement_score", "model_disagreements"])
    for row in ensemble_disagreements:
        writer.writerow(row)
print(f"Saved {len(ensemble_disagreements)} ensemble disagreements to {disagreement_file}")

print("\nConfusion matrix:")
header = ["actual->pred"] + classes
print("\t".join(header))
for actual in classes:
    row = [actual] + [str(confusion[actual][pred]) for pred in classes]
    print("\t".join(row))

print("\nGenerating Accuracy Graph...")
categories = classes
accuracy_counts = []

for c in categories:
    if class_total[c] > 0:
        acc = (class_correct[c] / class_total[c]) * 100
    else:
        acc = 0
    accuracy_counts.append(acc)

plt.figure(figsize=(10, 8))
bars = plt.bar(categories, accuracy_counts, color=['#654321', '#7CFC00', '#228B22', '#DAA520'])

plt.title(f"Model Accuracy by Class (Total: {correct}/{total})")
plt.xlabel("Microbe Type")
plt.ylabel("Accuracy (%)")
plt.ylim(0, 100)

for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/3., height,
             f'{int(height)}%',
             ha='center', va='bottom')

plt.show()


# Generate CAM visualizations for sample predictions
print("\n🎨 Generating Class Activation Maps (CAM) for interpretability...")

def display_cam_for_image(img_path, model_idx=0):
    """Display original image with CAM overlay"""
    img = Image.open(img_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device, memory_format=torch.channels_last)
    
    model = models[model_idx]
    
    # Get predictions from all models for ensemble voting display
    with torch.inference_mode():
        all_model_preds = []
        for m in models:
            output = F.softmax(m(img_tensor), dim=1)
            pred_idx = torch.argmax(output, dim=1).item()
            all_model_preds.append(CLASS_NAMES[pred_idx])
    
    # Generate CAM
    cam, class_idx = generate_cam(model, img_tensor)
    
    if cam is None:
        print(f"Could not generate CAM for {img_path}")
        return
    
    # Resize CAM to match original image size
    cam_resized = cv2.resize(cam, (img.size[0], img.size[1]))
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Original image
    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    # CAM heatmap
    im = axes[1].imshow(cam_resized, cmap='jet')
    axes[1].set_title(f"Class Activation Map (Model {model_idx+1} Pred: {CLASS_NAMES[class_idx]})")
    axes[1].axis("off")
    plt.colorbar(im, ax=axes[1])
    
    # Overlay CAM on original
    img_np = np.array(img) / 255.0
    axes[2].imshow(img_np)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.4)
    ensemble_text = f"Ensemble Votes: {all_model_preds}"
    axes[2].set_title(f"CAM Overlay\n{ensemble_text}")
    axes[2].axis("off")
    
    plt.tight_layout()
    # Save with unique name to avoid overwriting
    save_path = f"cam_sample_{os.path.basename(img_path)}.png"
    plt.savefig(save_path, dpi=120)
    plt.close()
    print(f"✅ Saved CAM visualization: {save_path}")


# Display CAM for a few sample images (including disagreement cases)
print("\nDisplaying CAM visualizations...\n")

cam_samples = []
# Prioritize disagreement cases
if ensemble_disagreements:
    cam_samples.extend(ensemble_disagreements[:2])
# Add uncertain predictions
if uncertain_predictions:
    cam_samples.extend(uncertain_predictions[:1])
# Fill remaining slots with random samples
if len(cam_samples) < 3 and len(samples) > 0:
    remaining = random.sample(range(len(samples)), min(3 - len(cam_samples), len(samples)))
    cam_samples.extend([samples[i] for i in remaining])

for i, sample_info in enumerate(cam_samples):
    img_path = sample_info[0]
    print(f"\nGenerating CAM {i+1}/{len(cam_samples)} for: {img_path}")
    try:
        display_cam_for_image(img_path, model_idx=0)
    except Exception as e:
        print(f"Error generating CAM: {e}")

if len(cam_samples) == 0:
    print("No samples available for CAM visualization.")

print("\n✅ Testing complete.")