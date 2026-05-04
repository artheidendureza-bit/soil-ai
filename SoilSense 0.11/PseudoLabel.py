import os
import shutil
import torch
from PIL import Image
from tqdm import tqdm

from MainMicrobiome import (
    AI_COUNT, CLASS_NAMES, load_ensemble, ensemble_predict, inference_transform
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
UNLABELED_DIR = "unlabeled_data"
TRAIN_DIR = "data/train"
REVIEW_DIR = "data/review"
CONFIDENCE_THRESHOLD = 0.95  
REVIEW_THRESHOLD = 0.70  

def pseudo_label_data():
    print("=" * 60)
    print("🧠 INITIATING PSEUDO-LABELING PIPELINE 🧠")
    print("=" * 60)
    
    if not os.path.exists(UNLABELED_DIR):
        print(f"[*] Creating '{UNLABELED_DIR}' directory.")
        os.makedirs(UNLABELED_DIR)
        print("[*] Add raw, unclassified microscope images here to let the AI teach itself!")
        return

    unlabeled_images = []
    for f in os.listdir(UNLABELED_DIR):
        if f.lower().endswith(('.png', '.jpg', '.jpeg')):
            unlabeled_images.append(os.path.join(UNLABELED_DIR, f))

    if not unlabeled_images:
        print(f"[*] No images found in '{UNLABELED_DIR}'. Add some to use Pseudo-Labeling.")
        return

    print(f"[*] Found {len(unlabeled_images)} unlabeled images. Waking up the {AI_COUNT}-AI Ensemble...")
    models = load_ensemble(DEVICE)
    if not models:
        print("[!] Ensemble not fully trained. Run MasterScript.py first.")
        return

    added_count = {c: 0 for c in CLASS_NAMES}

    for img_path in tqdm(unlabeled_images, desc="Pseudo-Labeling", unit="img"):
        try:
            img = Image.open(img_path).convert("RGB")
            tensor = inference_transform(img).unsqueeze(0).to(DEVICE, memory_format=torch.channels_last)
            
            # Predict
            avg_probs = ensemble_predict(models, tensor).squeeze().cpu().numpy()
            max_prob = float(avg_probs.max())
            max_idx = int(avg_probs.argmax())
            predicted_class = CLASS_NAMES[max_idx]

            # If the ensemble is highly confident, label it and move it to the train set!
            if max_prob >= CONFIDENCE_THRESHOLD:
                dest_dir = os.path.join(TRAIN_DIR, predicted_class)
                os.makedirs(dest_dir, exist_ok=True)
                
                filename = os.path.basename(img_path)
                dest_path = os.path.join(dest_dir, f"pseudo_{filename}")
                
                # Move the file
                shutil.move(img_path, dest_path)
                added_count[predicted_class] += 1
            
            # If the ensemble is unsure but has a hunch, move it to review folder
            elif max_prob >= REVIEW_THRESHOLD:
                review_dest = os.path.join(REVIEW_DIR, predicted_class)
                os.makedirs(review_dest, exist_ok=True)
                filename = os.path.basename(img_path)
                shutil.move(img_path, os.path.join(review_dest, f"review_{filename}"))
                print(f"\n[*] Image {filename} moved to review ({predicted_class} - {max_prob*100:.1f}%)")
                
        except Exception as e:
            print(f"[-] Error processing {img_path}: {e}")

    total_added = sum(added_count.values())
    print("\n" + "=" * 60)
    if total_added > 0:
        print(f"✅ SUCCESSFULLY PSEUDO-LABELED {total_added} IMAGES!")
        for c, count in added_count.items():
            if count > 0:
                print(f"   - Added {count} to {c}")
        print("\n[*] The AI has successfully taught itself from new data.")
        print("[*] Re-run MasterScript.py to train the ensemble on this new knowledge!")
    else:
        print("⚠️ No images met the 95% confidence threshold.")
        print("    The AI wasn't sure enough to label any of the provided images.")
    print("=" * 60)

if __name__ == "__main__":
    pseudo_label_data()
