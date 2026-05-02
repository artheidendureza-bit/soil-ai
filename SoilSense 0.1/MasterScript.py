import subprocess
import sys
import time
import os

print("=" * 60)
print("MICROBIOME AI PIPELINE")
print("=" * 60)

start_time = time.time()

print("\n[1/4] Running data augmentation...")
print("-" * 40)

if not os.path.exists("SetupMicrobiomeData.py"):
    print("ERROR: SetupMicrobiomeData.py not found!")
    sys.exit(1)

step_start = time.time()
result = subprocess.run([sys.executable, "SetupMicrobiomeData.py"])

if result.returncode != 0:
    print(f"ERROR: Augmentation failed with code {result.returncode}")
    sys.exit(1)

print(f"Completed in {(time.time() - step_start) / 60:.1f} minutes")

print("\n[2/4] Running Pseudo-Labeling (if unlabeled data exists)...")
print("-" * 40)

if os.path.exists("PseudoLabel.py"):
    step_start = time.time()
    result = subprocess.run([sys.executable, "PseudoLabel.py"])
    if result.returncode != 0:
        print(f"WARNING: Pseudo-Labeling had an issue (Code {result.returncode})")
    else:
        print(f"Completed in {(time.time() - step_start) / 60:.1f} minutes")

print("\n[3/4] Training AI...")
print("-" * 40)

if not os.path.exists("MainMicrobiome.py"):
    print("ERROR: MainMicrobiome.py not found!")
    sys.exit(1)

step_start = time.time()
result = subprocess.run([sys.executable, "MainMicrobiome.py"])

if result.returncode != 0:
    print(f"ERROR: Training failed with code {result.returncode}")
    sys.exit(1)

print(f"Completed in {(time.time() - step_start) / 60:.1f} minutes")

print("\n[4/4] Running Active Learning Remedial Pipeline...")
print("-" * 40)

if os.path.exists("wrong_predictions.csv") or os.path.exists("uncertain_predictions.csv"):
    step_start = time.time()
    result = subprocess.run([sys.executable, "ActiveLearning.py"])
    
    if result.returncode != 0:
        print(f"WARNING: Active Learning had an issue (Code {result.returncode})")
    else:
        print(f"Completed in {(time.time() - step_start) / 60:.1f} minutes")
else:
    print("Skipping Active Learning: No hard examples found from a previous test run.")

print("\n" + "=" * 60)
print(f"PIPELINE COMPLETE")
print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
print("=" * 60)