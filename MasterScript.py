import subprocess
import sys
import time
import os

print("=" * 60)
print("MICROBIOME AI PIPELINE")
print("=" * 60)

start_time = time.time()

print("\n[1/2] Running data augmentation...")
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

print("\n[2/2] Training AI...")
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

print("\n" + "=" * 60)
print(f"PIPELINE COMPLETE")
print(f"Total time: {(time.time() - start_time) / 60:.1f} minutes")
print("=" * 60)