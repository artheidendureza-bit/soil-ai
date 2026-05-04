import subprocess
import sys
import time
import os

# --- PRE-FLIGHT CHECK ---
def check_dependencies():
    required = ["optuna", "gradio", "opencv-python", "tqdm", "scipy", "torchvision", "pyttsx3"]
    for lib in required:
        try:
            import importlib.util
            spec = importlib.util.find_spec(lib.split('-')[0] if '-' not in lib else lib.replace('-', '_'))
            if spec is None: raise ImportError
        except ImportError:
            print(f"[*] Installing missing library: {lib}")
            subprocess.check_call([sys.executable, "-m", "pip", "install", lib])

def run_script(script_name):
    if not os.path.exists(script_name):
        print(f"❌ ERROR: {script_name} not found!")
        sys.exit(1)
    
    print(f"\n🚀 [STEP] STARTING: {script_name}")
    print("-" * 50)
    start = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name])
        duration = (time.time() - start) / 60
        
        if result.returncode == 0:
            print(f"✅ SUCCESS: {script_name} finished in {duration:.1f} mins")
            return True
        else:
            print(f"❌ FATAL ERROR: {script_name} failed (Code {result.returncode})")
            sys.exit(1)
    except Exception as e:
        print(f"❌ CRITICAL ERROR running {script_name}: {e}")
        sys.exit(1)

def main():
    print("=" * 60)
    print("🏆 SOILSENSE ULTIMATE: FULL AUTOMATED PIPELINE 🏆")
    print("=" * 60)
    total_start = time.time()
    
    check_dependencies()
    
    run_script("SetupMicrobiomeData.py")
    run_script("PseudoLabel.py")
    run_script("AutoTune.py")
    run_script("MainMicrobiome.py")
    run_script("TestModel.py")
    
    if os.path.exists("uncertain_predictions.csv") or os.path.exists("data/train"):
        run_script("ActiveLearning.py")
    
    print("\n" + "=" * 60)
    print(f"✨ PIPELINE COMPLETE in {(time.time() - total_start) / 60:.1f} mins")
    print("🖥️  LAUNCHING DASHBOARD...")
    print("=" * 60)
    run_script("GenerateMetrics.py")

if __name__ == "__main__":
    main()