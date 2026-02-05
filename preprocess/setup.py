import subprocess
import sys
import os

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PREPROCESS_DIR = os.path.join(BASE_DIR, "preprocess")
MODEL_DIR = os.path.join(BASE_DIR, "model")

def run(script_path):
    print(f"\nRunning {script_path}")
    subprocess.run(
        [sys.executable, script_path],
        check=True
    )
    print("Done")

if __name__ == "__main__":
    try:
        run(os.path.join(PREPROCESS_DIR, "filter_data.py"))
        run(os.path.join(PREPROCESS_DIR, "split_data.py"))
        run(os.path.join(PREPROCESS_DIR, "transform_data.py"))

        print("\nData preprocessing completed")

    except subprocess.CalledProcessError:
        print("\nSetup failed")
        sys.exit(1)
