import os
import numpy as np
import pandas as pd # type: ignore
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.ensemble import RandomForestClassifier # type: ignore
import joblib # type: ignore

DATA_DIR = "data"
MODEL_PATH = "models/gesture_model.pkl"

def load_dataset():
    X, y = [], []
    if not os.path.exists(DATA_DIR):
        print(f"[ERROR] Directory '{DATA_DIR}' not found.")
        return np.array([]), np.array([])
    
    files = [f for f in os.listdir(DATA_DIR) if f.endswith(".csv")]
    print(f"[INFO] Found {len(files)} dataset files.")
    
    for file in files:
        label = file.replace(".csv", "")
        try:
            df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
            X.extend(df.values)
            y.extend([label] * len(df))
            print(f"  - Loaded: {label} ({len(df)} samples)")
        except Exception as e:
            print(f"  - [WARNING] Failed to load {file}: {e}")
            
    return np.array(X), np.array(y)

def run_training():
    print("="*40)
    print("       MODEL TRAINING PROCESS")
    print("="*40)
    
    # 1. Load Data
    print("[1/4] Loading Datasets...")
    X, y = load_dataset()
    
    if len(X) == 0:
        print("[ABORT] No data found. Training cancelled.")
        return

    # 2. Split Data
    print("[2/4] Splitting Data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 3. Training
    print("[3/4] Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=200, random_state=42)
    model.fit(X_train, y_train)

    # 4. Evaluation
    acc = model.score(X_test, y_test) * 100
    print(f"[4/4] Evaluation Complete.")
    print("-" * 40)
    print(f"ACCURACY: {acc:.2f}%")
    print("-" * 40)

    # Save
    os.makedirs("models", exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Model saved to: {MODEL_PATH}")

if __name__ == "__main__":
    run_training()