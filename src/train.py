"""
Model training pipeline for static PE malware classification on EMBER 2018 v2 features.
Trains Random Forest, Logistic Regression, and LightGBM models, evaluates performance,
and saves models and metrics to C:\\MaliciousFileClassification\\models.
"""
import os
import sys
import json
import time
import joblib
import numpy as np

# Ensure project root in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import ember
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")


def train_and_evaluate(dataset_dir=DATASET_DIR, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    print("=" * 70)
    print("     STATIC MALWARE CLASSIFICATION - MODEL TRAINING PIPELINE")
    print("=" * 70)

    # 1. Load vectorized features
    print("\n[STEP 1/5] Loading memory-mapped vectorized features...")
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_dir)
    print(f"    - X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"    - y_train shape: {y_train.shape}, labels: {np.unique(y_train, return_counts=True)}")
    print(f"    - X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
    print(f"    - y_test shape:  {y_test.shape}, labels: {np.unique(y_test, return_counts=True)}")

    n_train = len(y_train)
    n_test = len(y_test)
    n_features = X_train.shape[1]

    # 2. Fit Preprocessor (StandardScaler) for linear models
    print("\n[STEP 2/5] Fitting preprocessor (StandardScaler)...")
    preprocessor = StandardScaler()
    # Fit only on training data to prevent data leakage!
    X_train_scaled = preprocessor.fit_transform(X_train)
    X_test_scaled = preprocessor.transform(X_test)

    preprocessor_path = os.path.join(models_dir, "preprocessor.pkl")
    joblib.dump(preprocessor, preprocessor_path)
    print(f"[+] Preprocessor saved to: {preprocessor_path}")

    # 3. Model Definitions
    models = {
        "random_forest": {
            "name": "Random Forest",
            "model": RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1),
            "use_scaled": False,
            "filename": "random_forest.pkl"
        },
        "logistic_regression": {
            "name": "Logistic Regression",
            "model": LogisticRegression(max_iter=300, C=1.0, random_state=42, solver='lbfgs'),
            "use_scaled": True,
            "filename": "logistic_regression.pkl"
        },
        "lightgbm": {
            "name": "LightGBM",
            "model": lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=63,
                                       random_state=42, n_jobs=-1, verbose=-1),
            "use_scaled": False,
            "filename": "lightgbm.pkl"
        }
    }

    # 4. Training and Evaluation Loop
    print("\n[STEP 3/5] Training and evaluating models...")
    results = {}
    best_model_key = None
    best_f1 = -1.0

    for key, item in models.items():
        name = item["name"]
        clf = item["model"]
        use_scaled = item["use_scaled"]
        filename = item["filename"]

        print(f"\n--- Training {name} ---")
        X_tr = X_train_scaled if use_scaled else X_train
        X_te = X_test_scaled if use_scaled else X_test

        t0 = time.time()
        clf.fit(X_tr, y_train)
        train_time = time.time() - t0

        t1 = time.time()
        preds = clf.predict(X_te)
        infer_time = time.time() - t1

        probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else None

        acc = float(accuracy_score(y_test, preds))
        prec = float(precision_score(y_test, preds, pos_label=1))
        rec = float(recall_score(y_test, preds, pos_label=1))
        f1 = float(f1_score(y_test, preds, pos_label=1))
        auc = float(roc_auc_score(y_test, probs)) if probs is not None else 0.0
        cm = confusion_matrix(y_test, preds).tolist()

        tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0
        detection_rate = rec

        print(f"    Training time:   {train_time:.2f}s")
        print(f"    Inference time:  {infer_time:.2f}s")
        print(f"    Accuracy:        {acc:.4f}")
        print(f"    Precision (mal): {prec:.4f}")
        print(f"    Recall (mal):    {rec:.4f}  (Critical for catching malware)")
        print(f"    Detection Rate:  {detection_rate:.4f}")
        print(f"    F1-Score (mal):  {f1:.4f}")
        print(f"    ROC-AUC:         {auc:.4f}")
        print(f"    False Pos Rate:  {fpr:.4f}")
        print(f"    False Neg Rate:  {fnr:.4f}")
        print(f"    Confusion Matrix: TN={tn}, FP={fp}, FN={fn}, TP={tp}")

        # Save individual model
        save_path = os.path.join(models_dir, filename)
        joblib.dump(clf, save_path)
        print(f"    Saved model -> {save_path}")

        results[key] = {
            "name": name,
            "filename": filename,
            "use_scaled": use_scaled,
            "train_time_sec": round(train_time, 2),
            "infer_time_sec": round(infer_time, 2),
            "accuracy": round(acc, 4),
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "detection_rate": round(detection_rate, 4),
            "f1_score": round(f1, 4),
            "roc_auc": round(auc, 4),
            "false_positive_rate": round(fpr, 4),
            "false_negative_rate": round(fnr, 4),
            "confusion_matrix": {
                "TN": int(tn),
                "FP": int(fp),
                "FN": int(fn),
                "TP": int(tp)
            }
        }

        # Model selection: in cybersecurity, high F1 and Recall for malicious class is paramount
        # to minimize catastrophic False Negatives (uncaught malware) while maintaining high precision.
        if f1 > best_f1:
            best_f1 = f1
            best_model_key = key

    # 5. Save Best Model
    print(f"\n[STEP 4/5] Selecting best model based on Malicious F1-Score (and Recall)...")
    print(f"    Best performer: {models[best_model_key]['name']} (F1 = {best_f1:.4f})")
    best_model_obj = models[best_model_key]["model"]
    best_model_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_model_obj, best_model_path)
    print(f"[+] Saved best model to: {best_model_path}")

    # 6. Save comprehensive summary metrics
    summary = {
        "dataset": {
            "train_samples": int(n_train),
            "test_samples": int(n_test),
            "features": int(n_features),
            "class_distribution_train": {"benign": int(np.sum(y_train == 0)), "malicious": int(np.sum(y_train == 1))},
            "class_distribution_test": {"benign": int(np.sum(y_test == 0)), "malicious": int(np.sum(y_test == 1))}
        },
        "selection_metric": "F1-Score and Recall on malicious class (minimizing False Negatives while controlling False Positives)",
        "best_model": {
            "key": best_model_key,
            "name": models[best_model_key]["name"],
            "filename": "best_model.pkl",
            "use_scaled": models[best_model_key]["use_scaled"]
        },
        "models": results
    }

    metrics_path = os.path.join(models_dir, "model_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"[+] Saved metrics report to: {metrics_path}")

    print("\n" + "=" * 70)
    print("                 MODEL TRAINING COMPLETED")
    print("=" * 70)
    return summary


if __name__ == "__main__":
    train_and_evaluate()

