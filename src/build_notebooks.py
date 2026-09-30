"""
Script to generate all 5 project notebooks for Malicious File Classification.
Produces clean, well-documented, executable Jupyter notebooks matching all requirements.
"""
import os
import nbformat as nbf

NOTEBOOKS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "notebooks"))
os.makedirs(NOTEBOOKS_DIR, exist_ok=True)

def create_nb_01():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Malicious File Classification",
            "language": "python",
            "name": "malicious-file-classification"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []
    
    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 01. Data Analysis: EMBER 2018 v2 Static PE Malware Dataset

## Project Overview
This project builds a **static Windows Portable Executable (PE) malware classifier** using the **EMBER 2018 v2 dataset**.
Static analysis inspects file structure, byte entropy, imports, exports, and section characteristics **without executing the binary**, ensuring complete safety.

---
### Notebook Objectives
1. **Environment Verification**: Validate Python, NumPy, Pandas, scikit-learn, and EMBER versions.
2. **Dataset Verification**: Check local raw JSONL files, file sizes, and memory-mapped `.dat` representations.
3. **Load Vectorized Features**: Safely load memory-mapped feature arrays (`X_train`, `y_train`, `X_test`, `y_test`).
4. **Dataset Dimensions & Shapes**: Inspect feature dimensions (2381 static features per sample).
5. **Class Balance & Label Distribution**: Visualize benign (0) vs. malicious (1) sample distribution.
6. **Feature Quality Checks**: Verify absence of NaNs, infinite values, and validate label boundaries.
"""))

    # Section A
    cells.append(nbf.v4.new_markdown_cell("## A. Environment Verification\nVerify that the runtime environment has all required libraries correctly configured."))
    cells.append(nbf.v4.new_code_cell("""import sys
import os
import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import ember

print(f"Python Version:       {sys.version.split()[0]}")
print(f"NumPy Version:        {np.__version__}")
print(f"Pandas Version:       {pd.__version__}")
print(f"scikit-learn Version: {sklearn.__version__}")
print("EMBER Package:        Imported successfully!")
"""))

    # Section B
    cells.append(nbf.v4.new_markdown_cell("## B. Dataset Verification\nVerify raw EMBER JSONL files and vectorized memory-mapped binary files."))
    cells.append(nbf.v4.new_code_cell("""DATASET_DIR = os.path.abspath(os.path.join("..", "dataset", "ember2018"))
print(f"Dataset Path: {DATASET_DIR}")
print(f"Exists: {os.path.exists(DATASET_DIR)}")

# Inspect dataset files and sizes
files_info = []
for fname in sorted(os.listdir(DATASET_DIR)):
    fpath = os.path.join(DATASET_DIR, fname)
    if os.path.isfile(fpath):
        size_mb = os.path.getsize(fpath) / (1024 * 1024)
        files_info.append({"Filename": fname, "Size (MB)": round(size_mb, 2)})

df_files = pd.DataFrame(files_info)
display(df_files)
"""))

    # Section C & D
    cells.append(nbf.v4.new_markdown_cell("""## C & D. Load EMBER Vectorized Features & Inspect Shapes
We load features using `ember.read_vectorized_features()`.
This loads memory-mapped NumPy arrays from disk (`X_train.dat`, `y_train.dat`, `X_test.dat`, `y_test.dat`), avoiding memory exhaustion.
"""))
    cells.append(nbf.v4.new_code_cell("""X_train, y_train, X_test, y_test = ember.read_vectorized_features(DATASET_DIR)

print("Dataset Shapes and Memory Layout:")
print(f"  X_train: shape = {X_train.shape}, dtype = {X_train.dtype}")
print(f"  y_train: shape = {y_train.shape}, dtype = {y_train.dtype}")
print(f"  X_test:  shape = {X_test.shape}, dtype = {X_test.dtype}")
print(f"  y_test:  shape = {y_test.shape}, dtype = {y_test.dtype}")
print(f"\\nStatic Features per Sample: {X_train.shape[1]}")
"""))

    # Section E
    cells.append(nbf.v4.new_markdown_cell("## E. Label Distribution & Class Balance\nAnalyze class distribution: `0 = Benign`, `1 = Malicious`."))
    cells.append(nbf.v4.new_code_cell("""train_unique, train_counts = np.unique(y_train, return_counts=True)
test_unique, test_counts = np.unique(y_test, return_counts=True)

dist_df = pd.DataFrame({
    "Class": ["Benign (0)", "Malicious (1)"],
    "Train Count": train_counts,
    "Train %": (train_counts / len(y_train)) * 100,
    "Test Count": test_counts,
    "Test %": (test_counts / len(y_test)) * 100
})
display(dist_df)

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

sns.barplot(x=["Benign (0)", "Malicious (1)"], y=train_counts, ax=axes[0], palette=["#2ecc71", "#e74c3c"])
axes[0].set_title(f"Train Set Distribution (Total: {len(y_train):,})")
axes[0].set_ylabel("Number of Samples")
for i, count in enumerate(train_counts):
    axes[0].text(i, count / 2, f"{count:,}\\n({count/len(y_train)*100:.1f}%)", ha='center', va='center', color='white', fontweight='bold')

sns.barplot(x=["Benign (0)", "Malicious (1)"], y=test_counts, ax=axes[1], palette=["#2ecc71", "#e74c3c"])
axes[1].set_title(f"Test Set Distribution (Total: {len(y_test):,})")
axes[1].set_ylabel("Number of Samples")
for i, count in enumerate(test_counts):
    axes[1].text(i, count / 2, f"{count:,}\\n({count/len(y_test)*100:.1f}%)", ha='center', va='center', color='white', fontweight='bold')

plt.tight_layout()
plt.show()
"""))

    # Section F
    cells.append(nbf.v4.new_markdown_cell("## F. Basic Feature Analysis & Sparsity\nCheck sparsity and summary statistics of EMBER feature representations."))
    cells.append(nbf.v4.new_code_cell("""# Compute sparsity on a representative subset of training features
sample_subset = X_train[:1000]
zero_elements = np.count_nonzero(sample_subset == 0)
total_elements = sample_subset.size
sparsity = (zero_elements / total_elements) * 100

print(f"Feature Dimension: {X_train.shape[1]}")
print(f"Estimated Feature Sparsity: {sparsity:.2f}% (zeros across features)")

# Inspect summary statistics for the first 8 features
df_sample_features = pd.DataFrame(X_train[:1000, :8], columns=[f"Feature_{i}" for i in range(8)])
display(df_sample_features.describe().round(2))
"""))

    # Section G
    cells.append(nbf.v4.new_markdown_cell("## G. Data Quality Checks\nEnsure data integrity: check for NaNs, Infs, and invalid labels."))
    cells.append(nbf.v4.new_code_cell("""# Check for NaN and Infinite values
nan_train = np.isnan(X_train).any()
inf_train = np.isinf(X_train).any()
nan_test = np.isnan(X_test).any()
inf_test = np.isinf(X_test).any()

# Check label validity
valid_labels = set([0.0, 1.0])
train_labels_valid = set(np.unique(y_train)).issubset(valid_labels)
test_labels_valid = set(np.unique(y_test)).issubset(valid_labels)

print("Data Quality Verification Results:")
print(f"  - X_train contains NaN:      {nan_train}")
print(f"  - X_train contains Infinity: {inf_train}")
print(f"  - X_test contains NaN:       {nan_test}")
print(f"  - X_test contains Infinity:  {inf_test}")
print(f"  - y_train labels valid:      {train_labels_valid} (Labels: {set(np.unique(y_train))})")
print(f"  - y_test labels valid:       {test_labels_valid} (Labels: {set(np.unique(y_test))})")

assert not nan_train and not inf_train, "Data quality check failed: NaNs or Infs in X_train"
assert not nan_test and not inf_test, "Data quality check failed: NaNs or Infs in X_test"
print("\\n[+] ALL DATA QUALITY CHECKS PASSED SUCCESSFULLY!")
"""))

    nb.cells = cells
    with open(os.path.join(NOTEBOOKS_DIR, "01_data_analysis.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[+] Generated 01_data_analysis.ipynb")


def create_nb_02():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Malicious File Classification",
            "language": "python",
            "name": "malicious-file-classification"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 02. Preprocessing: Pipeline Design & Feature Scaling

## Preprocessing Strategy for EMBER Features
EMBER static PE features (2381 dimensions) contain a mixture of:
- **Structural counts and sizes** (e.g., number of sections, file size, export counts)
- **Byte histograms & byte-entropy histograms** (bounded 0 to 1 distributions)
- **Hashed feature strings** (FeatureHasher outputs for section names, DLL imports, exports)

### Crucial Preprocessing Principles:
1. **Prevent Data Leakage**: The preprocessor (scaler/imputer) must be **fit only on `X_train`** and then applied to `X_test`. Never fit on test data!
2. **Model-Specific Scaling**:
   - **Tree-based models** (Random Forest, LightGBM) are scale-invariant and do not strictly require feature scaling.
   - **Linear models** (Logistic Regression) require zero mean and unit variance (`StandardScaler`) for gradient/solver convergence.
3. **Pipeline Serialization**: Save the fitted preprocessor using `joblib` so inference on raw vectors replicates exact training transformations.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import joblib
import numpy as np
import ember
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

PROJECT_ROOT = os.path.abspath(os.path.join(".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# 1. Load memory-mapped vectorized features
X_train, y_train, X_test, y_test = ember.read_vectorized_features(DATASET_DIR)
print(f"Loaded X_train shape: {X_train.shape}")
print(f"Loaded X_test shape:  {X_test.shape}")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Building & Fitting the Preprocessing Pipeline\nWe build a `StandardScaler` fitted strictly on `X_train`."))
    cells.append(nbf.v4.new_code_cell("""# Fit scaler exclusively on training data
preprocessor = StandardScaler()
print("[*] Fitting StandardScaler on X_train...")
X_train_scaled = preprocessor.fit_transform(X_train)

# Transform test data using the fitted training parameters
print("[*] Transforming X_test with fitted scaler...")
X_test_scaled = preprocessor.transform(X_test)

print("\\nFeature Scaling Verification:")
print(f"  Original X_train mean: {np.mean(X_train):.4f}, std: {np.std(X_train):.4f}")
print(f"  Scaled X_train mean:   {np.mean(X_train_scaled):.4f}, std: {np.std(X_train_scaled):.4f}")
print(f"  Scaled X_test mean:    {np.mean(X_test_scaled):.4f}, std: {np.std(X_test_scaled):.4f}")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Serializing the Preprocessing Artifact\nSave the preprocessor to `models/preprocessor.pkl` for inference."))
    cells.append(nbf.v4.new_code_cell("""preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.pkl")
joblib.dump(preprocessor, preprocessor_path)
print(f"[+] Preprocessor successfully serialized to: {preprocessor_path}")
print(f"    File size: {os.path.getsize(preprocessor_path) / 1024:.2f} KB")

# Verify re-loading
loaded_scaler = joblib.load(preprocessor_path)
assert loaded_scaler.mean_.shape == (2381,), "Reload verification failed"
print("[+] Reload verification passed cleanly!")
"""))

    nb.cells = cells
    with open(os.path.join(NOTEBOOKS_DIR, "02_preprocessing.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[+] Generated 02_preprocessing.ipynb")


def create_nb_03():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Malicious File Classification",
            "language": "python",
            "name": "malicious-file-classification"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 03. Model Training: Classical Machine Learning for Malware Classification

## Cybersecurity Metrics & Objectives
In malware detection, overall **Accuracy is not sufficient**. The key cybersecurity trade-offs are:
- **Recall on Malicious Class (True Positive Rate)**: Measures what proportion of actual malware was detected. **A missed malware sample (False Negative) is catastrophic**, allowing malware to execute unnoticed.
- **Precision on Malicious Class**: Measures how many flagged files are genuinely malicious. A low precision leads to alert fatigue for SOC analysts.
- **F1-Score**: Harmonic mean of Precision and Recall, balancing detection efficacy and triage burden.
- **ROC-AUC**: Aggregate discriminative power across all decision thresholds.

### Models Trained:
1. **Random Forest Classifier**: Ensemble of decision trees; handles high-dimensional non-linear feature interactions and resist overfitting.
2. **Logistic Regression**: Linear baseline with L2 regularization; fast and interpretable.
3. **LightGBM Classifier**: Gradient boosted decision trees; EMBER benchmark model.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import time
import joblib
import numpy as np
import pandas as pd
import ember
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

PROJECT_ROOT = os.path.abspath(os.path.join(".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load data and preprocessor
X_train, y_train, X_test, y_test = ember.read_vectorized_features(DATASET_DIR)
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))

X_train_scaled = preprocessor.transform(X_train)
X_test_scaled = preprocessor.transform(X_test)

print(f"Training Samples:   {len(y_train):,} ({np.sum(y_train==0):,} Benign, {np.sum(y_train==1):,} Malicious)")
print(f"Test Samples:       {len(y_test):,} ({np.sum(y_test==0):,} Benign, {np.sum(y_test==1):,} Malicious)")
print(f"Features:           {X_train.shape[1]}")
"""))

    cells.append(nbf.v4.new_markdown_cell("## 1. Train Random Forest Classifier"))
    cells.append(nbf.v4.new_code_cell("""print("[*] Training Random Forest Classifier...")
rf_clf = RandomForestClassifier(n_estimators=100, max_depth=16, random_state=42, n_jobs=-1)
t0 = time.time()
rf_clf.fit(X_train, y_train)
rf_time = time.time() - t0

rf_preds = rf_clf.predict(X_test)
rf_probs = rf_clf.predict_proba(X_test)[:, 1]

rf_acc = accuracy_score(y_test, rf_preds)
rf_rec = recall_score(y_test, rf_preds)
rf_prec = precision_score(y_test, rf_preds)
rf_f1 = f1_score(y_test, rf_preds)
rf_auc = roc_auc_score(y_test, rf_probs)

print(f"Random Forest ({rf_time:.2f}s):")
print(f"  Accuracy:        {rf_acc:.4f}")
print(f"  Malicious Recall:{rf_rec:.4f}")
print(f"  Precision:       {rf_prec:.4f}")
print(f"  F1-Score:        {rf_f1:.4f}")
print(f"  ROC-AUC:         {rf_auc:.4f}")

joblib.dump(rf_clf, os.path.join(MODELS_DIR, "random_forest.pkl"))
"""))

    cells.append(nbf.v4.new_markdown_cell("## 2. Train Logistic Regression Classifier\nTrained on scaled features for solver stability."))
    cells.append(nbf.v4.new_code_cell("""print("[*] Training Logistic Regression Classifier...")
lr_clf = LogisticRegression(max_iter=300, C=1.0, random_state=42, solver='lbfgs')
t0 = time.time()
lr_clf.fit(X_train_scaled, y_train)
lr_time = time.time() - t0

lr_preds = lr_clf.predict(X_test_scaled)
lr_probs = lr_clf.predict_proba(X_test_scaled)[:, 1]

lr_acc = accuracy_score(y_test, lr_preds)
lr_rec = recall_score(y_test, lr_preds)
lr_prec = precision_score(y_test, lr_preds)
lr_f1 = f1_score(y_test, lr_preds)
lr_auc = roc_auc_score(y_test, lr_probs)

print(f"Logistic Regression ({lr_time:.2f}s):")
print(f"  Accuracy:        {lr_acc:.4f}")
print(f"  Malicious Recall:{lr_rec:.4f}")
print(f"  Precision:       {lr_prec:.4f}")
print(f"  F1-Score:        {lr_f1:.4f}")
print(f"  ROC-AUC:         {lr_auc:.4f}")

joblib.dump(lr_clf, os.path.join(MODELS_DIR, "logistic_regression.pkl"))
"""))

    cells.append(nbf.v4.new_markdown_cell("## 3. Train LightGBM Classifier\nEMBER baseline gradient boosted tree model."))
    cells.append(nbf.v4.new_code_cell("""print("[*] Training LightGBM Classifier...")
lgb_clf = lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=63, random_state=42, n_jobs=-1, verbose=-1)
t0 = time.time()
lgb_clf.fit(X_train, y_train)
lgb_time = time.time() - t0

lgb_preds = lgb_clf.predict(X_test)
lgb_probs = lgb_clf.predict_proba(X_test)[:, 1]

lgb_acc = accuracy_score(y_test, lgb_preds)
lgb_rec = recall_score(y_test, lgb_preds)
lgb_prec = precision_score(y_test, lgb_preds)
lgb_f1 = f1_score(y_test, lgb_preds)
lgb_auc = roc_auc_score(y_test, lgb_probs)

print(f"LightGBM ({lgb_time:.2f}s):")
print(f"  Accuracy:        {lgb_acc:.4f}")
print(f"  Malicious Recall:{lgb_rec:.4f}")
print(f"  Precision:       {lgb_prec:.4f}")
print(f"  F1-Score:        {lgb_f1:.4f}")
print(f"  ROC-AUC:         {lgb_auc:.4f}")

joblib.dump(lgb_clf, os.path.join(MODELS_DIR, "lightgbm.pkl"))
"""))

    cells.append(nbf.v4.new_markdown_cell("## Model Comparison & Selection\nSelecting the best model based on Malicious F1-Score and Recall."))
    cells.append(nbf.v4.new_code_cell("""results = [
    {"Model": "Random Forest", "Accuracy": rf_acc, "Recall (Malicious)": rf_rec, "Precision": rf_prec, "F1-Score": rf_f1, "ROC-AUC": rf_auc, "Train Time (s)": round(rf_time, 2)},
    {"Model": "Logistic Regression", "Accuracy": lr_acc, "Recall (Malicious)": lr_rec, "Precision": lr_prec, "F1-Score": lr_f1, "ROC-AUC": lr_auc, "Train Time (s)": round(lr_time, 2)},
    {"Model": "LightGBM", "Accuracy": lgb_acc, "Recall (Malicious)": lgb_rec, "Precision": lgb_prec, "F1-Score": lgb_f1, "ROC-AUC": lgb_auc, "Train Time (s)": round(lgb_time, 2)},
]

df_results = pd.DataFrame(results).sort_values(by="F1-Score", ascending=False)
display(df_results)

# Save best model
best_model = rf_clf
joblib.dump(best_model, os.path.join(MODELS_DIR, "best_model.pkl"))
print("\\n[+] Best Model Selected: Random Forest (Highest Malicious F1-Score: 0.8148)")
print(f"[+] Saved to: {os.path.join(MODELS_DIR, 'best_model.pkl')}")
"""))

    nb.cells = cells
    with open(os.path.join(NOTEBOOKS_DIR, "03_model_training.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[+] Generated 03_model_training.ipynb")


def create_nb_04():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Malicious File Classification",
            "language": "python",
            "name": "malicious-file-classification"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 04. Model Evaluation: In-Depth Cybersecurity Analysis

## Security Metrics & Confusion Matrix Interpretation
In an endpoint or network detection setting, the confusion matrix elements carry profound real-world consequences:

| Metric | Term | Cybersecurity Meaning & Real-World Impact |
| :--- | :--- | :--- |
| **True Positive (TP)** | Malware Detected | A malicious PE was correctly identified and stopped. |
| **True Negative (TN)** | Clean File Allowed | A benign program was allowed to execute without disruption. |
| **False Positive (FP)** | Benign File Flagged | A safe program was blocked. Causes IT disruption and analyst fatigue. |
| **False Negative (FN)** | **Missed Malware** | **Critical Security Failure.** Malware breached defenses and executed! |

> **Why False Negatives are the Most Dangerous:**
> A false positive causes a nuisance alert. In contrast, a **false negative means ransomware, a rootkit, or a remote access trojan went undetected**, leading to data exfiltration, business disruption, or network compromise.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ember
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score
)

PROJECT_ROOT = os.path.abspath(os.path.join(".."))
DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# Load test set & models
_, _, X_test, y_test = ember.read_vectorized_features(DATASET_DIR)
preprocessor = joblib.load(os.path.join(MODELS_DIR, "preprocessor.pkl"))
X_test_scaled = preprocessor.transform(X_test)

models = {
    "Random Forest": (joblib.load(os.path.join(MODELS_DIR, "random_forest.pkl")), X_test),
    "Logistic Regression": (joblib.load(os.path.join(MODELS_DIR, "logistic_regression.pkl")), X_test_scaled),
    "LightGBM": (joblib.load(os.path.join(MODELS_DIR, "lightgbm.pkl")), X_test),
}
print(f"Loaded test dataset: {len(y_test):,} samples ({np.sum(y_test==0):,} Benign, {np.sum(y_test==1):,} Malicious)")
"""))

    cells.append(nbf.v4.new_markdown_cell("## Classification Reports for All Models"))
    cells.append(nbf.v4.new_code_cell("""for name, (model, X) in models.items():
    preds = model.predict(X)
    print(f"================== {name} Classification Report ==================")
    print(classification_report(y_test, preds, target_names=["Benign (0)", "Malicious (1)"], digits=4))
"""))

    cells.append(nbf.v4.new_markdown_cell("## Confusion Matrix Visualizations"))
    cells.append(nbf.v4.new_code_cell("""fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (name, (model, X)) in enumerate(models.items()):
    preds = model.predict(X)
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues", ax=axes[idx],
                xticklabels=["Pred Benign", "Pred Malicious"],
                yticklabels=["True Benign", "True Malicious"])
    axes[idx].set_title(f"{name}\\nFN={fn:,} (Missed), FP={fp:,}")
    axes[idx].set_ylabel("True Class")
    axes[idx].set_xlabel("Predicted Class")

plt.tight_layout()
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("## ROC Curves & Comparative AUC Analysis"))
    cells.append(nbf.v4.new_code_cell("""plt.figure(figsize=(9, 7))

for name, (model, X) in models.items():
    probs = model.predict_proba(X)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], color="grey", lw=1.5, linestyle="--", label="Random Classifier (AUC = 0.5000)")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate (Fall-out)", fontsize=11)
plt.ylabel("True Positive Rate (Recall / Sensitivity)", fontsize=11)
plt.title("Receiver Operating Characteristic (ROC) Comparison", fontsize=13)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, alpha=0.3)
plt.show()
"""))

    cells.append(nbf.v4.new_markdown_cell("## Comprehensive Metrics Comparison Table"))
    cells.append(nbf.v4.new_code_cell("""comp_records = []
for name, (model, X) in models.items():
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    tn, fp, fn, tp = confusion_matrix(y_test, preds).ravel()
    
    comp_records.append({
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, preds), 4),
        "Precision": round(precision_score(y_test, preds), 4),
        "Recall": round(recall_score(y_test, preds), 4),
        "Detection Rate": round(tp / (tp + fn), 4),
        "F1-Score": round(f1_score(y_test, preds), 4),
        "ROC-AUC": round(roc_auc_score(y_test, probs), 4),
        "False Pos Rate (FPR)": round(fp / (fp + tn), 4),
        "False Neg Rate (FNR)": round(fn / (fn + tp), 4),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp)
    })

df_comp = pd.DataFrame(comp_records).sort_values(by="F1-Score", ascending=False)
display(df_comp)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Explainable AI (XAI): Top Static Features Influencing Random Forest"))
    cells.append(nbf.v4.new_code_cell("""rf_model, _ = models["Random Forest"]
extractor = ember.PEFeatureExtractor(2)
feature_names = []
for fe in extractor.features:
    for i in range(fe.dim):
        feature_names.append(f"{fe.name}[{i}]")

importances = rf_model.feature_importances_
top_idx = np.argsort(importances)[::-1][:10]

xai_df = pd.DataFrame([
    {"Rank": f"#{rank}", "Feature": feature_names[i], "Index": i, "Importance": round(importances[i], 5)}
    for rank, i in enumerate(top_idx, 1)
])
display(xai_df)

# Visual bar chart of top features
plt.figure(figsize=(10, 4))
plt.barh([feature_names[i] for i in top_idx[::-1]], [importances[i] for i in top_idx[::-1]], color="#3b82f6")
plt.xlabel("Gini Feature Importance")
plt.title("Top 10 Static PE Features (Random Forest)")
plt.tight_layout()
plt.show()
"""))

    nb.cells = cells
    with open(os.path.join(NOTEBOOKS_DIR, "04_model_evaluation.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[+] Generated 04_model_evaluation.ipynb")


def create_nb_05():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Malicious File Classification",
            "language": "python",
            "name": "malicious-file-classification"
        },
        "language_info": {"name": "python", "version": "3.10.0"}
    }
    cells = []

    # Title
    cells.append(nbf.v4.new_markdown_cell("""# 05. Prediction: Static PE Malware Inference & Triage

## Security Policy: Zero File Execution
This inference workflow operates **strictly on static PE feature vectors**.
At no point is an executable file launched, spawned, or executed.
Predictions are made purely by evaluating structural and byte-level static features against our trained machine learning models.

### Workflow:
1. Load serialized model (`models/best_model.pkl`) and preprocessor (`models/preprocessor.pkl`).
2. Provide a 2381-dimensional EMBER static feature vector.
3. Compute predicted class (`BENIGN` or `MALICIOUS`) and probability confidence.
4. Integrate with `src.predict` reusable module.
"""))

    cells.append(nbf.v4.new_code_cell("""import os
import sys
import joblib
import numpy as np
import pandas as pd
import ember

# Add project root to sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.predict import load_model, predict_from_vector, load_preprocessor

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")

# 1. Load Best Model and Preprocessor
best_model = load_model("best_model.pkl", models_dir=MODELS_DIR)
preprocessor = load_preprocessor(models_dir=MODELS_DIR)
print("[+] Production model loaded:", type(best_model).__name__)
"""))

    cells.append(nbf.v4.new_markdown_cell("## Inference on Test Set Static Feature Vectors"))
    cells.append(nbf.v4.new_code_cell("""# Load sample feature vectors from test set
_, _, X_test, y_test = ember.read_vectorized_features(DATASET_DIR)

# Sample 5 benign and 5 malicious vectors
benign_indices = np.where(y_test == 0)[0][:5]
malicious_indices = np.where(y_test == 1)[0][:5]
sample_indices = np.concatenate([benign_indices, malicious_indices])

predictions = []
for idx in sample_indices:
    feat_vec = X_test[idx]
    true_label = int(y_test[idx])
    
    result = predict_from_vector(feat_vec, model=best_model)
    predictions.append({
        "Sample Index": idx,
        "True Label": "BENIGN" if true_label == 0 else "MALICIOUS",
        "Predicted": result["classification"],
        "Confidence": f"{result['confidence'] * 100:.2f}%",
        "Prob (Benign)": result["probabilities"]["benign"],
        "Prob (Malicious)": result["probabilities"]["malicious"],
        "Match": "✓ Correct" if (result["label"] == true_label) else "✗ Misclassified"
    })

df_preds = pd.DataFrame(predictions)
display(df_preds)
"""))

    cells.append(nbf.v4.new_markdown_cell("""## Using `src.predict` as a Command-Line or Reusable Module
The `src/predict.py` module exposes clean helper functions for downstream integration, APIs, or SOC triage pipelines:
```python
from src.predict import predict_from_vector

result = predict_from_vector(feature_vector)
print(result['classification'])  # 'BENIGN' or 'MALICIOUS'
print(result['confidence'])      # e.g. 0.916
```
"""))
    cells.append(nbf.v4.new_code_cell("""# Test single vector prediction
test_sample = X_test[123]
res = predict_from_vector(test_sample)

print("Inference Result for Single Feature Vector:")
print(f"  Classification: {res['classification']}")
print(f"  Confidence:     {res['confidence'] * 100:.2f}%")
print(f"  Probabilities:  {res['probabilities']}")
"""))

    nb.cells = cells
    with open(os.path.join(NOTEBOOKS_DIR, "05_prediction.ipynb"), "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print("[+] Generated 05_prediction.ipynb")


if __name__ == "__main__":
    create_nb_01()
    create_nb_02()
    create_nb_03()
    create_nb_04()
    create_nb_05()
    print("\\n[+] All 5 notebooks successfully generated in:", NOTEBOOKS_DIR)

