# MLCS-HACK-16 — Malicious File Classification
**Theme: Machine Learning for Cybersecurity**

---

## 1. Problem Statement
Develop an end-to-end Machine Learning prototype for **Malicious File Classification** that detects, classifies, and prioritizes cybersecurity-relevant threats using static Portable Executable (PE) features and machine learning.

The system performs **static-analysis-only** triage on Windows `.exe` and `.dll` files, extracting 2,381 structural, entropy, and header characteristics without executing untrusted binaries.

---

## 2. System Architecture & Workflow

```text
             EMBER 2018
                  ↓
          Data Preprocessing
                  ↓
         Static PE Features
                  ↓
           2,381 Features
                  ↓
          ML Model Training
        ┌─────────┼─────────┐
        ↓         ↓         ↓
   Random Forest Logistic  LightGBM
        └─────────┼─────────┘
                  ↓
           Model Evaluation
                  ↓
           Trained Model
                  ↓
             PE Upload
                  ↓
          Static PE Analysis
                  ↓
           Feature Extraction
                  ↓
           ML Classification
                  ↓
        ┌─────────┴─────────┐
        ↓                   ↓
     BENIGN              MALICIOUS
        ↓                   ↓
  Model Probability   Model Probability
```

---

## 3. Zero-Execution Security Policy

> **CRITICAL SECURITY REQUIREMENT:**
> This system operates **STRICTLY ON STATIC PE FEATURES ONLY**.
> - **Never** execute PE files or launch malware samples.
> - **Never** dynamically load uploaded DLLs via `ctypes.CDLL`, `WinDLL`, or Windows `LoadLibrary`.
> - **Never** use `subprocess`, `os.system`, `os.startfile`, or shell runners to trigger untrusted executables.
> - **Never** disable Windows Defender or recommend disabling antivirus protections.
> - Uploaded files are ingested purely as in-memory raw binary data, inspected statically via `lief`, converted into numerical vectors, and classified.
> - Uploaded binaries are never permanently written to disk.

---

## 4. Dataset Information & Pipeline

- **Benchmark Dataset**: EMBER (Endgame Malware BEnchmark for Research) 2018 v2
- **Feature Representation**: 2,381 numerical static features per PE file:
  1. Byte histogram (256 dims)
  2. Byte-entropy 2D distribution (256 dims)
  3. String metrics and path entropy (104 dims)
  4. General file info (size, relocs, debug, exports/imports) (10 dims)
  5. Header characteristics (machine, subsystem, DLL characteristics, magic) (62 dims)
  6. Section characteristics (entropy, permissions, entry section) (255 dims)
  7. Imports table (hashed libraries and API functions) (1,280 dims)
  8. Exports table (hashed exported symbols) (128 dims)
  9. Data directories (virtual sizes and characteristics) (30 dims)
- **Experimental Training Subset**: **20,000 balanced samples** (10,000 Benign, 10,000 Malicious).  
  *(Note: 20,000 is the configured experimental subset used for this project, not the 1,000,000 total samples of the full raw archive).*
- **Test Evaluation Set**: **5,000 balanced samples** (2,500 Benign, 2,500 Malicious).

---

## 5. Model Evaluation & Security Metrics

Models evaluated on the held-out 5,000-sample test set (2,500 Benign, 2,500 Malicious):

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Detection Rate | False Pos Rate (FPR) | False Neg Rate (FNR) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Random Forest ⭐** | **80.12%** | **76.27%** | **87.44%** | **0.8148** | **0.8794** | **87.44%** | **27.20%** | **12.56%** |
| **Logistic Regression** | 70.66% | 66.69% | 82.56% | 0.7378 | 0.7812 | 82.56% | 41.24% | 17.44% |
| **LightGBM** | 62.16% | 58.06% | 87.60% | 0.6983 | 0.6764 | 87.60% | 63.28% | 12.40% |

*Selected Best Model: **Random Forest** due to superior Malicious F1-Score (0.8148), balanced Precision/Recall, and lowest False Positive Rate.*

### Actual Confusion Matrix (Random Forest):
```text
                      PREDICTED
                  Benign    Malicious
ACTUAL Benign    1,820 (TN)    680 (FP)
ACTUAL Malicious   314 (FN)  2,186 (TP)
```

- **True Negatives (TN)** = 1,820 (Clean files correctly allowed)
- **False Positives (FP)** = 680 (Clean files flagged for review)
- **False Negatives (FN)** = 314 (Missed malware)
- **True Positives (TP)** = 2,186 (Malware successfully detected)

### Security Metric Formulations:
- **Detection Rate** $= \frac{\text{TP}}{\text{TP} + \text{FN}} = \frac{2186}{2500} = \mathbf{87.44\%}$
- **False Positive Rate (FPR)** $= \frac{\text{FP}}{\text{FP} + \text{TN}} = \frac{680}{2500} = \mathbf{27.20\%}$
- **False Negative Rate (FNR)** $= \frac{\text{FN}}{\text{FN} + \text{TP}} = \frac{314}{2500} = \mathbf{12.56\%}$
- **F1-Score** $= 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = \mathbf{0.8148}$

---

## 6. Security Interpretation

| Metric / Term | Definition | Cybersecurity Real-World Impact |
| :--- | :--- | :--- |
| **False Negative (FN)** | Malicious file classified as Benign | **Most Critical Failure Mode.** A missed trojan, ransomware, or wiper passes perimeter defenses and executes, resulting in data breaches, ransomware encryption, or persistent compromise. |
| **False Positive (FP)** | Benign file classified as Malicious | Leads to SOC analyst alert fatigue, ticket backlogs, and user frustration when legitimate software is quarantined. |
| **Detection Rate** | True Positive Rate ($\text{Recall}$) | The proportion of total malware threats successfully intercepted by the ML model. |
| **FPR** | Fall-out Rate | The frequency at which harmless business software is flagged as suspicious. |

---

## 7. Explainable AI (XAI): Top Static PE Features

Using native Random Forest **Gini Feature Importance** across the 2,381 EMBER dimensions:

| Rank | Feature | Category | Importance | Cybersecurity Interpretation |
| :---: | :--- | :--- | :---: | :--- |
| **#1** | `header[0]` | COFF Timestamp | **0.0423** | Compilation timestamp anomalies (spoofed, zeroed, or future dates common in malware) |
| **#2** | `header[11]` | Optional Header Subsystem | **0.0186** | Subsystem specifications & major image linker versions |
| **#3** | `byteentropy[250]` | Byte Entropy Histogram | **0.0177** | High entropy code distribution (indicative of packers, cryptors, or obfuscation) |
| **#4** | `byteentropy[246]` | Byte Entropy Histogram | **0.0138** | High entropy block frequency |
| **#5** | `byteentropy[243]` | Byte Entropy Histogram | **0.0135** | Entropy variance between code sections and resources |
| **#6** | `histogram[90]` | Byte Histogram | **0.0131** | Frequency of byte 0x5A ('Z' in 'MZ' DOS header signature) |
| **#7** | `histogram[151]` | Byte Histogram | **0.0119** | Opcode distribution frequency for byte 0x97 |
| **#8** | `byteentropy[245]` | Byte Entropy Histogram | **0.0116** | Dense entropy clusters in executable sections |
| **#9** | `byteentropy[254]` | Byte Entropy Histogram | **0.0115** | Maximum entropy bin (encrypted payloads & compressed shellcode) |
| **#10** | `strings[84]` | Strings Info | **0.0112** | Printable string length, path entropy, and URL indicators |

---

## 8. Real-Time Static PE Classification (.EXE / .DLL Upload)

The application provides a dedicated interface and REST API for real-time static analysis of Windows PE files without executing them.

> **Security Statement:**  
> "This project performs static malware classification only. Uploaded PE files are treated as untrusted binary data and are never executed."

### Step-by-Step Analysis Flow:
1. **File Upload**: The user uploads an untrusted `.exe` or `.dll` file (up to 100 MB).
2. **Raw Byte Ingestion**: The server receives the file purely as in-memory raw bytes.
3. **PE Structure Validation**: Validates the DOS header signature (`MZ`) and PE header signature (`PE\0\0`). Non-PE files and corrupted binaries are immediately rejected.
4. **Cryptographic Identification**: Computes the SHA-256 hash and file size.
5. **Static PE Inspection**: Parses structural metadata (architecture, sections count, imported DLLs/functions, exports, entry point, entropy) via LIEF without execution.
6. **EMBER Feature Extraction**: Extracts the exact 2,381-dimensional static EMBER feature vector from raw bytes.
7. **Preprocessing & Model Inference**: The 2,381 feature vector is fed to `best_model.pkl` (Random Forest).
8. **Verdict & Probability Output**: Computes classification (`BENIGN` or `MALICIOUS`), confidence, and class probabilities.
9. **Zero-Execution Assurance**: The uploaded file is processed in memory and never executed, loaded via `ctypes`/`LoadLibrary`, or stored permanently.

### REST API Usage:
```powershell
curl -X POST "http://127.0.0.1:8000/api/predict-file" `
     -F "file=@C:\Windows\System32\notepad.exe"
```

Example JSON Response:
```json
{
  "success": true,
  "filename": "notepad.exe",
  "file_size": 360448,
  "sha256": "468ffe129c395abf6b21a09efdf261910a95fb98aa982ead73caa7b2b684577e",
  "pe_valid": true,
  "feature_count": 2381,
  "prediction": "BENIGN",
  "benign_probability": 0.5883,
  "malicious_probability": 0.4117,
  "confidence": 0.5883,
  "inference_ms": 187.94,
  "static_metadata": {
    "architecture": "AMD64",
    "pe_type": "PE32_PLUS",
    "entry_point": "0x19c0",
    "image_base": "0x140000000",
    "sections_count": 8,
    "imported_dlls": 50,
    "imported_functions": 315,
    "exported_functions": 0,
    "average_entropy": 4.17
  },
  "security_notice": "Static analysis only. Zero executable execution."
}
```

---

## 9. Quickstart & Commands

### 1. Activate Environment
```powershell
cd C:\MaliciousFileClassification
.\.venv\Scripts\Activate.ps1
```

### 2. Run Dashboard Web Service
```powershell
python -m uvicorn src.app:app --host 127.0.0.1 --port 8000
```
Open **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)** in your browser.

### 3. Run Automated Validation Test Suite
```powershell
python src/test_upload_endpoint.py
```

### 4. Run Notebooks (in order)
```powershell
jupyter lab
```
1. `notebooks/01_data_analysis.ipynb`
2. `notebooks/02_preprocessing.ipynb`
3. `notebooks/03_model_training.ipynb`
4. `notebooks/04_model_evaluation.ipynb`
5. `notebooks/05_prediction.ipynb`
