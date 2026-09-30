"""
Static PE Malware Prediction Module.
Loads trained machine learning models and performs inference on static PE feature vectors
and raw PE file bytes (via static EMBER feature extraction).

SECURITY POLICY:
Strictly static feature analysis only - NO file execution, NO dynamic DLL loading, NO subprocess execution.
"""
import os
import sys
import struct
import hashlib
import time
import joblib
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

DEFAULT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100 MB maximum upload size
ALLOWED_EXTENSIONS = {".exe", ".dll"}


def validate_pe_bytes(bytez: bytes):
    """
    Statically validates whether byte sequence conforms to a valid Windows PE file.
    Checks:
    - Minimum size (at least 64 bytes for DOS header)
    - DOS signature ('MZ' / 0x5A4D)
    - PE signature offset at 0x3C
    - PE signature ('PE\\0\\0' / 0x00004550)
    
    Returns:
        (is_valid: bool, reason: str)
    """
    if not isinstance(bytez, (bytes, bytearray)):
        return False, "Input must be raw bytes."

    if len(bytez) < 64:
        return False, "File too small to be a Windows PE file (less than 64 bytes)."

    if bytez[:2] != b"MZ":
        return False, "Invalid Windows PE: Missing DOS header signature ('MZ')."

    try:
        e_lfanew = struct.unpack_from("<I", bytez, 0x3C)[0]
        if e_lfanew <= 0 or e_lfanew + 4 > len(bytez):
            return False, "Invalid Windows PE: PE header offset (e_lfanew) is out of bounds."
        if bytez[e_lfanew:e_lfanew + 4] != b"PE\x00\x00":
            return False, "Invalid Windows PE: Missing PE signature ('PE\\0\\0')."
    except Exception as e:
        return False, f"Invalid Windows PE header structure: {str(e)}"

    return True, "Valid Windows PE binary"


def extract_static_pe_metadata(bytez: bytes):
    """
    Statically inspects PE structure via LIEF parser without executing the binary.
    Extracts high-level descriptive metadata (architecture, entry point, sections, imports).
    """
    import lief
    metadata = {
        "architecture": "Unknown",
        "pe_type": "Unknown",
        "entry_point": "0x0",
        "image_base": "0x0",
        "sections_count": 0,
        "imported_dlls": 0,
        "imported_functions": 0,
        "exported_functions": 0,
        "average_entropy": 0.0,
        "sections_summary": []
    }

    try:
        b = lief.PE.parse(list(bytez))
        if b is None:
            return metadata

        if hasattr(b, "header") and hasattr(b.header, "machine"):
            metadata["architecture"] = str(b.header.machine).replace("HEADER_MACHINES.", "").replace("MACHINE_TYPES.", "")

        if hasattr(b, "optional_header"):
            if hasattr(b.optional_header, "magic"):
                metadata["pe_type"] = str(b.optional_header.magic).replace("PE_TYPE.", "")
            if hasattr(b.optional_header, "addressof_entrypoint"):
                metadata["entry_point"] = hex(b.optional_header.addressof_entrypoint)
            if hasattr(b.optional_header, "imagebase"):
                metadata["image_base"] = hex(b.optional_header.imagebase)

        if hasattr(b, "sections"):
            metadata["sections_count"] = len(b.sections)
            entropies = [s.entropy for s in b.sections]
            metadata["average_entropy"] = round(float(sum(entropies) / len(entropies)), 2) if entropies else 0.0
            metadata["sections_summary"] = [
                {"name": s.name[:8], "size": int(s.size), "entropy": round(float(s.entropy), 2)}
                for s in b.sections[:10]
            ]

        if hasattr(b, "imports"):
            metadata["imported_dlls"] = len(b.imports)
            metadata["imported_functions"] = sum(len(imp.entries) for imp in b.imports)

        if hasattr(b, "exported_functions"):
            metadata["exported_functions"] = len(b.exported_functions)

    except Exception as e:
        metadata["parsing_note"] = f"Partial metadata extracted: {str(e)}"

    return metadata


def extract_features_from_pe_bytes(bytez: bytes):
    """
    Statically extracts the exact 2381-dimensional EMBER v2 feature vector
    from raw PE file bytes.
    Matches the training representation.
    """
    import ember
    extractor = ember.PEFeatureExtractor(feature_version=2)
    feature_vec = extractor.feature_vector(bytez)
    
    if len(feature_vec) != 2381:
        raise ValueError(f"Feature extraction produced {len(feature_vec)} features, but the model expects 2381.")
    
    return feature_vec.astype(np.float32)


def load_model(model_name="best_model.pkl", models_dir=DEFAULT_MODELS_DIR):
    """Load a trained model by filename from models directory."""
    model_path = os.path.join(models_dir, model_name)
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    return joblib.load(model_path)


def load_preprocessor(models_dir=DEFAULT_MODELS_DIR):
    """Load the saved feature preprocessor (StandardScaler) if available."""
    prep_path = os.path.join(models_dir, "preprocessor.pkl")
    if os.path.isfile(prep_path):
        return joblib.load(prep_path)
    return None


def predict_from_vector(vector, model=None, preprocessor=None, use_scaled=False, model_name="best_model.pkl"):
    """
    Accept an EMBER static feature vector (2381 elements) and return classification.
    """
    if model is None:
        model = load_model(model_name=model_name)

    vec = np.array(vector, dtype=np.float32)
    single_sample = False
    if vec.ndim == 1:
        if len(vec) != 2381:
            raise ValueError(f"Expected 2381 EMBER features, received {len(vec)} features.")
        vec = vec.reshape(1, -1)
        single_sample = True
    elif vec.ndim == 2:
        if vec.shape[1] != 2381:
            raise ValueError(f"Expected feature shape (*, 2381), received {vec.shape}")
    else:
        raise ValueError(f"Invalid input dimensions: {vec.ndim}")

    if use_scaled:
        if preprocessor is None:
            preprocessor = load_preprocessor()
        if preprocessor is not None:
            vec = preprocessor.transform(vec)

    preds = model.predict(vec)
    probs = model.predict_proba(vec) if hasattr(model, "predict_proba") else None

    results = []
    for i in range(len(preds)):
        lbl = int(preds[i])
        cls_name = "MALICIOUS" if lbl == 1 else "BENIGN"
        prob_dict = {}
        confidence = 1.0

        if probs is not None:
            p_benign = float(probs[i][0])
            p_mal = float(probs[i][1])
            prob_dict = {"benign": round(p_benign, 4), "malicious": round(p_mal, 4)}
            confidence = prob_dict["malicious"] if lbl == 1 else prob_dict["benign"]

        results.append({
            "label": lbl,
            "classification": cls_name,
            "confidence": round(confidence, 4),
            "probabilities": prob_dict
        })

    return results[0] if single_sample else results


def predict_from_pe_bytes(bytez: bytes, filename: str = "sample.exe",
                          model=None, preprocessor=None, model_name="best_model.pkl"):
    """
    Full secure pipeline for static PE classification from raw bytes:
    1. Validation of PE structure
    2. Calculation of SHA-256
    3. Static metadata extraction
    4. EMBER 2,381 feature vector extraction
    5. Inference with trained model
    """
    # 1. Validation
    is_valid, reason = validate_pe_bytes(bytez)
    if not is_valid:
        raise ValueError(f"Invalid Windows PE file: {reason}")

    # 2. SHA-256 & file size
    sha256_hash = hashlib.sha256(bytez).hexdigest()
    file_size = len(bytez)

    # 3. Static metadata
    meta = extract_static_pe_metadata(bytez)

    # 4. Feature extraction
    t0 = time.time()
    feature_vec = extract_features_from_pe_bytes(bytez)
    extract_time_ms = (time.time() - t0) * 1000.0

    # 5. Model Inference
    use_scaled = "logistic" in model_name.lower()
    t1 = time.time()
    pred_res = predict_from_vector(
        vector=feature_vec,
        model=model,
        preprocessor=preprocessor,
        use_scaled=use_scaled,
        model_name=model_name
    )
    infer_time_ms = (time.time() - t1) * 1000.0
    total_time_ms = extract_time_ms + infer_time_ms

    return {
        "success": True,
        "filename": os.path.basename(filename),
        "file_size": file_size,
        "sha256": sha256_hash,
        "pe_valid": True,
        "feature_count": len(feature_vec),
        "prediction": pred_res["classification"],
        "label": pred_res["label"],
        "confidence": pred_res["confidence"],
        "benign_probability": pred_res["probabilities"].get("benign", 0.0),
        "malicious_probability": pred_res["probabilities"].get("malicious", 0.0),
        "probabilities": pred_res["probabilities"],
        "inference_ms": round(total_time_ms, 2),
        "static_metadata": meta,
        "feature_preview": [round(float(x), 4) for x in feature_vec[:30]],
        "security_notice": "Static analysis only. Zero executable execution."
    }


def predict_features(features, model=None, preprocessor=None, use_scaled=False):
    """Alias for predict_from_vector to adhere to pipeline interfaces."""
    return predict_from_vector(features, model=model, preprocessor=preprocessor, use_scaled=use_scaled)


if __name__ == "__main__":
    print("[*] Testing predict_from_pe_bytes with C:\\Windows\\System32\\notepad.exe...")
    with open(r"C:\Windows\System32\notepad.exe", "rb") as f:
        data = f.read()

    res = predict_from_pe_bytes(data, filename="notepad.exe")
    print(f"Result for {res['filename']}:")
    print(f"  SHA-256:       {res['sha256']}")
    print(f"  PE Valid:      {res['pe_valid']}")
    print(f"  Architecture:  {res['static_metadata']['architecture']}")
    print(f"  Features:      {res['feature_count']}")
    print(f"  Prediction:    {res['prediction']} (Confidence: {res['confidence']*100:.1f}%)")
    print(f"  Probabilities: {res['probabilities']}")
    print(f"  Inference:     {res['inference_ms']} ms")
    print("\n[+] predict.py self-test passed successfully.")
