"""
Small validation test script to verify EMBER feature extraction and scikit-learn compatibility.
Tests a small number of records from the local EMBER dataset without consuming high memory.
"""
import os
import sys
import json
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

def test_ember_vectorization(dataset_dir=None, num_samples=10):
    if dataset_dir is None:
        dataset_dir = os.path.join(PROJECT_ROOT, "dataset", "ember2018")
    
    print(f"[*] Starting small EMBER vectorization test...")
    print(f"[*] Project root: {PROJECT_ROOT}")
    print(f"[*] Dataset directory: {dataset_dir}")
    
    # Check dataset existence
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    
    train_file = os.path.join(dataset_dir, "train_features_0.jsonl")
    if not os.path.isfile(train_file):
        raise FileNotFoundError(f"Missing train feature file: {train_file}")
    
    # Import EMBER and initialize PEFeatureExtractor
    import ember
    import sklearn
    print(f"[*] scikit-learn version: {sklearn.__version__}")
    print(f"[*] Initializing PEFeatureExtractor(feature_version=2)...")
    extractor = ember.PEFeatureExtractor(feature_version=2)
    expected_dim = extractor.dim
    print(f"[*] Expected feature dimension: {expected_dim}")
    
    # Read a few records
    print(f"[*] Reading first {num_samples} records from {os.path.basename(train_file)}...")
    extracted_features = []
    labels = []
    sha256_list = []
    
    with open(train_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx >= num_samples:
                break
            record = json.loads(line)
            
            # Verify fields
            label = record.get("label")
            sha256 = record.get("sha256")
            
            # Extract features - this tests FeatureHasher compatibility
            feat_vec = extractor.process_raw_features(record)
            
            # Validations
            assert isinstance(feat_vec, np.ndarray), f"Row {idx}: Feature is not a numpy array"
            assert feat_vec.shape == (expected_dim,), f"Row {idx}: Shape mismatch! Expected ({expected_dim},), got {feat_vec.shape}"
            assert label in [0, 1, -1], f"Row {idx}: Invalid label {label} (expected 0, 1, or -1)"
            assert not np.isnan(feat_vec).any(), f"Row {idx}: NaN values found in feature vector"
            assert not np.isinf(feat_vec).any(), f"Row {idx}: Infinite values found in feature vector"
            
            extracted_features.append(feat_vec)
            labels.append(label)
            sha256_list.append(sha256)
            print(f"  [+] Sample {idx+1}/{num_samples}: sha256={sha256[:12]}..., label={label}, vector_dim={len(feat_vec)}")
            
    features_matrix = np.array(extracted_features)
    print(f"\n[+] SUCCESS: Tested {num_samples} records successfully!")
    print(f"    - Matrix shape: {features_matrix.shape}")
    print(f"    - Data type: {features_matrix.dtype}")
    print(f"    - Unique labels tested: {set(labels)}")
    print(f"    - FeatureHasher compatibility verified: No errors thrown.")
    return True

if __name__ == "__main__":
    test_ember_vectorization()

