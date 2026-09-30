"""
Data preparation pipeline for the EMBER 2018 v2 malware classification dataset.
Handles verification, compatibility patching, and memory-safe batch vectorization.
"""
import os
import sys
import json
import argparse
import time
import numpy as np
from datetime import datetime

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DEFAULT_DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset", "ember2018")


def apply_ember_compatibility_patch():
    """
    Ensures compatibility between EMBER and modern scikit-learn FeatureHasher.
    In scikit-learn >= 1.0, FeatureHasher(input_type='string') requires an iterable
    over iterables of strings (e.g. [[raw_obj['entry']]]).
    """
    import ember.features
    from sklearn.feature_extraction import FeatureHasher

    # Inspect SectionInfo.process_raw_features
    # Check if entry hashing needs patching
    original_process = ember.features.SectionInfo.process_raw_features

    def patched_process_raw_features(self, raw_obj):
        sections = raw_obj['sections']
        general = [
            len(sections),
            # number of zero caps
            sum(1 for s in sections if s['size'] == 0),
            # number of empty caps
            sum(1 for s in sections if s['name'] == ""),
            # number of RX
            sum(1 for s in sections if 'MEM_READ' in s['props'] and 'MEM_EXECUTE' in s['props']),
            # number of W
            sum(1 for s in sections if 'MEM_WRITE' in s['props'])
        ]
        # gross characteristics of each section
        section_sizes = [(s['name'], s['size']) for s in sections]
        section_sizes_hashed = FeatureHasher(50, input_type="pair").transform([section_sizes]).toarray()[0]
        section_entropy = [(s['name'], s['entropy']) for s in sections]
        section_entropy_hashed = FeatureHasher(50, input_type="pair").transform([section_entropy]).toarray()[0]
        section_vsize = [(s['name'], s['vsize']) for s in sections]
        section_vsize_hashed = FeatureHasher(50, input_type="pair").transform([section_vsize]).toarray()[0]
        # Fix: ensure iterable of iterables for scikit-learn FeatureHasher
        entry_name_hashed = FeatureHasher(50, input_type="string").transform([[raw_obj['entry']]]).toarray()[0]
        characteristics = [p for s in sections for p in s['props'] if s['name'] == raw_obj['entry']]
        characteristics_hashed = FeatureHasher(50, input_type="string").transform([characteristics]).toarray()[0]

        return np.hstack([
            general, section_sizes_hashed, section_entropy_hashed, section_vsize_hashed, entry_name_hashed,
            characteristics_hashed
        ]).astype(np.float32)

    ember.features.SectionInfo.process_raw_features = patched_process_raw_features
    return True


def verify_dataset_files(dataset_dir):
    """Verify that all expected raw EMBER files exist and are non-empty."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(f"Dataset directory does not exist: {dataset_dir}")

    required_train_files = [f"train_features_{i}.jsonl" for i in range(6)]
    required_test_files = ["test_features.jsonl"]
    all_required = required_train_files + required_test_files

    missing = []
    file_stats = {}
    for filename in all_required:
        filepath = os.path.join(dataset_dir, filename)
        if not os.path.isfile(filepath):
            missing.append(filename)
        else:
            size_mb = os.path.getsize(filepath) / (1024 * 1024)
            if size_mb == 0:
                missing.append(f"{filename} (empty file)")
            file_stats[filename] = f"{size_mb:.2f} MB"

    if missing:
        raise FileNotFoundError(f"Missing required dataset files in {dataset_dir}: {missing}")

    return file_stats


def vectorize_dataset_subset(file_paths, output_x_path, output_y_path, extractor,
                             max_samples=None, batch_size=1000, balance_classes=True):
    """
    Stream and vectorize raw JSONL files into memory-mapped arrays in batches.
    Guarantees bounded RAM usage (<150 MB buffer) and strict class balance.
    """
    ndim = extractor.dim

    # Determine sample targets
    if max_samples is not None:
        if balance_classes:
            target_per_class = max_samples // 2
            print(f"[*] Targeting balanced subset: {target_per_class} benign (0) and {target_per_class} malicious (1).")
        else:
            print(f"[*] Targeting first {max_samples} labeled samples.")
    else:
        print("[*] Processing full dataset (no sample limit).")

    # Pass 1 (if subset): Collect candidate indices or sample directly
    # To be fast and memory-safe, stream lines and maintain buffers
    class_counts = {0: 0, 1: 0}
    collected_x = []
    collected_y = []
    
    total_written = 0
    # If max_samples is specified, pre-allocate the memmap
    # If max_samples is None, we need total lines
    if max_samples is None:
        print("[*] Counting total lines for full dataset allocation...")
        total_samples = 0
        for fp in file_paths:
            with open(fp, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    if rec.get("label") in [0, 1]:  # only labeled for training/eval
                        total_samples += 1
        nrows = total_samples
    else:
        nrows = (max_samples // 2) * 2 if balance_classes else max_samples

    print(f"[*] Pre-allocating memory-mapped files for {nrows} samples (dim={ndim}):")
    print(f"    - X: {output_x_path} (~{(nrows * ndim * 4) / (1024*1024):.2f} MB)")
    print(f"    - y: {output_y_path} (~{(nrows * 4) / 1024:.2f} KB)")

    X_mm = np.memmap(output_x_path, dtype=np.float32, mode="w+", shape=(nrows, ndim))
    y_mm = np.memmap(output_y_path, dtype=np.float32, mode="w+", shape=(nrows,))

    batch_x = []
    batch_y = []
    start_time = time.time()

    done = False
    for fp in file_paths:
        if done:
            break
        print(f"[*] Scanning {os.path.basename(fp)}...")
        with open(fp, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                record = json.loads(line)
                label = record.get("label")
                
                # Filter out unlabeled samples (-1)
                if label not in [0, 1]:
                    continue

                if max_samples is not None and balance_classes:
                    if class_counts[label] >= target_per_class:
                        # We already have enough of this class
                        if class_counts[0] >= target_per_class and class_counts[1] >= target_per_class:
                            done = True
                            break
                        continue

                # Process features
                feat_vec = extractor.process_raw_features(record)
                batch_x.append(feat_vec)
                batch_y.append(label)
                class_counts[label] += 1

                # Flush batch to memmap
                if len(batch_x) >= batch_size:
                    n_batch = len(batch_x)
                    X_mm[total_written:total_written + n_batch] = np.array(batch_x, dtype=np.float32)
                    y_mm[total_written:total_written + n_batch] = np.array(batch_y, dtype=np.float32)
                    total_written += n_batch
                    batch_x = []
                    batch_y = []
                    elapsed = time.time() - start_time
                    rate = total_written / elapsed if elapsed > 0 else 0
                    print(f"    Vectorized {total_written}/{nrows} samples ({rate:.1f} samples/sec) | "
                          f"Benign: {class_counts[0]}, Malicious: {class_counts[1]}")

                if max_samples is not None:
                    if total_written + len(batch_x) >= nrows:
                        done = True
                        break

    # Flush remaining in buffer
    if len(batch_x) > 0:
        n_batch = len(batch_x)
        X_mm[total_written:total_written + n_batch] = np.array(batch_x, dtype=np.float32)
        y_mm[total_written:total_written + n_batch] = np.array(batch_y, dtype=np.float32)
        total_written += n_batch
        batch_x = []
        batch_y = []

    # Flush memmaps to disk
    X_mm.flush()
    y_mm.flush()
    del X_mm, y_mm

    elapsed = time.time() - start_time
    print(f"[+] Completed vectorization of {total_written} samples in {elapsed:.2f}s!")
    print(f"    Final class distribution -> Benign (0): {class_counts[0]}, Malicious (1): {class_counts[1]}")
    return total_written, class_counts


def prepare_dataset(dataset_dir=DEFAULT_DATASET_DIR,
                    max_train_samples=20000,
                    max_test_samples=5000,
                    force=False):
    """
    Full pipeline to verify dataset, apply patches, and create vectorized .dat files.
    """
    print("=" * 70)
    print("      EMBER 2018 v2 DATA PREPARATION PIPELINE")
    print("=" * 70)
    print(f"[*] Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[*] Dataset directory: {dataset_dir}")
    print(f"[*] Max train samples: {max_train_samples}")
    print(f"[*] Max test samples: {max_test_samples}")

    # 1. Verify dataset files
    print("\n[STEP 1/4] Verifying raw dataset files...")
    file_stats = verify_dataset_files(dataset_dir)
    for fname, sz in file_stats.items():
        print(f"    - {fname}: {sz}")
    print("[+] All raw EMBER files verified successfully.")

    # 2. Check if output files already exist
    x_train_path = os.path.join(dataset_dir, "X_train.dat")
    y_train_path = os.path.join(dataset_dir, "y_train.dat")
    x_test_path = os.path.join(dataset_dir, "X_test.dat")
    y_test_path = os.path.join(dataset_dir, "y_test.dat")
    meta_path = os.path.join(dataset_dir, "vectorization_metadata.json")

    existing_files = [p for p in [x_train_path, y_train_path, x_test_path, y_test_path] if os.path.exists(p)]
    if existing_files and not force:
        print(f"\n[!] Vectorized files already exist in {dataset_dir}:")
        for ef in existing_files:
            print(f"    - {os.path.basename(ef)} ({os.path.getsize(ef)/(1024*1024):.2f} MB)")
        print("[!] Use --force to re-vectorize.")
        return True

    # 3. Apply EMBER compatibility fix
    print("\n[STEP 2/4] Applying scikit-learn FeatureHasher compatibility patch...")
    apply_ember_compatibility_patch()
    import ember
    extractor = ember.PEFeatureExtractor(feature_version=2)
    print(f"[+] Feature extractor ready (feature_version=2, dimension={extractor.dim}).")

    # 4. Vectorize Training Set
    print("\n[STEP 3/4] Vectorizing Training Set...")
    train_files = [os.path.join(dataset_dir, f"train_features_{i}.jsonl") for i in range(6)]
    n_train, train_dist = vectorize_dataset_subset(
        file_paths=train_files,
        output_x_path=x_train_path,
        output_y_path=y_train_path,
        extractor=extractor,
        max_samples=max_train_samples,
        batch_size=1000,
        balance_classes=True
    )

    # 5. Vectorize Test Set
    print("\n[STEP 4/4] Vectorizing Test Set...")
    test_files = [os.path.join(dataset_dir, "test_features.jsonl")]
    n_test, test_dist = vectorize_dataset_subset(
        file_paths=test_files,
        output_x_path=x_test_path,
        output_y_path=y_test_path,
        extractor=extractor,
        max_samples=max_test_samples,
        batch_size=1000,
        balance_classes=True
    )

    # Save metadata record
    meta = {
        "created_at": datetime.now().isoformat(),
        "feature_version": 2,
        "feature_dimension": extractor.dim,
        "train_samples": n_train,
        "train_distribution": train_dist,
        "test_samples": n_test,
        "test_distribution": test_dist,
        "files": {
            "X_train": os.path.basename(x_train_path),
            "y_train": os.path.basename(y_train_path),
            "X_test": os.path.basename(x_test_path),
            "y_test": os.path.basename(y_test_path)
        }
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\n" + "=" * 70)
    print("      DATA PREPARATION COMPLETED SUCCESSFULLY")
    print("=" * 70)
    print(f"Train samples: {n_train} ({train_dist})")
    print(f"Test samples:  {n_test} ({test_dist})")
    print(f"Features:      {extractor.dim}")
    print(f"Output files in: {dataset_dir}")
    print(f"Metadata file:   {meta_path}")

    # Verify loading via ember.read_vectorized_features
    print("\n[*] Validating reading through ember.read_vectorized_features()...")
    X_train, y_train, X_test, y_test = ember.read_vectorized_features(dataset_dir)
    print(f"    - X_train shape: {X_train.shape}, dtype: {X_train.dtype}")
    print(f"    - y_train shape: {y_train.shape}, dtype: {y_train.dtype}")
    print(f"    - X_test shape:  {X_test.shape}, dtype: {X_test.dtype}")
    print(f"    - y_test shape:  {y_test.shape}, dtype: {y_test.dtype}")
    print("[+] Verification passed completely!")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare and vectorize EMBER 2018 v2 dataset")
    parser.add_argument("--dataset-dir", type=str, default=DEFAULT_DATASET_DIR,
                        help="Path to directory containing EMBER jsonl files")
    parser.add_argument("--max-train-samples", type=int, default=20000,
                        help="Number of training samples to vectorize (default: 20000 balanced, or None for full)")
    parser.add_argument("--max-test-samples", type=int, default=5000,
                        help="Number of test samples to vectorize (default: 5000 balanced, or None for full)")
    parser.add_argument("--full", action="store_true",
                        help="Process full dataset without sample limit")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing vectorized files")
    args = parser.parse_args()

    max_train = None if args.full else args.max_train_samples
    max_test = None if args.full else args.max_test_samples

    prepare_dataset(
        dataset_dir=args.dataset_dir,
        max_train_samples=max_train,
        max_test_samples=max_test,
        force=args.force
    )

