# EMBER 2018 Dataset Directory

This directory stores the EMBER (Endgame Malware BENGIN and Malicious Executable Representation) 2018 v2 dataset.

Raw dataset files (`.tar.bz2`, `.jsonl`, `.dat`) are excluded from version control due to large file sizes (>10 GB).

### Dataset Information:
- **Dataset**: EMBER 2018 v2
- **SHA-256 Archive**: `B6052EB8D350A49A8D5A5396FBE7D16CF42848B86FF969B77464434CF2997812`
- **Reference**: Anderson, H. S., & Roth, P. (2018). *EMBER: An Open Dataset for Training Static PE Classifiers*.

### Pre-trained Models:
Trained model weights (`best_model.pkl`, `random_forest.pkl`, `lightgbm.pkl`, `preprocessor.pkl`) and evaluation metrics are already included in the `models/` directory, so you can immediately run inference without re-downloading or re-vectorizing the entire 10GB dataset.
