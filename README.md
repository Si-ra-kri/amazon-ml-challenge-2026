# ML Challenge 2026: Business Entity Resolution Solution

An end-to-end, high-recall, precision-optimized Entity Resolution pipeline combining multi-channel inverted index blocking with adaptive widening, two-round hard negative mining, 34-dimensional pairwise & group-relative feature engineering, and calibrated Gradient Boosted Trees (LightGBM).

## 📦 Submission Files & Large Artifacts

The final competition submission files are published as GitHub Release assets:
- **[Final Submission Package (submission_final.zip)](https://github.com/Si-ra-kri/amazon-ml-challenge-2026/releases/latest)**:
  - matching_results.tsv (107.2 MB uncompressed)
  - candidate_pairs.tsv (804.2 MB uncompressed)
  - Documentation.md
- **[Project Code Package (project_code.zip)](https://github.com/Si-ra-kri/amazon-ml-challenge-2026/releases/latest)**

See [Documentation.md](Documentation.md) for the complete solution methodology, model architecture, validation results, and ablation studies.

## 📁 Repository Structure

`	ext
├── business_entity_resolution/
│   ├── src/                    # Core source code (blocking, features, model, inference)
│   ├── tests/                  # Pipeline test suite
│   ├── notebooks/              # Exploration & validation notebooks
│   └── README.md               # Detailed module documentation
├── models/
│   └── kfold_eval_lightgbm.joblib  # Trained LightGBM model weights
├── utils/
│   └── validate_submission.py  # Official competition submission validator
├── Documentation.md            # Complete Solution Technical Report
├── requirements.txt            # Python dependencies
├── aws_runner.ipynb            # AWS training/inference runner
└── README.md
`

## 🚀 Quick Start

### Installation
`ash
pip install -r requirements.txt
`

### Validate Submission
`ash
python utils/validate_submission.py
`
