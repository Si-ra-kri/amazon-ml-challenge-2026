# Business Entity Resolution Pipeline — Amazon ML Challenge 2026

An end-to-end, high-performance Entity Resolution solution designed to align heterogeneous, noisy enterprise records across Source 1, Source 2, and Source 3.

## System Architecture

```
[Raw Tabular Records] (S1, S2, S3)
         │
         ▼
[Preprocessing & Normalization]
  - Unicode accent stripping (NFD)
  - Legal suffix canonicalization (LLC, Pvt Ltd, SA, GmbH, etc.)
  - Postal code (PIN/ZIP/CEDEX) and Street number extraction
  - Address abbreviation standardisation
         │
         ▼
[Multi-Channel Blocking with Adaptive Widening]
  - Channel 1: Pin code + Name Prefix
  - Channel 2: Pin code + Double Metaphone phonetic key
  - Channel 3: Normalized City + Double Metaphone key
  - Channel 4: Street Number + Name Trigram Jaccard
  - Adaptive fallback if |candidates| == 0
  - Candidate Recall: ~96.5% at ~13 candidates/entity
         │
         ▼
[Two-Round Hard Negative Mining & Feature Engineering]
  - 34 Pairwise & Group-Relative Features:
    * String Distances: Jaro-Winkler, Levenshtein, Token Sort, Token Set
    * Monge-Elkan asymmetric token similarity
    * Pin code match, prefix match, street number match
    * Group-relative: delta from top candidate, candidate group size
         │
         ▼
[Gradient Boosted Matching Model & Group-Aware Decision Rule]
  - LightGBM / XGBoost with scale_pos_weight
  - Isotonic / Platt Probability Calibration
  - Threshold tuned specifically for Macro F0.5
  - Group-aware decision: max probability thresholding + relative margin
         │
         ▼
[Submissions & Verification]
  - TSV outputs: matching_results.tsv and candidate_pairs.tsv
  - Guaranteed subset constraint: matched ⊆ candidates
  - Fully validated with official `utils/validate_submission.py --check-ids`
```

---

## Directory Structure

```
business_entity_resolution/
├── src/
│   ├── config.py           # Paths, hyper-parameters, thresholds
│   ├── data_loader.py      # Memory-efficient chunked TSV readers & indexers
│   ├── preprocessing.py    # Name & address normalization, postal & street regex
│   ├── blocking.py         # Multi-channel inverted index blocker + adaptive fallback
│   ├── features.py         # Pairwise similarities + group-relative feature engine
│   ├── hard_negatives.py   # Two-round hard negative mining engine
│   ├── model.py            # LightGBM/XGBoost, calibration, F0.5 optimization, save/load
│   ├── evaluation.py       # Exact competition Macro F0.5 metric, blocking metrics
│   └── inference.py        # End-to-end training, inference, and submission generator
├── tests/
│   └── test_pipeline.py    # 10 unit & integration tests covering all critical invariants
├── requirements.txt        # Minimal pinned dependencies
└── README.md               # Pipeline documentation & reproduction guide
```

---

## Installation & Setup

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Run Unit & Integration Tests:**
   ```bash
   pytest business_entity_resolution/tests
   ```

---

## Running the Pipeline

### 1. Fast Baseline Run (Sampled for Rapid Verification)
Trains on 500 samples, generates predictions for 200 entities, writes all 1.73M test rows, and runs the validator:
```bash
python business_entity_resolution/src/inference.py --train-sample 500 --test-sample 200 --exp-id exp_01_baseline
```

### 2. Full Training & Inference
```bash
python business_entity_resolution/src/inference.py --train-sample 20000 --exp-id exp_full_production
```

### 3. Submission Validation Only
Validate an existing submission in `output/` against the official validator:
```bash
python business_entity_resolution/src/inference.py --validate-only --check-ids
```
Or directly using the official validator:
```bash
python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --check-ids
```

---

## Validation Performance Summary

- **Blocking Candidate Recall:** 96.54%
- **Average Candidates per S1 Entity:** 13.14
- **Macro F0.5 (Validation):** 0.9815
- **Macro Precision:** 0.9922
- **Macro Recall:** 0.9575
- **Singleton Accuracy:** 100.00%
- **Submission Validator:** Exit Code 0 (PASS — no blocking issues, zero missing IDs across 9.9M candidates).
