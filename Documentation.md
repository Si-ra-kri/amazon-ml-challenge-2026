# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:**  
**Team Members:**  
**Submission Date:**  

---

## 1. Executive Summary

We developed an end-to-end, high-recall, precision-optimized Entity Resolution pipeline combining multi-channel inverted index blocking with adaptive widening, two-round hard negative mining, 34-dimensional pairwise & group-relative feature engineering, and calibrated Gradient Boosted Trees (LightGBM). On out-of-fold cross-validation, our pipeline achieves a candidate blocking recall of **92.99%** with **29.71 candidates per entity**, an optimal Macro F0.5 of **0.9503** (Macro Precision: **0.9731**, Macro Recall: **0.9091**, Singleton Accuracy: **94.55%**), and scales seamlessly across the full test set of **1,732,544** Source 1 entities against **9.97 million** candidate records from Sources 2 and 3, passing the official competition submission validator with zero errors.

---

## 2. Methodology

### 2.1 Problem Analysis
Exploratory Data Analysis across the 2,206,821 Source 1 entities in the training ground truth and test sources revealed key structural characteristics:
1. **Multi-Source Matches & Singleton Proportion:** In the training ground truth, **94.42%** of entities (2,083,574) have matches in Source 2 and/or Source 3, averaging **3.67 matching records per non-singleton entity** (3.46 matches per entity overall). True singletons (entities with zero matches) account for **5.58%** (123,247 entities). Similarly, on the test set, our model predicts matches for **91.38%** (1,583,224 entities, avg 3.72 matches per entity) and predicts singletons for **8.62%** (149,320 entities).
2. **Asymmetric Precision Weighting ($\beta = 0.5$):** Because Macro F0.5 weights precision twice as heavily as recall, any false positive prediction severely degrades the entity's score. On singletons, a false positive match drops the score from 1.0 to 0.0; on matching entities, merging non-matching distractor branches drops precision significantly.
3. **Text Noise & Multi-lingual Heterogeneity:** Business names and addresses exhibit severe accent variations, legal form variations (`Corp`, `LLC`, `Pvt Ltd`, `SARL`, `GmbH`), concatenated or missing postal codes, and multilingual transliterations (e.g. English, French, Hindi, Bengali).
4. **Extreme Comparison Scale ($1.73 \times 10^6 \times 10^7 \approx 1.7 \times 10^{13}$ pairs):** Full pairwise comparison is intractable. Multi-channel blocking with tight candidate bounds ($k \le 35$) is required to achieve $>92\%$ recall while bounding comparisons to $\sim 6 \times 10^7$ pairs.

### 2.2 Solution Strategy
**Approach Type:** Multi-Channel Blocking + Two-Round Hard Negative Mining + Calibrated Gradient Boosting + Group-Aware Decision Rule  
**Core Technical Contributions:**
- **Adaptive Multi-Channel Inverted Index:** Combines spatial/postal proximity, phonetic name representations (Double Metaphone), and token/character n-grams. When primary high-specificity channels yield fewer than `min_candidates=2`, an adaptive trigram fallback broadens the search, ensuring virtually zero candidate dropouts across 1.73M entities.
- **Group-Relative Contrastive Features:** For each candidate in an entity's candidate pool, relative features are computed (e.g., probability margin to top candidate, similarity rank, z-score within candidate set), preventing spurious low-probability merges in ambiguous candidate clusters.

---

## 3. Candidate Generation (Blocking)

To reduce the comparison space from $10^{13}$ to $\approx 6 \times 10^7$ candidate pairs without sacrificing true matches:

- **Blocking keys used:**
  1. `pin_prefix4_name_prefix3`: Pin code prefix (4 digits) + first 3 characters of normalized business name.
  2. `pin_metaphone`: 6-digit postal code + primary Double Metaphone phonetic key of the business name.
  3. `city_metaphone`: Canonicalized city name + Double Metaphone phonetic key (recovers records where postal code was missing or corrupted).
  4. `street_trigram`: Extracted street house number + first word trigram overlap.
  5. `adaptive_fallback`: Trigram Jaccard indexing on normalized name + city when primary channels return empty.
- **Candidate pairs generated:** 
  - **Test Set:** **60,527,607 total candidate pairs** across 1,732,544 entities, averaging **34.94 candidates per S1 entity** (1,732,543 non-empty rows, 1 empty row).
  - **Validation Set:** Averaged **29.71 candidates per entity** with a candidate blocking recall of **92.99%** on ground truth.
- **How true matches were retained:**
  Orthogonal indexing across spatial (PIN, city, street number) and phonetic/textual dimensions (Double Metaphone, name prefix, character trigrams), complemented by adaptive widening when strict channels produce empty pools.

---

## 4. Matching Model

### Features Used (34 Total Features)
- **Name Features:**
  - Jaro-Winkler similarity on normalized business names.
  - Levenshtein ratio, token sort ratio, and token set ratio.
  - Longest Common Subsequence (LCS) ratio and character 3-gram cosine similarity.
  - Phonetic match indicators (Double Metaphone and Soundex equality).
  - Exact token intersection count & Jaccard index.
- **Address & Geo Features:**
  - Postal code exact match (binary), presence/absence indicators, and mismatch penalties.
  - Street number exact match and mismatch indicators.
  - City token containment and string similarity.
  - Full address token overlap and Jaccard similarity.
- **Meta & Group-Relative Features:**
  - Country match indicator, source indicator flags (`is_source2`, `is_source3`).
  - Name and address length differences and ratios.
  - Delta score to highest-scoring candidate in the entity's candidate pool (`group_score_gap_to_top`).
  - Score margin to next candidate (`group_score_gap_to_next`) and candidate group z-score.

### Model Architecture & Optimization
- **Model Type:** LightGBM Classifier trained with two-round hard negative mining to handle extreme class imbalance.
- **Probability Calibration:** Sigmoid Platt Scaling (`CalibratedClassifierCV`) ensuring predicted values represent true match probabilities $P(\text{Match} \mid \mathbf{x})$.
- **Threshold & Multi-Match Decision Rule:**
  Grid sweep on out-of-fold validation set directly optimizing Macro F0.5:
  $$F_{0.5} = \frac{(1 + 0.5^2) \cdot P \cdot R}{0.5^2 \cdot P + R} = \frac{1.25 \cdot P \cdot R}{0.25 \cdot P + R}$$
  Optimal decision threshold identified: $\tau^* = 0.68$ combined with relative margin gating $\Delta = (P_{\max}(e_1) - P(\text{pair})) \le 0.30$. A candidate is accepted as a match if and only if $P(\text{pair}) \ge \tau^*$ AND $(P_{\max}(e_1) - P(\text{pair})) \le \Delta$, accommodating legitimate multi-source matches (both S2 and S3) while suppressing distant distractors.

---

## 5. Results & Error Analysis

Out-of-fold cross-validation metrics evaluated across 1,000 validation entities (including 55 true singletons):

- **Macro F0.5 Score:** **0.9503**
- **Macro Precision:** **0.9731**
- **Macro Recall:** **0.9091**
- **Candidate Blocking Recall:** **92.99%** (with avg 29.71 candidates/entity)
- **Singleton Accuracy:** **94.55%** (52 of 55 true singletons correctly predicted empty; false merge count = 24)
- **Common False Positives (Wrong Merges):**
  - **Franchise / Chain Branches:** Distinct retail outlets of identical brands (e.g. banking branches, convenience stores) that share city names but differ slightly in street addresses. Mitigated by street number mismatch penalties and strict probability margin gating.
- **Common False Negatives (Missed Matches):**
  - **Compound Corruptions:** Records where the postal code was omitted or corrupted concurrently with heavy transliteration or extreme name abbreviation. Mitigated by the phonetic metaphone and city-level blocking channel.

---

## 6. Conclusion
The proposed architecture delivers a scalable, mathematically principled solution tailored to the precision-weighted Macro F0.5 objective. By coupling high-coverage multi-channel blocking with calibrated gradient boosted trees, two-round hard negative mining, and group-relative margin gating, the pipeline achieves an empirical Macro F0.5 of 0.9503 on out-of-fold validation while processing 1.73M test entities in streaming memory-safe chunks and passing the official submission validator with zero defects.

---

## Appendix

### A. Code Artefacts
All code resides under `business_entity_resolution/`:
- `src/config.py`: Centralized configuration, hyper-parameters, and file paths.
- `src/preprocessing.py`: Regex extraction for PINs, street numbers, legal suffix removal, and unicode normalization.
- `src/blocking.py`: Multi-channel inverted index with adaptive widening.
- `src/features.py`: 34-feature pairwise & group-relative extraction engine.
- `src/hard_negatives.py`: Two-round hard negative miner.
- `src/model.py`: LightGBM classifier, probability calibrator, and F0.5 threshold optimizer.
- `src/evaluation.py`: Exact competition Macro F0.5 and candidate recall evaluator.
- `src/inference.py`: Pipeline runner with cross-validation and export.
- `run_accelerated.py`: Memory-safe streaming production runner with auto-resume across 1.73M test entities.
- `tests/test_pipeline.py`: Comprehensive test suite verifying all invariants.

**Entry points:**
- Run Pipeline / K-Fold Evaluation: `python -m business_entity_resolution.src.inference --train-sample 5000`
- Run Submission Validator: `python utils/validate_submission.py --matching output/matching_results.tsv --candidate output/candidate_pairs.tsv --test-dir dataset/test`
- Run Test Suite: `pytest business_entity_resolution/tests`

### B. Submission Verification Log
Actual output from running `utils/validate_submission.py` on the complete generated test outputs:

```
ML Challenge 2026 – submission validator
  test dir: dataset/test
  required S1 entities: 1732544
  matching_results.tsv: 1732544 rows (149320 empty, 1583224 non-empty).
  candidate_pairs.tsv: 1732544 rows (1 empty, 1732543 non-empty).

PASS – no blocking issues found. Safe to submit.
```
