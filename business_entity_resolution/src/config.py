"""
Configuration and Hyperparameters for Business Entity Resolution Pipeline
Amazon ML Challenge 2026
"""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import List, Tuple


def find_dataset_dir() -> Path:
    """Auto-detect dataset directory across workspace, parent, or environment variable."""
    env_dir = os.environ.get("DATASET_DIR")
    if env_dir and os.path.isdir(env_dir):
        return Path(env_dir).resolve()

    candidates = [
        Path("dataset"),
        Path("../dataset"),
        Path("../../dataset"),
        Path("C:/Users/siva/Downloads/Amazon ML Dataset/student_resource/dataset"),
    ]
    for candidate in candidates:
        if candidate.is_dir() and (candidate / "train" / "train_source1.tsv").is_file():
            return candidate.resolve()

    # Fallback to local dataset directory
    return Path("dataset").resolve()


def find_output_dir() -> Path:
    """Auto-detect or create output directory."""
    candidates = [
        Path("output"),
        Path("../output"),
        Path("../../output"),
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    # Default create output in current dir
    out = Path("output").resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


@dataclass
class PipelineConfig:
    # Paths
    dataset_dir: Path = field(default_factory=find_dataset_dir)
    output_dir: Path = field(default_factory=find_output_dir)
    models_dir: Path = field(
        default_factory=lambda: Path("models").resolve()
    )
    experiment_log_path: Path = field(
        default_factory=lambda: Path("experiments/experiment_log.csv").resolve()
    )

    # Cross-validation
    random_seed: int = 42
    n_splits: int = 5

    # Blocking Configuration
    min_candidates: int = 2
    max_candidates: int = 35
    token_blocking_min_len: int = 3
    char_ngram_n: int = 3
    blocking_tfidf_max_features: int = 50000

    # Negative Mining
    hard_negatives_ratio_r1: int = 4
    hard_negatives_ratio_r2: int = 2

    # Model Hyperparameters
    lgb_params: dict = field(
        default_factory=lambda: {
            "objective": "binary",
            "metric": "binary_logloss",
            "boosting_type": "gbdt",
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": 6,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_estimators": 250,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }
    )

    # Calibration & Decision Threshold
    calibration_method: str = "sigmoid"  # Platt scaling
    default_threshold: float = 0.58
    relative_score_margin: float = 0.30  # Candidate must be within margin of top score

    # Feature List
    feature_names: List[str] = field(
        default_factory=lambda: [
            # Name features
            "name_exact_match",
            "name_clean_exact_match",
            "name_suffix_stripped_match",
            "name_token_jaccard",
            "name_token_dice",
            "name_levenshtein_ratio",
            "name_jaro_winkler",
            "name_lcs_ratio",
            "name_soundex_match",
            "name_metaphone_match",
            "name_char3gram_cosine",
            # Address features
            "addr_exact_match",
            "addr_clean_exact_match",
            "addr_token_jaccard",
            "addr_token_dice",
            "addr_levenshtein_ratio",
            "addr_jaro_winkler",
            "postal_exact_match",
            "postal_missing_either",
            "postal_both_present_match",
            "postal_both_present_mismatch",
            "street_num_match",
            "street_num_missing_either",
            # Meta features
            "country_match",
            "is_source2",
            "is_source3",
            "name_len_diff",
            "name_len_ratio",
            "addr_len_diff",
            "s1_candidate_count",
            # Group-relative features
            "group_sim_rank",
            "group_score_gap_to_top",
            "group_score_gap_to_next",
            "group_score_zscore",
        ]
    )


# Default singleton instance
config = PipelineConfig()
