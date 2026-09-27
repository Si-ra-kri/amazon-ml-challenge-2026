"""
End-to-End Pipeline Runner, Train-Validation-Test Inference, and Submission Generator
Amazon ML Challenge 2026

Coordinates:
1. Data loading and K-fold split
2. Candidate generation via Multi-Channel Blocking + Adaptive Widening
3. Pairwise & Group-relative feature extraction
4. Two-round hard negative mining and model training
5. Threshold optimization for Macro F0.5
6. Test set inference with full constraint adherence
7. Formatted TSV export (matching_results.tsv and candidate_pairs.tsv)
8. Automated validation via utils/validate_submission.py
"""

import argparse
import csv
from datetime import datetime
import os
from pathlib import Path
import subprocess
import sys
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np
import pandas as pd

from .blocking import MultiChannelBlocker
from .config import config
from .data_loader import create_kfold_splits, load_ground_truth, load_source_tsv
from .evaluation import compute_blocking_metrics, evaluate_macro_f05
from .features import build_candidate_feature_matrix
from .hard_negatives import mine_hard_negatives_round1, mine_hard_negatives_round2
from .model import EntityMatchingModel, optimize_threshold
from .preprocessing import preprocess_record


def write_submission_tsv(
    output_path: Path,
    s1_ids: List[str],
    id_mapping: Dict[str, Iterable[str]],
    is_candidate_file: bool = False,
):
    """Write submission TSV strictly enforcing format contracts:
    - Tab-separated
    - Header: source1_entity_id\tmatched_entity_ids (or candidate_entity_ids)
    - One row per S1 entity
    - No quoting, comma-separated with no spaces
    - Empty string for singletons/no matches
    - Preserves exact ordering of s1_ids
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header_col = "candidate_entity_ids" if is_candidate_file else "matched_entity_ids"

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="\t", quoting=csv.QUOTE_NONE, escapechar="\\")
        writer.writerow(["source1_entity_id", header_col])

        for s1_id in s1_ids:
            matches = id_mapping.get(s1_id, [])
            # Deduplicate while preserving order, filter out empty
            seen = set()
            clean_matches = []
            for m in matches:
                m_str = str(m).strip()
                if m_str and m_str not in seen and m_str != s1_id:
                    seen.add(m_str)
                    clean_matches.append(m_str)
            joined = ",".join(clean_matches)
            writer.writerow([s1_id, joined])


def log_experiment(
    log_path: Path,
    exp_id: str,
    blocking_method: str,
    candidate_recall: float,
    feature_version: str,
    model_type: str,
    threshold: float,
    local_macro_f05: float,
    notes: str = "",
    public_score: str = "",
):
    """Append experiment run to experiments/experiment_log.csv."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = log_path.is_file()

    fieldnames = [
        "experiment_id",
        "timestamp",
        "blocking_method",
        "candidate_recall",
        "feature_version",
        "model_type",
        "threshold",
        "local_macro_f05",
        "public_leaderboard_score",
        "notes",
    ]

    with open(log_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            "experiment_id": exp_id,
            "timestamp": datetime.now().isoformat(),
            "blocking_method": blocking_method,
            "candidate_recall": f"{candidate_recall:.4f}",
            "feature_version": feature_version,
            "model_type": model_type,
            "threshold": f"{threshold:.4f}",
            "local_macro_f05": f"{local_macro_f05:.4f}",
            "public_leaderboard_score": public_score,
            "notes": notes,
        })


def run_pipeline(
    train_sample_size: Optional[int] = 5000,
    test_sample_size: Optional[int] = None,
    model_type: str = "lightgbm",
    exp_id: str = "exp_01_baseline",
) -> Tuple[EntityMatchingModel, Dict[str, float]]:
    """Execute complete reproducible train, validation, and inference pipeline."""
    print("=" * 70)
    print(f"Starting Entity Resolution Pipeline: {exp_id} ({model_type})")
    print("=" * 70)

    dataset_dir = config.dataset_dir
    train_dir = dataset_dir / "train"
    test_dir = dataset_dir / "test"

    # 1. Load Ground Truth and Source 1
    print(f"[1/6] Loading Source 1 and Ground Truth (sample size: {train_sample_size})...")
    gt_all = load_ground_truth(train_dir / "train_ground_truth.tsv", nrows=train_sample_size)
    needed_s1 = set(gt_all.keys())

    s1_records = {}
    with open(train_dir / "train_source1.tsv", "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            eid = parts[0]
            if eid in needed_s1:
                raw_dict = dict(zip(header, parts))
                s1_records[eid] = preprocess_record(raw_dict)
                if len(s1_records) == len(needed_s1):
                    break

    common_s1_ids = list(s1_records.keys())
    ground_truth = {eid: gt_all[eid] for eid in common_s1_ids}

    # 2. Collect targets and load candidate pool (S2 and S3)
    needed_targets = set()
    for mset in ground_truth.values():
        needed_targets.update(mset)

    print(f"[2/6] Loading candidate sources (targeting {len(needed_targets)} true matches)...")
    target_records = {}

    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        p = train_dir / fname
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for idx, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                # Include all needed true matches plus distractor pool
                if eid in needed_targets or idx < (train_sample_size or 5000):
                    raw_dict = dict(zip(header, parts))
                    target_records[eid] = preprocess_record(raw_dict)

    print(f"Indexed {len(target_records)} target candidates from S2 and S3.")

    # 3. K-Fold Cross Validation Split
    print("[3/6] Setting up Stratified K-Fold Split...")
    splits = create_kfold_splits(common_s1_ids, ground_truth, n_splits=config.n_splits)
    train_s1_ids, val_s1_ids = splits[0]
    print(f"Fold 0: Train S1 = {len(train_s1_ids)}, Val S1 = {len(val_s1_ids)}")

    # 4. Multi-Channel Blocking on Train and Val
    print("[4/6] Executing Multi-Channel Blocking...")
    blocker = MultiChannelBlocker(
        min_candidates=config.min_candidates,
        max_candidates=config.max_candidates,
    )
    blocker.index_targets(target_records.values())
    blocker.prune_frequent_tokens()

    train_s1_recs = [s1_records[eid] for eid in train_s1_ids]
    val_s1_recs = [s1_records[eid] for eid in val_s1_ids]

    train_candidates = blocker.block_all(train_s1_recs)
    val_candidates = blocker.block_all(val_s1_recs)

    # Evaluate blocking recall on validation split
    val_gt = {eid: ground_truth[eid] for eid in val_s1_ids}
    val_cand_sets = {eid: set(cids) for eid, cids in val_candidates.items()}
    blocking_metrics = compute_blocking_metrics(val_gt, val_cand_sets)
    cand_recall = blocking_metrics["candidate_recall"]
    print(f"Validation Candidate Recall: {cand_recall * 100:.2f}%")
    print(f"Average Candidates per Entity: {blocking_metrics['avg_candidates_per_entity']:.2f}")

    # 5. Feature Extraction and Hard Negative Mining
    print("[5/6] Feature Engineering and Two-Round Hard Negative Training...")
    # Mine Round 1 Hard Negatives for training fold
    r1_negs = mine_hard_negatives_round1(
        s1_records=s1_records,
        candidate_records=target_records,
        candidate_pairs=train_candidates,
        ground_truth=ground_truth,
        ratio=config.hard_negatives_ratio_r1,
    )

    # Form positive pairs
    pos_pairs = []
    for s1_id in train_s1_ids:
        for cid in ground_truth.get(s1_id, set()):
            if cid in target_records:
                pos_pairs.append((s1_id, cid))

    all_train_pairs = list(set(pos_pairs + r1_negs))
    train_pair_dict = {}
    for s1_id, cid in all_train_pairs:
        train_pair_dict.setdefault(s1_id, []).append(cid)

    X_train_df, train_pair_order = build_candidate_feature_matrix(
        s1_records=s1_records,
        candidate_records=target_records,
        candidate_pairs=train_pair_dict,
    )
    y_train = np.array([
        1 if cid in ground_truth.get(s1_id, set()) else 0
        for s1_id, cid in train_pair_order
    ])

    # Validation feature matrix
    X_val_df, val_pair_order = build_candidate_feature_matrix(
        s1_records=s1_records,
        candidate_records=target_records,
        candidate_pairs=val_candidates,
    )
    y_val = np.array([
        1 if cid in ground_truth.get(s1_id, set()) else 0
        for s1_id, cid in val_pair_order
    ])

    # Fit Model Round 1
    model = EntityMatchingModel(model_type=model_type)
    model.fit(X_train_df.values, y_train, X_val_df.values, y_val)

    # Mine Round 2 Hard Negatives
    r2_negs = mine_hard_negatives_round2(
        model=model,
        feature_matrix=X_train_df.values,
        pair_ids=train_pair_order,
        ground_truth=ground_truth,
        ratio=config.hard_negatives_ratio_r2,
        existing_neg_set=set(r1_negs),
    )

    if r2_negs:
        print(f"Augmenting training set with {len(r2_negs)} Round 2 hard negatives...")
        all_train_pairs = list(set(all_train_pairs + r2_negs))
        train_pair_dict = {}
        for s1_id, cid in all_train_pairs:
            train_pair_dict.setdefault(s1_id, []).append(cid)

        X_train_df, train_pair_order = build_candidate_feature_matrix(
            s1_records=s1_records,
            candidate_records=target_records,
            candidate_pairs=train_pair_dict,
        )
        y_train = np.array([
            1 if cid in ground_truth.get(s1_id, set()) else 0
            for s1_id, cid in train_pair_order
        ])
        model.fit(X_train_df.values, y_train, X_val_df.values, y_val)

    # Calibrated probability inference on validation pairs
    val_probs = model.predict_proba(X_val_df.values)
    val_pair_probs = {pair: prob for pair, prob in zip(val_pair_order, val_probs)}

    # Optimize threshold for Macro F0.5
    best_th, best_f05, sweep_log = optimize_threshold(
        model=model,
        candidate_pairs=val_candidates,
        pair_probabilities=val_pair_probs,
        ground_truth=val_gt,
        threshold_range=(0.40, 0.90, 0.02),
        relative_margin=config.relative_score_margin,
    )
    val_metrics = sweep_log[best_th]
    print(f"Optimal Threshold: {best_th}")
    print(f"Validation Macro F0.5: {best_f05:.4f}")
    print(f"Validation Precision: {val_metrics['macro_precision']:.4f}")
    print(f"Validation Recall: {val_metrics['macro_recall']:.4f}")
    print(f"Singleton Accuracy: {val_metrics['singleton_accuracy'] * 100:.2f}%")

    # Log to experiment log
    log_experiment(
        log_path=config.experiment_log_path,
        exp_id=exp_id,
        blocking_method="multi_channel_with_adaptive_widening",
        candidate_recall=cand_recall,
        feature_version="v1_pairwise_and_group_relative_34feat",
        model_type=model_type,
        threshold=best_th,
        local_macro_f05=best_f05,
        notes=f"Train sample: {train_sample_size}, Val F0.5: {best_f05:.4f}",
    )

    # Save model artifact
    config.models_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = config.models_dir / f"{exp_id}_{model_type}.joblib"
    model.save(str(model_save_path))

    return model, val_metrics


def run_test_inference(
    model: EntityMatchingModel,
    test_sample_size: Optional[int] = None,
    output_dir: Optional[Path] = None,
):
    """Run end-to-end inference on test data and generate submission files."""
    print("=" * 70)
    print("Running Test Set Inference...")
    print("=" * 70)

    dataset_dir = config.dataset_dir
    test_dir = dataset_dir / "test"
    out_dir = output_dir or config.output_dir

    # Load all required Test Source 1 IDs to guarantee complete submission rows
    print("Reading all test Source 1 entity IDs...")
    with open(test_dir / "test_source1.tsv", "r", encoding="utf-8") as f:
        next(f)
        all_test_s1_ids = [line.split("\t")[0] for line in f]
    print(f"Total required Test S1 entities: {len(all_test_s1_ids):,}")

    # Load Test Source 1 records to process
    n_to_process = test_sample_size if test_sample_size is not None else len(all_test_s1_ids)
    test_s1_df = load_source_tsv(test_dir / "test_source1.tsv", nrows=n_to_process)
    print(f"Processing inference for {len(test_s1_df):,} test Source 1 entities...")

    test_s1_records = {
        row["entity_id"]: preprocess_record(row)
        for row in test_s1_df.to_dict(orient="records")
    }

    # Index Test Sources 2 and 3
    print("Indexing Test Source 2 and Source 3...")
    blocker = MultiChannelBlocker(
        min_candidates=config.min_candidates,
        max_candidates=config.max_candidates,
    )

    target_records = {}
    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        p = test_dir / fname
        if not p.is_file():
            continue
        print(f"Reading {fname}...")
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for idx, line in enumerate(f):
                if test_sample_size and idx >= (test_sample_size * 4):
                    break
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                raw_dict = dict(zip(header, parts))
                p_rec = preprocess_record(raw_dict)
                target_records[eid] = p_rec

    print(f"Indexing {len(target_records)} candidates into multi-channel blocker...")
    blocker.index_targets(target_records.values())
    blocker.prune_frequent_tokens()

    # Block test entities
    print("Generating candidate pairs for test entities...")
    test_candidates = blocker.block_all(test_s1_records.values())

    # Build features
    print("Extracting test feature matrix...")
    X_test_df, test_pair_order = build_candidate_feature_matrix(
        s1_records=test_s1_records,
        candidate_records=target_records,
        candidate_pairs=test_candidates,
    )

    # Predict
    print(f"Scoring {len(test_pair_order)} candidate pairs...")
    if len(test_pair_order) > 0:
        probs = model.predict_proba(X_test_df.values)
        pair_probs = {pair: prob for pair, prob in zip(test_pair_order, probs)}
    else:
        pair_probs = {}

    predicted_matches = model.predict_matches(
        candidate_pairs=test_candidates,
        pair_probabilities=pair_probs,
        threshold=model.threshold,
        relative_margin=model.relative_margin,
    )

    # Write output files
    matching_tsv = out_dir / "matching_results.tsv"
    candidate_tsv = out_dir / "candidate_pairs.tsv"

    print(f"Writing {candidate_tsv} ({len(all_test_s1_ids):,} rows)...")
    write_submission_tsv(
        output_path=candidate_tsv,
        s1_ids=all_test_s1_ids,
        id_mapping=test_candidates,
        is_candidate_file=True,
    )

    print(f"Writing {matching_tsv} ({len(all_test_s1_ids):,} rows)...")
    write_submission_tsv(
        output_path=matching_tsv,
        s1_ids=all_test_s1_ids,
        id_mapping=predicted_matches,
        is_candidate_file=False,
    )

    # Also copy to root output/ if out_dir is different
    root_out = Path("output").resolve()
    if root_out != out_dir.resolve():
        root_out.mkdir(parents=True, exist_ok=True)
        write_submission_tsv(
            output_path=root_out / "candidate_pairs.tsv",
            s1_ids=all_test_s1_ids,
            id_mapping=test_candidates,
            is_candidate_file=True,
        )
        write_submission_tsv(
            output_path=root_out / "matching_results.tsv",
            s1_ids=all_test_s1_ids,
            id_mapping=predicted_matches,
            is_candidate_file=False,
        )

    print("Submission files written successfully!")


def run_local_validation(matching_path: Path, candidate_path: Path, test_dir: Path, check_ids: bool = False):
    """Run utils/validate_submission.py to ensure zero validation errors."""
    validator_path = Path("utils/validate_submission.py")
    if not validator_path.is_file():
        candidates = [
            Path("../utils/validate_submission.py"),
            Path("C:/Users/siva/Downloads/Amazon ML Dataset/student_resource/utils/validate_submission.py"),
        ]
        for c in candidates:
            if c.is_file():
                validator_path = c
                break

    if not validator_path.is_file():
        print("Warning: validate_submission.py not found, skipping validation check.")
        return

    print("=" * 70)
    print("Running Competition Submission Validator...")
    print("=" * 70)
    cmd = [
        sys.executable,
        str(validator_path),
        "--matching",
        str(matching_path),
        "--candidate",
        str(candidate_path),
        "--test-dir",
        str(test_dir),
    ]
    if check_ids:
        cmd.append("--check-ids")
    res = subprocess.run(cmd, capture_output=True, text=True, errors="replace")
    print(res.stdout)
    if res.stderr:
        print("Errors / Warnings:")
        print(res.stderr)

    if res.returncode == 0:
        print(">>> SUCCESS: validate_submission.py PASSED (Exit Code 0) <<<")
    else:
        print(f">>> FAILED: Validator returned exit code {res.returncode} <<<")


def main():
    parser = argparse.ArgumentParser(description="Business Entity Resolution Pipeline")
    parser.add_argument("--train-sample", type=int, default=1000, help="Train sample size")
    parser.add_argument("--test-sample", type=int, default=500, help="Test sample size (None for full)")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"])
    parser.add_argument("--exp-id", type=str, default="exp_01_baseline")
    parser.add_argument("--validate-only", action="store_true", help="Only run submission validator")
    parser.add_argument("--check-ids", action="store_true", help="Enable strict ID existence checks in validator")
    args = parser.parse_args()

    matching_path = config.output_dir / "matching_results.tsv"
    candidate_path = config.output_dir / "candidate_pairs.tsv"

    if args.validate_only:
        run_local_validation(matching_path, candidate_path, config.dataset_dir / "test", check_ids=args.check_ids)
        return

    model, metrics = run_pipeline(
        train_sample_size=args.train_sample,
        model_type=args.model_type,
        exp_id=args.exp_id,
    )

    run_test_inference(
        model=model,
        test_sample_size=args.test_sample,
        output_dir=config.output_dir,
    )

    run_local_validation(matching_path, candidate_path, config.dataset_dir / "test", check_ids=args.check_ids)


if __name__ == "__main__":
    main()
