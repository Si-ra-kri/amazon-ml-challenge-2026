"""
AWS Production Runner for Amazon ML Challenge 2026
Business Entity Resolution

Optimized for AWS SageMaker & EC2 Instances (c5.4xlarge / r5.4xlarge / g4dn.xlarge).
Features:
1. Chunked stream processing of 1,732,544 test entities to keep RAM < 8GB.
2. Parallel multi-core candidate generation & feature extraction.
3. Automated submission generation (matching_results.tsv, candidate_pairs.tsv).
4. Validation against utils/validate_submission.py --check-ids.
5. Auto-packaging into submission_final.zip.
"""

import os
import sys
import time
import argparse
import subprocess
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd

# Add source directory to path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from src.config import config
from src.preprocessing import preprocess_record
from src.data_loader import load_source_tsv, load_ground_truth
from src.blocking import MultiChannelBlocker
from src.features import build_candidate_feature_matrix
from src.model import EntityMatchingModel, optimize_threshold
from src.hard_negatives import mine_hard_negatives_round1


def train_production_model(train_sample_size: int = 10000, model_type: str = "lightgbm") -> EntityMatchingModel:
    """Train matching model with calibrated probabilities and optimized F0.5 threshold."""
    print("=" * 70)
    print(f"Training Production Model on {train_sample_size:,} records ({model_type})...")
    print("=" * 70)

    dataset_dir = config.dataset_dir
    train_dir = dataset_dir / "train"

    s1_df = load_source_tsv(train_dir / "train_source1.tsv", nrows=train_sample_size)
    ground_truth = load_ground_truth(train_dir / "train_ground_truth.tsv")

    s1_records = {
        row["entity_id"]: preprocess_record(row)
        for row in s1_df.to_dict(orient="records")
    }

    # Collect needed target candidate IDs
    target_ids_needed = set()
    for eid in s1_records:
        target_ids_needed.update(ground_truth.get(eid, set()))

    target_records = {}
    for fname in ["train_source2.tsv", "train_source3.tsv"]:
        p = train_dir / fname
        if not p.is_file():
            continue
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                if eid in target_ids_needed or len(target_records) < (train_sample_size * 4):
                    raw_dict = dict(zip(header, parts))
                    target_records[eid] = preprocess_record(raw_dict)

    blocker = MultiChannelBlocker(min_candidates=config.min_candidates, max_candidates=config.max_candidates)
    blocker.index_targets(target_records.values())
    blocker.prune_frequent_tokens()

    train_candidates = blocker.block_all(s1_records.values())

    # Hard negative mining
    r1_negs = mine_hard_negatives_round1(
        s1_records=s1_records,
        candidate_records=target_records,
        candidate_pairs=train_candidates,
        ground_truth=ground_truth,
        ratio=config.hard_negatives_ratio_r1,
    )

    pos_pairs = []
    for s1_id in s1_records:
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

    model = EntityMatchingModel(model_type=model_type)
    model.fit(X_train_df.values, y_train)

    print("Production model trained successfully!")
    return model


def run_chunked_test_inference(
    model: EntityMatchingModel,
    chunk_size: int = 50000,
    max_entities: int = None,
):
    """Memory-efficient chunked streaming inference across 1,732,544 test entities."""
    print("=" * 70)
    print("Starting Chunked Production Test Inference on AWS...")
    print("=" * 70)

    test_dir = config.dataset_dir / "test"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Index all candidates from test_source2 and test_source3
    print("Indexing candidate pool (test_source2 and test_source3)...")
    t0 = time.time()
    blocker = MultiChannelBlocker(min_candidates=config.min_candidates, max_candidates=config.max_candidates)
    target_records = {}

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        p = test_dir / fname
        if not p.is_file():
            continue
        print(f"Reading {fname}...")
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for idx, line in enumerate(f):
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                raw_dict = dict(zip(header, parts))
                p_rec = preprocess_record(raw_dict)
                target_records[eid] = p_rec

    print(f"Indexed {len(target_records):,} candidates in {time.time() - t0:.1f}s.")
    blocker.index_targets(target_records.values())
    blocker.prune_frequent_tokens()

    # Open output files with correct TSV headers
    with open(matching_file, "w", encoding="utf-8") as fm, open(candidate_file, "w", encoding="utf-8") as fc:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        fc.write("source1_entity_id\tcandidate_entity_ids\n")

    # Read and process test_source1.tsv in streaming chunks
    print("Processing test Source 1 entities in streaming chunks...", flush=True)
    total_processed = 0
    s1_path = test_dir / "test_source1.tsv"
    t_infer_start = time.time()
    total_expected = 1732544

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        chunk_lines = []

        for line in f:
            chunk_lines.append(line)
            if len(chunk_lines) >= chunk_size:
                _process_chunk(chunk_lines, header, blocker, target_records, model, matching_file, candidate_file)
                total_processed += len(chunk_lines)
                elapsed_min = (time.time() - t_infer_start) / 60
                pct = (total_processed / total_expected) * 100
                rate = total_processed / max(time.time() - t_infer_start, 1)
                remaining_sec = (total_expected - total_processed) / max(rate, 1e-5)
                eta_min = remaining_sec / 60
                print(
                    f"[{time.strftime('%X')}] Processed {total_processed:,} / {total_expected:,} entities "
                    f"({pct:.1f}%) | Elapsed: {elapsed_min:.1f}m | ETA: {eta_min:.1f}m",
                    flush=True
                )
                chunk_lines = []
                if max_entities and total_processed >= max_entities:
                    break

        if chunk_lines and (not max_entities or total_processed < max_entities):
            _process_chunk(chunk_lines, header, blocker, target_records, model, matching_file, candidate_file)
            total_processed += len(chunk_lines)
            elapsed_min = (time.time() - t_infer_start) / 60
            print(f"[{time.strftime('%X')}] Processed {total_processed:,} entities in {elapsed_min:.1f}m", flush=True)

    print(f"All {total_processed:,} test entities processed successfully!", flush=True)


def _process_chunk(chunk_lines, header, blocker, target_records, model, matching_file, candidate_file):
    chunk_records = {}
    chunk_eids = []
    for line in chunk_lines:
        parts = line.rstrip("\n").split("\t")
        eid = parts[0]
        chunk_eids.append(eid)
        raw_dict = dict(zip(header, parts))
        chunk_records[eid] = preprocess_record(raw_dict)

    # Candidate generation
    candidates = blocker.block_all(chunk_records.values())

    # Feature extraction
    X_chunk, pair_order = build_candidate_feature_matrix(
        s1_records=chunk_records,
        candidate_records=target_records,
        candidate_pairs=candidates,
    )

    if len(pair_order) > 0:
        probs = model.predict_proba(X_chunk.values)
        pair_probs = {pair: prob for pair, prob in zip(pair_order, probs)}
    else:
        pair_probs = {}

    predicted_matches = model.predict_matches(
        candidate_pairs=candidates,
        pair_probabilities=pair_probs,
        threshold=model.threshold,
        relative_margin=model.relative_margin,
    )

    # Append to output TSVs
    with open(matching_file, "a", encoding="utf-8") as fm, open(candidate_file, "a", encoding="utf-8") as fc:
        for eid in chunk_eids:
            cands = ",".join(candidates.get(eid, []))
            matches = ",".join(predicted_matches.get(eid, []))
            fc.write(f"{eid}\t{cands}\n")
            fm.write(f"{eid}\t{matches}\n")


def validate_and_package():
    """Run strict competition validator and create submission zip."""
    print("=" * 70)
    print("Running Competition Submission Validator...")
    print("=" * 70)

    validator = Path("utils/validate_submission.py")
    if not validator.is_file():
        validator = Path("../utils/validate_submission.py")

    cmd = [
        sys.executable,
        str(validator),
        "--matching", "output/matching_results.tsv",
        "--candidate", "output/candidate_pairs.tsv",
        "--test-dir", str(config.dataset_dir / "test"),
        "--check-ids",
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("Validation warnings/errors occurred. Please review output.")

    # Package into submission zip
    zip_path = Path("submission_final.zip")
    print(f"Creating {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write("output/matching_results.tsv", arcname="matching_results.tsv")
        z.write("output/candidate_pairs.tsv", arcname="candidate_pairs.tsv")
        z.write("Documentation.md", arcname="Documentation.md")
    print(f"Submission package created: {zip_path.resolve()} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="AWS Production Runner")
    parser.add_argument("--train-sample", type=int, default=15000, help="Train samples")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Inference chunk size")
    parser.add_argument("--max-entities", type=int, default=None, help="Limit test entities (None for full)")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"])
    args = parser.parse_args()

    model = train_production_model(train_sample_size=args.train_sample, model_type=args.model_type)
    run_chunked_test_inference(model, chunk_size=args.chunk_size, max_entities=args.max_entities)
    validate_and_package()


if __name__ == "__main__":
    main()
