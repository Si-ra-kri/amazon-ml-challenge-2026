"""
Accelerated Production Runner with Resume Support
Amazon ML Challenge 2026: Business Entity Resolution

Optimized for:
- Intel Core Ultra 9 275HX (24 Cores / 24 Threads)
- NVIDIA GeForce RTX 5060 Laptop GPU
- 16 GB DDR5 RAM (Streaming chunking with strict memory bounds < 6 GB)

Features:
1. Automatic Resume: Detects existing line count in output files and resumes without losing progress.
2. High-Throughput Streaming: Memory-safe chunked candidate generation & feature extraction.
3. GPU / Multi-Core Inference: Utilizes all hardware threads and GPU where applicable.
4. Automated Validation: Runs utils/validate_submission.py upon completion.
5. Packaging: Creates submission_final.zip ready for upload.
"""

import argparse
import gc
import os
from pathlib import Path
import subprocess
import sys
import time
import zipfile
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
from src.model import EntityMatchingModel
from src.hard_negatives import mine_hard_negatives_round1


def train_production_model(train_sample_size: int = 15000, model_type: str = "lightgbm") -> EntityMatchingModel:
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

    print("Production model trained and calibrated successfully!\n")
    return model


def count_existing_rows(file_path: Path) -> int:
    """Return number of data rows in existing TSV (excluding header)."""
    if not file_path.is_file():
        return 0
    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        _ = f.readline()  # header
        for _ in f:
            count += 1
    return count


class CompactCandidate:
    __slots__ = (
        "entity_id", "business_name_raw", "country", "name_clean",
        "name_suffix_stripped", "name_tokens", "name_soundex",
        "name_metaphone", "addr_clean", "postal_code", "street_num",
        "addr_tokens"
    )
    def __init__(self, d):
        for k in self.__slots__:
            v = d.get(k, "")
            setattr(self, k, sys.intern(str(v)) if v else "")

    def get(self, k, default=""):
        return getattr(self, k, default)

    def __getitem__(self, k):
        return getattr(self, k)

    def __contains__(self, k):
        return hasattr(self, k)


def run_accelerated_inference(
    model: EntityMatchingModel,
    chunk_size: int = 50000,
    resume: bool = True,
    max_entities: int = None,
):
    """Memory-efficient streaming inference with resume capability across 1,732,544 test entities."""
    print("=" * 70)
    print("Starting Accelerated Test Inference Pipeline...")
    print("=" * 70)

    test_dir = config.dataset_dir / "test"
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    matching_file = output_dir / "matching_results.tsv"
    candidate_file = output_dir / "candidate_pairs.tsv"

    # Check resume status
    existing_m = count_existing_rows(matching_file) if resume else 0
    existing_c = count_existing_rows(candidate_file) if resume else 0

    if resume and existing_m > 0 and existing_m == existing_c:
        skip_count = existing_m
        print(f"Resuming from checkpoint: {skip_count:,} entities already processed.")
        file_mode = "a"
    else:
        skip_count = 0
        file_mode = "w"
        if resume and (existing_m > 0 or existing_c > 0):
            print(f"Line counts mismatched (matching: {existing_m}, candidate: {existing_c}). Restarting from scratch.")

    # Index candidate targets from Source 2 and Source 3
    print("Indexing candidate pool (test_source2.tsv and test_source3.tsv)...")
    t0 = time.time()
    blocker = MultiChannelBlocker(min_candidates=config.min_candidates, max_candidates=config.max_candidates)
    target_records = {}

    for fname in ["test_source2.tsv", "test_source3.tsv"]:
        p = test_dir / fname
        if not p.is_file():
            continue
        print(f"Loading {fname}...", flush=True)
        t_sub = time.time()
        with open(p, "r", encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                eid = parts[0]
                raw_dict = dict(zip(header, parts))
                p_rec = preprocess_record(raw_dict)
                target_records[eid] = CompactCandidate(p_rec)
        print(f"  Loaded in {time.time() - t_sub:.1f}s. Current candidate count: {len(target_records):,}", flush=True)

    print(f"Total {len(target_records):,} candidates indexed in {time.time() - t0:.1f}s.", flush=True)
    blocker.index_targets(target_records.values())
    blocker.prune_frequent_tokens()
    print("Candidate inverted indices pruned and ready.\n", flush=True)

    # Prepare output files
    if file_mode == "w":
        with open(matching_file, "w", encoding="utf-8") as fm, open(candidate_file, "w", encoding="utf-8") as fc:
            fm.write("source1_entity_id\tmatched_entity_ids\n")
            fc.write("source1_entity_id\tcandidate_entity_ids\n")

    # Streaming chunks from test_source1.tsv
    total_expected = 1732544
    s1_path = test_dir / "test_source1.tsv"
    total_processed = skip_count
    t_start = time.time()

    print(f"Beginning streaming inference from entity {total_processed + 1:,} to {total_expected:,} (Chunk size: {chunk_size:,})...", flush=True)

    with open(s1_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")

        # Fast forward skipped rows
        if skip_count > 0:
            print(f"Skipping {skip_count:,} already processed rows...", flush=True)
            for _ in range(skip_count):
                f.readline()
            print("Fast-forward complete. Resuming inference.\n", flush=True)

        chunk_lines = []
        for line in f:
            chunk_lines.append(line)
            if len(chunk_lines) >= chunk_size:
                _process_and_append_chunk(chunk_lines, header, blocker, target_records, model, matching_file, candidate_file)
                total_processed += len(chunk_lines)
                elapsed_min = (time.time() - t_start) / 60
                new_done = total_processed - skip_count
                rate = new_done / max(time.time() - t_start, 1)
                rem_entities = total_expected - total_processed
                eta_min = (rem_entities / max(rate, 1e-5)) / 60
                pct = (total_processed / total_expected) * 100
                print(
                    f"[{time.strftime('%X')}] Processed {total_processed:,} / {total_expected:,} ({pct:.1f}%) | "
                    f"Speed: {rate:,.0f} ent/s | Elapsed: {elapsed_min:.1f}m | ETA: {eta_min:.1f}m",
                    flush=True
                )
                chunk_lines = []
                gc.collect()

                if max_entities and total_processed >= max_entities:
                    break

        if chunk_lines and (not max_entities or total_processed < max_entities):
            _process_and_append_chunk(chunk_lines, header, blocker, target_records, model, matching_file, candidate_file)
            total_processed += len(chunk_lines)
            elapsed_min = (time.time() - t_start) / 60
            print(f"[{time.strftime('%X')}] Final chunk processed! Total: {total_processed:,} in {elapsed_min:.1f}m", flush=True)

    print(f"\nAll {total_processed:,} entities processed successfully!", flush=True)


def _process_and_append_chunk(chunk_lines, header, blocker, target_records, model, matching_file, candidate_file):
    chunk_records = {}
    chunk_eids = []
    for line in chunk_lines:
        parts = line.rstrip("\n").split("\t")
        eid = parts[0]
        chunk_eids.append(eid)
        raw_dict = dict(zip(header, parts))
        chunk_records[eid] = preprocess_record(raw_dict)

    # 1. Blocking
    candidates = blocker.block_all(chunk_records.values())

    # 2. Features
    X_chunk, pair_order = build_candidate_feature_matrix(
        s1_records=chunk_records,
        candidate_records=target_records,
        candidate_pairs=candidates,
    )

    # 3. Model scoring
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

    # 4. Append to disk in TSV format
    with open(matching_file, "a", encoding="utf-8") as fm, open(candidate_file, "a", encoding="utf-8") as fc:
        for eid in chunk_eids:
            cands = ",".join(candidates.get(eid, []))
            matches = ",".join(predicted_matches.get(eid, []))
            fc.write(f"{eid}\t{cands}\n")
            fm.write(f"{eid}\t{matches}\n")


def validate_and_package():
    """Run strict competition validator and create submission zip."""
    print("\n" + "=" * 70)
    print("Running Official Competition Submission Validator...")
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
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("Warning: Validator returned issues. Please inspect above output.")
    else:
        print("PASS! Zero errors detected by competition validator.")

    # Package into submission zip
    zip_path = Path("submission_final.zip")
    print(f"\nPackaging into {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
        z.write("output/matching_results.tsv", arcname="matching_results.tsv")
        z.write("output/candidate_pairs.tsv", arcname="candidate_pairs.tsv")
        doc_path = Path("Documentation.md")
        if doc_path.is_file():
            z.write("Documentation.md", arcname="Documentation.md")
    print(f"Final submission package created: {zip_path.resolve()} ({zip_path.stat().st_size / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Accelerated Production Runner with Resume Support")
    parser.add_argument("--train-sample", type=int, default=15000, help="Train samples for model fitting")
    parser.add_argument("--chunk-size", type=int, default=50000, help="Streaming chunk size")
    parser.add_argument("--no-resume", action="store_true", help="Do not resume; start fresh from 0")
    parser.add_argument("--max-entities", type=int, default=None, help="Limit total entities (for testing)")
    parser.add_argument("--model-type", type=str, default="lightgbm", choices=["lightgbm", "xgboost"])
    args = parser.parse_args()

    model = train_production_model(train_sample_size=args.train_sample, model_type=args.model_type)
    run_accelerated_inference(
        model,
        chunk_size=args.chunk_size,
        resume=(not args.no_resume),
        max_entities=args.max_entities,
    )
    validate_and_package()


if __name__ == "__main__":
    main()
