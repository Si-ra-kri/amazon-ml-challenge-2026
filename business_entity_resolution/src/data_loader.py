"""
Data Loader, Schema Validation, Exploratory Data Analysis, and K-Fold Splitter
Amazon ML Challenge 2026
"""

from collections import Counter
import os
from pathlib import Path
from typing import Dict, Generator, Iterator, List, Optional, Set, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from .config import config

EXPECTED_SOURCE_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
EXPECTED_GT_COLUMNS = ["source1_entity_id", "matched_entity_ids"]


def validate_schema(df: pd.DataFrame, expected_cols: List[str], expected_prefix: Optional[str] = None) -> bool:
    """Validate DataFrame columns and ID prefixes."""
    cols = df.columns.tolist()
    if cols != expected_cols:
        raise ValueError(f"Schema mismatch: expected {expected_cols}, got {cols}")
    if expected_prefix:
        sample_ids = df["entity_id"].dropna().head(100)
        invalid = [eid for eid in sample_ids if not eid.startswith(expected_prefix)]
        if invalid:
            raise ValueError(f"ID prefix mismatch: expected {expected_prefix}, found invalid examples: {invalid[:5]}")
    return True


def load_source_tsv(file_path: Path, nrows: Optional[int] = None, use_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """Load a source TSV explicitly with tab separator and string types."""
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    df = pd.read_csv(
        file_path,
        sep="\t",
        nrows=nrows,
        usecols=use_cols,
        dtype=str,
        na_values=[],
        keep_default_na=False,
    )
    # Replace any empty strings with empty string rather than NaN
    df.fillna("", inplace=True)
    return df


def iter_source_tsv(
    file_path: Path, chunksize: int = 100000, use_cols: Optional[List[str]] = None
) -> Iterator[pd.DataFrame]:
    """Iterate over large TSV in memory-safe chunks."""
    if not file_path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")
    for chunk in pd.read_csv(
        file_path,
        sep="\t",
        chunksize=chunksize,
        usecols=use_cols,
        dtype=str,
        na_values=[],
        keep_default_na=False,
    ):
        chunk.fillna("", inplace=True)
        yield chunk


def load_ground_truth(file_path: Path, nrows: Optional[int] = None) -> Dict[str, Set[str]]:
    """Load ground truth TSV into a mapping: source1_entity_id -> set(matched_ids)."""
    gt_map: Dict[str, Set[str]] = {}
    if not file_path.is_file():
        raise FileNotFoundError(f"Ground truth not found: {file_path}")

    with open(file_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header != EXPECTED_GT_COLUMNS:
            raise ValueError(f"Ground truth header mismatch: expected {EXPECTED_GT_COLUMNS}, got {header}")
        for idx, line in enumerate(f):
            if nrows is not None and idx >= nrows:
                break
            parts = line.rstrip("\n").split("\t")
            s1_id = parts[0]
            matched_str = parts[1] if len(parts) > 1 else ""
            if matched_str.strip():
                gt_map[s1_id] = set(matched_str.split(","))
            else:
                gt_map[s1_id] = set()
    return gt_map


def compute_dataset_stats(dataset_dir: Path, sample_size: int = 100000) -> Dict[str, any]:
    """Perform fast, streaming exploratory analysis across train/test files."""
    stats = {}

    for split in ["train", "test"]:
        sub_dir = dataset_dir / split
        if not sub_dir.is_dir():
            continue
        stats[split] = {}
        for fname in sorted(os.listdir(sub_dir)):
            if not fname.endswith(".tsv"):
                continue
            fpath = sub_dir / fname
            row_count = 0
            missing_name = 0
            missing_addr = 0
            country_counter = Counter()
            name_lens = []
            addr_lens = []
            duplicate_ids = 0
            seen_ids = set()

            with open(fpath, "r", encoding="utf-8") as f:
                header = f.readline().rstrip("\n").split("\t")
                is_gt = "ground_truth" in fname
                for idx, line in enumerate(f):
                    row_count += 1
                    parts = line.rstrip("\n").split("\t")
                    eid = parts[0]

                    if is_gt:
                        continue

                    if idx < sample_size:
                        if eid in seen_ids:
                            duplicate_ids += 1
                        seen_ids.add(eid)

                        b_name = parts[1] if len(parts) > 1 else ""
                        b_addr = parts[2] if len(parts) > 2 else ""
                        b_country = parts[3] if len(parts) > 3 else ""

                        if not b_name.strip():
                            missing_name += 1
                        if not b_addr.strip():
                            missing_addr += 1

                        country_counter[b_country] += 1
                        name_lens.append(len(b_name))
                        addr_lens.append(len(b_addr))

            stats[split][fname] = {
                "row_count": row_count,
                "missing_name_rate": (missing_name / sample_size) if sample_size else 0,
                "missing_addr_rate": (missing_addr / sample_size) if sample_size else 0,
                "sample_countries": dict(country_counter),
                "avg_name_length": float(np.mean(name_lens)) if name_lens else 0,
                "avg_addr_length": float(np.mean(addr_lens)) if addr_lens else 0,
                "sample_duplicate_ids": duplicate_ids,
            }

    return stats


def create_kfold_splits(
    s1_ids: List[str],
    ground_truth: Dict[str, Set[str]],
    n_splits: int = 5,
    random_state: int = 42,
) -> List[Tuple[List[str], List[str]]]:
    """Create Stratified K-Fold splits on Source 1 entities based on singleton vs non-singleton.

    Ensures zero leakage across folds.
    Returns:
        List of (train_s1_ids, val_s1_ids) tuples for each fold.
    """
    labels = np.array([0 if len(ground_truth.get(eid, set())) == 0 else 1 for eid in s1_ids])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    splits = []
    s1_array = np.array(s1_ids)
    for train_idx, val_idx in skf.split(s1_array, labels):
        train_eids = s1_array[train_idx].tolist()
        val_eids = s1_array[val_idx].tolist()
        splits.append((train_eids, val_eids))

    return splits
