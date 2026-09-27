"""
Two-Round Hard Negative Mining for Imbalance and Precision Optimization
Amazon ML Challenge 2026

Round 1: String similarity-based hard negative mining (high-similarity non-matches)
Round 2: Model-based hard negative mining (high false positive probability candidates from first pass)
Strict fold separation: ensures no validation entity leakage during training.
"""

from typing import Dict, List, Set, Tuple
import numpy as np
import rapidfuzz.distance.JaroWinkler as jw

from .config import config


def mine_hard_negatives_round1(
    s1_records: Dict[str, dict],
    candidate_records: Dict[str, dict],
    candidate_pairs: Dict[str, List[str]],
    ground_truth: Dict[str, Set[str]],
    ratio: int = 4,
) -> List[Tuple[str, str]]:
    """Round 1: Mine high string-similarity non-matches.

    For each S1 entity, ranks all non-match candidates by name similarity and
    selects the hardest negatives (e.g., same brand name, different location).
    """
    mined_negatives: List[Tuple[str, str]] = []

    for s1_id, cids in candidate_pairs.items():
        if not cids:
            continue
        s1_rec = s1_records.get(s1_id)
        if not s1_rec:
            continue

        true_matches = ground_truth.get(s1_id, set())
        non_matches = [cid for cid in cids if cid not in true_matches]

        if not non_matches:
            continue

        n_needed = max(1, len(true_matches)) * ratio
        if len(non_matches) <= n_needed:
            for cid in non_matches:
                mined_negatives.append((s1_id, cid))
            continue

        # Score non-matches by string similarity to S1
        s1_name = s1_rec.get("name_suffix_stripped", "") or s1_rec.get("name_clean", "")
        scored_negs = []
        for cid in non_matches:
            cand_rec = candidate_records.get(cid)
            if not cand_rec:
                continue
            c_name = cand_rec.get("name_suffix_stripped", "") or cand_rec.get("name_clean", "")
            score = float(jw.similarity(s1_name, c_name)) if s1_name and c_name else 0.0
            scored_negs.append((score, cid))

        scored_negs.sort(key=lambda x: x[0], reverse=True)
        for _, cid in scored_negs[:n_needed]:
            mined_negatives.append((s1_id, cid))

    return mined_negatives


def mine_hard_negatives_round2(
    model,
    feature_matrix: np.ndarray,
    pair_ids: List[Tuple[str, str]],
    ground_truth: Dict[str, Set[str]],
    ratio: int = 2,
    existing_neg_set: Set[Tuple[str, str]] = None,
) -> List[Tuple[str, str]]:
    """Round 2: Model-based hard negative mining.

    Uses the round-1 trained model to score negative candidates and selects
    the highest-probability false positives (candidates that fooled the round-1 model).
    """
    if feature_matrix.shape[0] == 0 or len(pair_ids) == 0:
        return []

    preds = model.predict_proba(feature_matrix)
    if hasattr(preds, "ndim") and preds.ndim == 2:
        preds = preds[:, 1]

    # Filter to non-matches not already in existing_neg_set
    fp_candidates = []
    n_positives = 0

    for idx, (s1_id, cid) in enumerate(pair_ids):
        true_matches = ground_truth.get(s1_id, set())
        is_true = cid in true_matches
        if is_true:
            n_positives += 1
            continue

        pair = (s1_id, cid)
        if existing_neg_set and pair in existing_neg_set:
            continue

        fp_candidates.append((preds[idx], pair))

    # Sort descending by predicted probability
    fp_candidates.sort(key=lambda x: x[0], reverse=True)
    n_needed = max(1, n_positives) * ratio

    return [pair for _, pair in fp_candidates[:n_needed]]
