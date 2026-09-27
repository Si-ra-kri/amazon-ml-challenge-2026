"""
Evaluation Metric: Macro F0.5 Score for Business Entity Resolution
Amazon ML Challenge 2026

Calculates per-entity F0.5 (with beta=0.5, weighting precision 2x over recall),
including singletons (entities with 0 true matches), and macro-averages across
all Source 1 entities.
"""

from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np


def compute_entity_f05(true_ids: Set[str], pred_ids: Set[str]) -> Tuple[float, float, float]:
    """Compute (f05, precision, recall) for a single Source 1 entity.

    Rules:
    - If true_ids is empty (singleton):
        - pred_ids is empty -> f05 = 1.0, precision = 1.0, recall = 1.0
        - pred_ids not empty -> f05 = 0.0, precision = 0.0, recall = 1.0 (false merge)
    - If true_ids is non-empty:
        - pred_ids is empty -> f05 = 0.0, precision = 0.0, recall = 0.0
        - pred_ids not empty:
            - tp = len(true_ids & pred_ids)
            - if tp == 0 -> f05 = 0.0, precision = 0.0, recall = 0.0
            - p = tp / len(pred_ids)
            - r = tp / len(true_ids)
            - f05 = (1.25 * p * r) / (0.25 * p + r)
    """
    n_true = len(true_ids)
    n_pred = len(pred_ids)

    # Singleton case
    if n_true == 0:
        if n_pred == 0:
            return 1.0, 1.0, 1.0
        else:
            return 0.0, 0.0, 1.0

    # Non-singleton true match
    if n_pred == 0:
        return 0.0, 0.0, 0.0

    tp = len(true_ids & pred_ids)
    if tp == 0:
        return 0.0, 0.0, 0.0

    p = tp / n_pred
    r = tp / n_true
    denom = 0.25 * p + r
    if denom <= 0:
        f05 = 0.0
    else:
        f05 = (1.25 * p * r) / denom

    return f05, p, r


def evaluate_macro_f05(
    ground_truth: Dict[str, Set[str]],
    predictions: Dict[str, Set[str]],
    required_s1_ids: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Compute Macro F0.5, Macro Precision, Macro Recall, and diagnostic metrics.

    Args:
        ground_truth: Dict mapping source1_entity_id -> set of true matching IDs
        predictions: Dict mapping source1_entity_id -> set of predicted matching IDs
        required_s1_ids: Optional list of S1 IDs that must be evaluated. If None,
                         defaults to union of keys from ground_truth and predictions.

    Returns:
        Dict with keys:
            - macro_f05: Macro-averaged F0.5
            - macro_precision: Macro-averaged precision
            - macro_recall: Macro-averaged recall
            - singleton_accuracy: % of true singletons correctly predicted empty
            - false_merge_count: number of false positives on non-matches
            - n_entities: total S1 entities evaluated
            - n_singletons: total true singletons
    """
    if required_s1_ids is None:
        eval_ids = sorted(ground_truth.keys())
    else:
        eval_ids = list(required_s1_ids)

    if not eval_ids:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "false_merge_count": 0,
            "n_entities": 0,
            "n_singletons": 0,
        }

    f05_scores = []
    precisions = []
    recalls = []

    n_singletons = 0
    correct_singletons = 0
    false_merges = 0

    for s1_id in eval_ids:
        true_set = ground_truth.get(s1_id, set())
        pred_set = predictions.get(s1_id, set())

        score, p, r = compute_entity_f05(true_set, pred_set)
        f05_scores.append(score)
        precisions.append(p)
        recalls.append(r)

        if len(true_set) == 0:
            n_singletons += 1
            if len(pred_set) == 0:
                correct_singletons += 1
            else:
                false_merges += len(pred_set)
        else:
            false_merges += len(pred_set - true_set)

    singleton_acc = (
        (correct_singletons / n_singletons) if n_singletons > 0 else 1.0
    )

    return {
        "macro_f05": float(np.mean(f05_scores)),
        "macro_precision": float(np.mean(precisions)),
        "macro_recall": float(np.mean(recalls)),
        "singleton_accuracy": float(singleton_acc),
        "false_merge_count": int(false_merges),
        "n_entities": len(eval_ids),
        "n_singletons": n_singletons,
    }


def compute_blocking_metrics(
    ground_truth: Dict[str, Set[str]],
    candidates: Dict[str, Set[str]],
    total_comparisons_full: Optional[int] = None,
) -> Dict[str, float]:
    """Compute candidate recall and reduction ratio for the blocking stage.

    Candidate Recall = (Total true matches captured in candidate set) / (Total true matches)
    Reduction Ratio = 1 - (Total candidate pairs) / (Total Cartesian product size)
    """
    total_true_matches = 0
    captured_true_matches = 0
    total_candidate_pairs = 0
    candidate_counts = []

    for s1_id, true_set in ground_truth.items():
        cand_set = candidates.get(s1_id, set())
        n_true = len(true_set)
        total_true_matches += n_true
        if n_true > 0:
            captured_true_matches += len(true_set & cand_set)
        n_cand = len(cand_set)
        total_candidate_pairs += n_cand
        candidate_counts.append(n_cand)

    recall = (
        (captured_true_matches / total_true_matches)
        if total_true_matches > 0
        else 1.0
    )

    reduction_ratio = 0.0
    if total_comparisons_full and total_comparisons_full > 0:
        reduction_ratio = 1.0 - (total_candidate_pairs / total_comparisons_full)

    return {
        "candidate_recall": float(recall),
        "captured_matches": int(captured_true_matches),
        "total_true_matches": int(total_true_matches),
        "total_candidate_pairs": int(total_candidate_pairs),
        "reduction_ratio": float(reduction_ratio),
        "avg_candidates_per_entity": float(np.mean(candidate_counts)) if candidate_counts else 0.0,
        "max_candidates_per_entity": int(np.max(candidate_counts)) if candidate_counts else 0,
        "median_candidates_per_entity": float(np.median(candidate_counts)) if candidate_counts else 0.0,
    }
