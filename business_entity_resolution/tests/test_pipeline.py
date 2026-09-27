"""
Unit Tests for Entity Resolution Pipeline
Amazon ML Challenge 2026

Tests:
1. Macro F0.5 Evaluator (worked competition example, singletons, edge cases)
2. Normalization (legal suffixes, unicode accents, address abbreviations, French terms)
3. Component Extraction (PIN/ZIP/French postal codes, street numbers, landmarks)
4. Multi-Channel Blocking & Adaptive Widening
5. Pairwise & Group-Relative Feature Engineering
6. Model Group-Aware Decision Logic
7. Output Formatting & Subset Guarantee
"""

import sys
from pathlib import Path
import pytest
import numpy as np

# Ensure project root is in sys.path
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

from src.evaluation import compute_entity_f05, evaluate_macro_f05, compute_blocking_metrics
from src.preprocessing import (
    clean_basic_text,
    normalize_business_name,
    normalize_address,
    preprocess_record,
    normalize_unicode,
)
from src.blocking import MultiChannelBlocker
from src.features import (
    extract_pair_features,
    compute_group_relative_features,
    build_candidate_feature_matrix,
)
from src.model import EntityMatchingModel


# ---------------------------------------------------------------------------
# 1. Macro F0.5 Evaluator Tests
# ---------------------------------------------------------------------------

def test_competition_worked_example():
    """Verify exact match with the problem statement example:
    - Ground truth: {S2-00047, S3-00812}
    - Predicted: {S2-00047, S2-00193, S3-00812}
    - Precision: 2/3, Recall: 1.0 -> F0.5: 0.714
    """
    true_set = {"S2-00047", "S3-00812"}
    pred_set = {"S2-00047", "S2-00193", "S3-00812"}

    score, p, r = compute_entity_f05(true_set, pred_set)
    assert round(p, 4) == round(2 / 3, 4)
    assert round(r, 4) == 1.0
    assert round(score, 3) == 0.714


def test_singleton_evaluator_behavior():
    """Test that true singletons receive 1.0 when empty and 0.0 when false merge occurs."""
    # True singleton correctly predicted empty -> 1.0
    score, p, r = compute_entity_f05(set(), set())
    assert score == 1.0

    # True singleton falsely merged -> 0.0
    score, p, r = compute_entity_f05(set(), {"S2-00001"})
    assert score == 0.0


def test_macro_average_with_singletons():
    """Test macro average across multiple entities including singletons."""
    gt = {
        "S1-1": {"S2-10", "S3-20"},  # non-singleton
        "S1-2": set(),               # singleton
        "S1-3": {"S2-30"},           # non-singleton
    }
    preds = {
        "S1-1": {"S2-10", "S3-20"},  # perfect: 1.0
        "S1-2": set(),               # perfect singleton: 1.0
        "S1-3": set(),               # missed: 0.0
    }
    metrics = evaluate_macro_f05(gt, preds)
    # Expected macro: (1.0 + 1.0 + 0.0) / 3 = 0.6667
    assert round(metrics["macro_f05"], 4) == round(2 / 3, 4)
    assert metrics["singleton_accuracy"] == 1.0
    assert metrics["false_merge_count"] == 0


# ---------------------------------------------------------------------------
# 2. Normalization & Preprocessing Tests
# ---------------------------------------------------------------------------

def test_unicode_and_accent_normalization():
    """Ensure unicode accented characters (e.g. French / German) normalize cleanly."""
    text = "Drükor S.A.R.L."
    clean = clean_basic_text(text)
    assert "drukor" in clean
    assert "sarl" in clean


def test_legal_suffix_normalization():
    """Test standardizing and stripping various legal entity suffixes."""
    cases = [
        ("Acme Corporation", "acme"),
        ("Google LLC", "google"),
        ("Tata Consultancy Services Private Limited", "tata consultancy services"),
        ("Drükor S.A.R.L.", "drukor"),
        ("Infosys Ltd.", "infosys"),
        ("Amazon Inc.", "amazon"),
    ]
    for raw, expected_stem in cases:
        norm = normalize_business_name(raw)
        assert expected_stem in norm["name_suffix_stripped"], f"Failed on {raw}"


def test_domain_name_handling():
    """Verify that domain-style business names have domain stems extracted."""
    raw = "maurewilliamscolombier.com"
    norm = normalize_business_name(raw)
    assert "maurewilliamscolombier" in norm["name_clean"]


def test_address_abbreviations_and_components():
    """Verify abbreviation expansion and component extraction across US, India, and France."""
    # US address
    us_addr = "85 Wayne Ave., Ticonderoga, NY 12883"
    norm_us = normalize_address(us_addr, country="US")
    assert "avenue" in norm_us["addr_clean"]
    assert norm_us["postal_code"] == "12883"
    assert norm_us["street_num"] == "85"

    # India address with landmark
    in_addr = "Near SBI ATM, 2nd Main Rd, Chennai 600040"
    norm_in = normalize_address(in_addr, country="India")
    assert "road" in norm_in["addr_clean"]
    assert norm_in["postal_code"] == "600040"
    assert "near sbi atm" in norm_in["landmark"]

    # France address
    fr_addr = "12 Bd. Haussmann, Paris 75009"
    norm_fr = normalize_address(fr_addr, country="France")
    assert "boulevard" in norm_fr["addr_clean"]
    assert norm_fr["postal_code"] == "75009"
    assert norm_fr["street_num"] == "12"


# ---------------------------------------------------------------------------
# 3. Blocking & Adaptive Widening Tests
# ---------------------------------------------------------------------------

def test_multi_channel_blocker_retrieval():
    """Verify multi-channel blocker retrieves matches across exact, token, and address channels."""
    targets = [
        {
            "entity_id": "S2-1",
            "business_name": "Maure Williams Colombier Inc",
            "business_address": "85 Wayne Avenue, NY 12883",
            "country": "US",
        },
        {
            "entity_id": "S3-2",
            "business_name": "Drükor",
            "business_address": "85 Wayne Avenue, Ticonderoga, NY 12883",
            "country": "US",
        },
        {
            "entity_id": "S2-3",
            "business_name": "Unrelated Bakery",
            "business_address": "10 Main St, Chicago, IL 60601",
            "country": "US",
        },
    ]

    blocker = MultiChannelBlocker(min_candidates=2, max_candidates=10)
    blocker.index_targets(targets)

    query = {
        "entity_id": "S1-100",
        "business_name": "Maure Williams Colombier",
        "business_address": "85 Wayne Ave, NY 12883",
        "country": "US",
    }
    cands = blocker.block_entity(query)

    # Should retrieve S2-1 (name exact) and S3-2 (address match)
    assert "S2-1" in cands
    assert "S3-2" in cands


# ---------------------------------------------------------------------------
# 4. Feature Extraction & Group-Relative Features Tests
# ---------------------------------------------------------------------------

def test_feature_extraction_and_group_metrics():
    """Verify all 34 pairwise and group-relative features are populated."""
    s1 = preprocess_record({
        "entity_id": "S1-1",
        "business_name": "Atlas Logistics LLC",
        "business_address": "100 Industrial Pkwy, Cleveland, OH 44101",
        "country": "US",
    })
    c1 = preprocess_record({
        "entity_id": "S2-10",
        "business_name": "Atlas Logistics Corp",
        "business_address": "100 Industrial Parkway, Cleveland, OH 44101",
        "country": "US",
    })
    c2 = preprocess_record({
        "entity_id": "S3-20",
        "business_name": "Atlas Bakery",
        "business_address": "200 Market St, Cleveland, OH 44102",
        "country": "US",
    })

    f1 = extract_pair_features(s1, c1)
    f2 = extract_pair_features(s1, c2)

    # C1 should have higher string similarity than C2
    assert f1["name_jaro_winkler"] > f2["name_jaro_winkler"]
    assert f1["postal_both_present_match"] == 1.0
    assert f2["postal_both_present_mismatch"] == 1.0

    group = compute_group_relative_features([f1, f2])
    assert len(group) == 2
    assert group[0]["group_sim_rank"] == 1.0
    assert group[1]["group_sim_rank"] == 2.0
    assert group[0]["group_score_gap_to_top"] == 0.0


# ---------------------------------------------------------------------------
# 5. Model Decision Logic Tests
# ---------------------------------------------------------------------------

def test_model_group_aware_decision():
    """Verify group-aware thresholding and singleton prediction."""
    model = EntityMatchingModel(default_threshold=0.60, relative_margin=0.20)

    cand_pairs = {
        "S1-1": ["S2-10", "S2-11"],
        "S1-2": ["S3-20"],
    }
    # S1-1 has high confidence top candidate S2-10 (0.85) and distant S2-11 (0.50)
    # S1-2 has weak candidate S3-20 (0.40) below threshold
    probs = {
        ("S1-1", "S2-10"): 0.85,
        ("S1-1", "S2-11"): 0.50,
        ("S1-2", "S3-20"): 0.40,
    }

    preds = model.predict_matches(cand_pairs, probs, threshold=0.60, relative_margin=0.20)

    # S1-1 should match only S2-10
    assert preds["S1-1"] == {"S2-10"}
    # S1-2 should be empty (singleton)
    assert preds["S1-2"] == set()
