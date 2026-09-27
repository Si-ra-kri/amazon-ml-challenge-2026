"""
Pairwise and Group-Relative Feature Engineering
Amazon ML Challenge 2026

Extracts comprehensive, vectorized pairwise features between Source 1 and candidate records:
1. Name features (exact match, token Jaccard/Dice, Levenshtein, Jaro-Winkler, LCS, phonetics, char n-grams)
2. Address features (token Jaccard/Dice, edit distance, postal code match/missingness indicators, street number)
3. Meta features (country match boolean, source pair S1-S2 vs S1-S3, length ratios and deltas)
4. Group-relative features (similarity rank, gap to top candidate, gap to next candidate, z-score within group)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import rapidfuzz.distance.JaroWinkler as jw
import rapidfuzz.distance.LCSseq as lcs
import rapidfuzz.distance.Levenshtein as lev

from .config import config


def get_char_ngrams(text: str, n: int = 3) -> set:
    """Extract character n-grams from text."""
    if len(text) < n:
        return {text} if text else set()
    return {text[i : i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set1: set, set2: set) -> float:
    """Compute Jaccard similarity between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersect = len(set1 & set2)
    union = len(set1 | set2)
    return intersect / union if union > 0 else 0.0


def dice_similarity(set1: set, set2: set) -> float:
    """Compute Dice coefficient between two sets."""
    if not set1 and not set2:
        return 1.0
    if not set1 or not set2:
        return 0.0
    intersect = len(set1 & set2)
    total = len(set1) + len(set2)
    return (2.0 * intersect) / total if total > 0 else 0.0


def extract_pair_features(s1_rec: dict, cand_rec: dict) -> Dict[str, float]:
    """Compute raw pairwise similarity features for a single (S1, Candidate) pair."""
    # Names
    name_clean1 = s1_rec.get("name_clean", "")
    name_clean2 = cand_rec.get("name_clean", "")

    name_stripped1 = s1_rec.get("name_suffix_stripped", "")
    name_stripped2 = cand_rec.get("name_suffix_stripped", "")

    name_tokens1 = set(s1_rec.get("name_tokens", "").split())
    name_tokens2 = set(cand_rec.get("name_tokens", "").split())

    # Exact matches
    name_exact = 1.0 if s1_rec.get("business_name_raw", "").strip().lower() == cand_rec.get("business_name_raw", "").strip().lower() and s1_rec.get("business_name_raw") else 0.0
    name_clean_exact = 1.0 if name_clean1 and name_clean1 == name_clean2 else 0.0
    name_stripped_exact = 1.0 if name_stripped1 and name_stripped1 == name_stripped2 else 0.0

    # Token similarities
    name_jaccard = jaccard_similarity(name_tokens1, name_tokens2)
    name_dice = dice_similarity(name_tokens1, name_tokens2)

    # String distance metrics
    name_target1 = name_stripped1 if name_stripped1 else name_clean1
    name_target2 = name_stripped2 if name_stripped2 else name_clean2

    if name_target1 and name_target2:
        name_lev = float(lev.normalized_similarity(name_target1, name_target2))
        name_jw = float(jw.similarity(name_target1, name_target2))
        name_lcs = float(lcs.normalized_similarity(name_target1, name_target2))
    else:
        name_lev = 0.0
        name_jw = 0.0
        name_lcs = 0.0

    # Phonetics
    s1_soundex = s1_rec.get("name_soundex", "")
    c_soundex = cand_rec.get("name_soundex", "")
    name_soundex_match = 1.0 if s1_soundex and s1_soundex == c_soundex else 0.0

    s1_meta = s1_rec.get("name_metaphone", "")
    c_meta = cand_rec.get("name_metaphone", "")
    name_meta_match = 1.0 if s1_meta and s1_meta == c_meta else 0.0

    # Char 3-grams
    ngrams1 = get_char_ngrams(name_clean1, 3)
    ngrams2 = get_char_ngrams(name_clean2, 3)
    name_char3gram = dice_similarity(ngrams1, ngrams2)

    # Addresses
    addr_clean1 = s1_rec.get("addr_clean", "")
    addr_clean2 = cand_rec.get("addr_clean", "")

    addr_tokens1 = set(s1_rec.get("addr_tokens", "").split())
    addr_tokens2 = set(cand_rec.get("addr_tokens", "").split())

    addr_exact = 1.0 if s1_rec.get("business_address_raw", "").strip().lower() == cand_rec.get("business_address_raw", "").strip().lower() and s1_rec.get("business_address_raw") else 0.0
    addr_clean_exact = 1.0 if addr_clean1 and addr_clean1 == addr_clean2 else 0.0

    addr_jaccard = jaccard_similarity(addr_tokens1, addr_tokens2)
    addr_dice = dice_similarity(addr_tokens1, addr_tokens2)

    if addr_clean1 and addr_clean2:
        addr_lev = float(lev.normalized_similarity(addr_clean1, addr_clean2))
        addr_jw = float(jw.similarity(addr_clean1, addr_clean2))
    else:
        addr_lev = 0.0
        addr_jw = 0.0

    # Postal code features with explicit missingness states
    postal1 = s1_rec.get("postal_code", "").strip()
    postal2 = cand_rec.get("postal_code", "").strip()

    postal_missing_either = 1.0 if (not postal1 or not postal2) else 0.0
    if not postal1 and not postal2:
        postal_exact = 0.5  # Neutral when both missing
        postal_both_present_match = 0.0
        postal_both_present_mismatch = 0.0
    elif not postal1 or not postal2:
        postal_exact = 0.5
        postal_both_present_match = 0.0
        postal_both_present_mismatch = 0.0
    else:
        match = (postal1 == postal2)
        postal_exact = 1.0 if match else 0.0
        postal_both_present_match = 1.0 if match else 0.0
        postal_both_present_mismatch = 0.0 if match else 1.0

    # Street number
    street1 = s1_rec.get("street_num", "").strip()
    street2 = cand_rec.get("street_num", "").strip()
    street_num_missing_either = 1.0 if (not street1 or not street2) else 0.0
    street_num_match = 1.0 if street1 and street1 == street2 else 0.0

    # Meta features
    c1 = s1_rec.get("country", "").strip().upper()
    c2 = cand_rec.get("country", "").strip().upper()
    country_match = 1.0 if c1 and c1 == c2 else (0.5 if not c1 or not c2 else 0.0)

    cid = cand_rec.get("entity_id", "")
    is_s2 = 1.0 if cid.startswith("S2-") else 0.0
    is_s3 = 1.0 if cid.startswith("S3-") else 0.0

    len1 = len(name_clean1)
    len2 = len(name_clean2)
    name_len_diff = float(abs(len1 - len2))
    max_len = max(len1, len2)
    name_len_ratio = (min(len1, len2) / max_len) if max_len > 0 else 1.0

    alen1 = len(addr_clean1)
    alen2 = len(addr_clean2)
    addr_len_diff = float(abs(alen1 - alen2))

    return {
        "name_exact_match": name_exact,
        "name_clean_exact_match": name_clean_exact,
        "name_suffix_stripped_match": name_stripped_exact,
        "name_token_jaccard": name_jaccard,
        "name_token_dice": name_dice,
        "name_levenshtein_ratio": name_lev,
        "name_jaro_winkler": name_jw,
        "name_lcs_ratio": name_lcs,
        "name_soundex_match": name_soundex_match,
        "name_metaphone_match": name_meta_match,
        "name_char3gram_cosine": name_char3gram,
        "addr_exact_match": addr_exact,
        "addr_clean_exact_match": addr_clean_exact,
        "addr_token_jaccard": addr_jaccard,
        "addr_token_dice": addr_dice,
        "addr_levenshtein_ratio": addr_lev,
        "addr_jaro_winkler": addr_jw,
        "postal_exact_match": postal_exact,
        "postal_missing_either": postal_missing_either,
        "postal_both_present_match": postal_both_present_match,
        "postal_both_present_mismatch": postal_both_present_mismatch,
        "street_num_match": street_num_match,
        "street_num_missing_either": street_num_missing_either,
        "country_match": country_match,
        "is_source2": is_s2,
        "is_source3": is_s3,
        "name_len_diff": name_len_diff,
        "name_len_ratio": name_len_ratio,
        "addr_len_diff": addr_len_diff,
    }


def compute_group_relative_features(pair_features_list: List[Dict[str, float]]) -> List[Dict[str, float]]:
    """Compute group-relative features for candidates belonging to the same Source 1 entity:
    - rank among candidates based on composite base score
    - score gap to top candidate
    - score gap to next candidate
    - z-score of composite score relative to group mean and std
    """
    n = len(pair_features_list)
    if n == 0:
        return []

    # Composite similarity score for ranking within group
    scores = np.array([
        0.45 * f["name_jaro_winkler"]
        + 0.25 * f["name_token_jaccard"]
        + 0.15 * f["addr_jaro_winkler"]
        + 0.15 * f["addr_token_jaccard"]
        for f in pair_features_list
    ])

    order = np.argsort(-scores)  # descending order
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)

    top_score = np.max(scores)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    std_safe = std_score if std_score > 1e-5 else 1.0

    out = []
    for i, f in enumerate(pair_features_list):
        feat = dict(f)
        s = scores[i]
        curr_rank = ranks[i]

        # Gap to next best candidate
        if curr_rank < n:
            next_idx = order[curr_rank]  # candidate ranked right below
            gap_next = float(s - scores[next_idx])
        else:
            gap_next = float(s)  # lowest rank

        feat["s1_candidate_count"] = float(n)
        feat["group_sim_rank"] = float(curr_rank)
        feat["group_score_gap_to_top"] = float(top_score - s)
        feat["group_score_gap_to_next"] = gap_next
        feat["group_score_zscore"] = float((s - mean_score) / std_safe)
        out.append(feat)

    return out


def build_candidate_feature_matrix(
    s1_records: Dict[str, dict],
    candidate_records: Dict[str, dict],
    candidate_pairs: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, List[Tuple[str, str]]]:
    """Generate DataFrame of features for all (S1, Candidate) pairs.

    Returns:
        (df_features, pair_ids_list) where pair_ids_list is list of (s1_id, candidate_id)
    """
    all_rows = []
    pair_ids = []

    for s1_id, cids in candidate_pairs.items():
        if not cids:
            continue
        s1_rec = s1_records.get(s1_id)
        if not s1_rec:
            continue

        group_pairs = []
        group_pair_ids = []
        for cid in cids:
            cand_rec = candidate_records.get(cid)
            if not cand_rec:
                continue
            f = extract_pair_features(s1_rec, cand_rec)
            group_pairs.append(f)
            group_pair_ids.append((s1_id, cid))

        enriched_group = compute_group_relative_features(group_pairs)
        all_rows.extend(enriched_group)
        pair_ids.extend(group_pair_ids)

    if not all_rows:
        return pd.DataFrame(columns=config.feature_names), []

    df = pd.DataFrame(all_rows)
    # Ensure columns match config.feature_names order
    for col in config.feature_names:
        if col not in df.columns:
            df[col] = 0.0
    df = df[config.feature_names]

    return df, pair_ids
