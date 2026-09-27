"""
Multi-Channel Blocking and Adaptive Widening for Candidate Pair Generation
Amazon ML Challenge 2026

Generates candidate pairs (Source 1, Candidate S2/S3) using:
1. Exact / Normalized Name Blocking
2. Significant Token Inverted Index Blocking
3. Phonetic Blocking (Metaphone / Soundex)
4. Address Component Blocking (Postal Code + Name Token)
5. Name Prefix / 3-gram Inverted Index
6. Adaptive Widening for sparse candidate sets (< min_candidates)
"""

from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set, Tuple
import numpy as np

from .config import config
from .preprocessing import COMMON_NAME_STOPWORDS, preprocess_record


class MultiChannelBlocker:
    def __init__(
        self,
        min_candidates: int = 2,
        max_candidates: int = 35,
        max_token_postings: int = 1500,
    ):
        self.min_candidates = min_candidates
        self.max_candidates = max_candidates
        self.max_token_postings = max_token_postings

        # Inverted index channels for candidate records (S2 and S3)
        self.index_exact_name = defaultdict(list)
        self.index_suffix_stripped = defaultdict(list)
        self.index_name_tokens = defaultdict(list)
        self.index_phonetic = defaultdict(list)
        self.index_postal_name_token = defaultdict(list)
        self.index_prefix4 = defaultdict(list)
        # Address-based blocking channels (for DBA / alternate trade names)
        self.index_addr_street_token = defaultdict(list)
        self.index_addr_tokens = defaultdict(list)
        self.index_postal_only = defaultdict(list)

        # Store candidate basic metadata: id -> (country, clean_name, clean_addr)
        self.target_records: Dict[str, dict] = {}

    def index_targets(self, records: Iterable[dict]):
        """Index candidate records from Source 2 and Source 3.

        records can be raw dicts or preprocessed dicts.
        """
        for rec in records:
            if "name_clean" not in rec:
                p_rec = preprocess_record(rec)
            else:
                p_rec = rec

            eid = p_rec["entity_id"]
            country = p_rec.get("country", "")

            name_clean = p_rec.get("name_clean", "")
            name_stripped = p_rec.get("name_suffix_stripped", "")
            tokens = [t for t in p_rec.get("name_tokens", "").split() if t]
            soundex = p_rec.get("name_soundex", "")
            metaphone = p_rec.get("name_metaphone", "")
            postal = p_rec.get("postal_code", "")
            street_num = p_rec.get("street_num", "")
            addr_tokens = [t for t in p_rec.get("addr_tokens", "").split() if t]

            # Channel 1: Exact / Clean Name
            if name_clean:
                self.index_exact_name[(country, name_clean)].append(eid)
                if country:
                    self.index_exact_name[("", name_clean)].append(eid)

            # Channel 2: Suffix-stripped Name
            if name_stripped and name_stripped != name_clean:
                self.index_suffix_stripped[(country, name_stripped)].append(eid)
                if country:
                    self.index_suffix_stripped[("", name_stripped)].append(eid)

            # Channel 3: Significant Name Tokens (bounded)
            for t in tokens:
                if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS:
                    key = (country, t)
                    lst = self.index_name_tokens[key]
                    if len(lst) < self.max_token_postings:
                        lst.append(eid)

            # Channel 4: Phonetic (Country + Metaphone)
            if metaphone:
                self.index_phonetic[(country, metaphone[:6])].append(eid)
            elif soundex:
                self.index_phonetic[(country, soundex)].append(eid)

            # Channel 5: Postal Code + First Name Token
            if postal and tokens:
                first_tok = tokens[0]
                self.index_postal_name_token[(country, postal, first_tok[:4])].append(eid)

            # Channel 6: 4-character Prefix of clean name (bounded)
            if len(name_stripped) >= 4:
                prefix = name_stripped[:4]
                key = (country, prefix)
                lst = self.index_prefix4[key]
                if len(lst) < self.max_token_postings * 2:
                    lst.append(eid)

            # Channel 7: Street Number + Significant Address Token
            if street_num and addr_tokens:
                for at in addr_tokens[:3]:
                    if len(at) >= 4:
                        self.index_addr_street_token[(country, street_num, at)].append(eid)

            # Channel 8: Postal code alone (bounded)
            if postal:
                key = (country, postal)
                lst = self.index_postal_only[key]
                if len(lst) < 200:
                    lst.append(eid)

            # Channel 9: Significant address tokens (bounded)
            for at in addr_tokens:
                if len(at) >= 5:
                    key = (country, at)
                    lst = self.index_addr_tokens[key]
                    if len(lst) < self.max_token_postings:
                        lst.append(eid)

    def prune_frequent_tokens(self):
        """Prune excessively frequent token postings to prevent combinatorial explosion."""
        for key in list(self.index_name_tokens.keys()):
            if len(self.index_name_tokens[key]) > self.max_token_postings:
                del self.index_name_tokens[key]

        for key in list(self.index_prefix4.keys()):
            if len(self.index_prefix4[key]) > self.max_token_postings * 2:
                del self.index_prefix4[key]

        for key in list(self.index_postal_only.keys()):
            if len(self.index_postal_only[key]) > 200:
                del self.index_postal_only[key]

        for key in list(self.index_addr_tokens.keys()):
            if len(self.index_addr_tokens[key]) > self.max_token_postings:
                del self.index_addr_tokens[key]

    def block_entity(self, s1_rec: dict) -> List[str]:
        """Generate candidates for a single Source 1 entity across all channels."""
        if "name_clean" not in s1_rec:
            p_s1 = preprocess_record(s1_rec)
        else:
            p_s1 = s1_rec

        country = p_s1.get("country", "")
        name_clean = p_s1.get("name_clean", "")
        name_stripped = p_s1.get("name_suffix_stripped", "")
        tokens = [t for t in p_s1.get("name_tokens", "").split() if t]
        soundex = p_s1.get("name_soundex", "")
        metaphone = p_s1.get("name_metaphone", "")
        postal = p_s1.get("postal_code", "")
        street_num = p_s1.get("street_num", "")
        addr_tokens = [t for t in p_s1.get("addr_tokens", "").split() if t]

        candidate_scores = defaultdict(float)

        # Channel 1: Exact / Clean Name Match (high priority weight)
        for key in [(country, name_clean), ("", name_clean)]:
            if key in self.index_exact_name:
                for cid in self.index_exact_name[key]:
                    candidate_scores[cid] += 5.0

        # Channel 2: Suffix-stripped Name Match
        if name_stripped:
            for key in [(country, name_stripped), ("", name_stripped)]:
                if key in self.index_suffix_stripped:
                    for cid in self.index_suffix_stripped[key]:
                        candidate_scores[cid] += 4.0

        # Channel 3: Token Inverted Index Match
        for t in tokens:
            if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS:
                key = (country, t)
                if key in self.index_name_tokens:
                    postings = self.index_name_tokens[key]
                    w = 2.0 / (1.0 + np.log1p(len(postings)))
                    for cid in postings[:100]:
                        candidate_scores[cid] += w

        # Channel 4: Phonetic Match
        if metaphone:
            key = (country, metaphone[:6])
            if key in self.index_phonetic:
                for cid in self.index_phonetic[key][:100]:
                    candidate_scores[cid] += 1.5
        elif soundex:
            key = (country, soundex)
            if key in self.index_phonetic:
                for cid in self.index_phonetic[key][:100]:
                    candidate_scores[cid] += 1.0

        # Channel 5: Postal Code + Name Prefix
        if postal and tokens:
            key = (country, postal, tokens[0][:4])
            if key in self.index_postal_name_token:
                for cid in self.index_postal_name_token[key][:100]:
                    candidate_scores[cid] += 2.5

        # Channel 6: Prefix 4-letter
        if len(name_stripped) >= 4:
            key = (country, name_stripped[:4])
            if key in self.index_prefix4:
                postings = self.index_prefix4[key]
                w = 1.0 / (1.0 + np.log1p(len(postings)))
                for cid in postings[:100]:
                    candidate_scores[cid] += w

        # Channel 7: Street Number + Significant Address Token
        if street_num and addr_tokens:
            for at in addr_tokens[:3]:
                if len(at) >= 4:
                    key = (country, street_num, at)
                    if key in self.index_addr_street_token:
                        postings = self.index_addr_street_token[key]
                        w = 3.0 / (1.0 + np.log1p(len(postings)))
                        for cid in postings[:100]:
                            candidate_scores[cid] += w

        # Channel 8: Postal code alone (when rare)
        if postal and (country, postal) in self.index_postal_only:
            postings = self.index_postal_only[(country, postal)]
            if len(postings) <= 100:
                w = 1.5 / (1.0 + np.log1p(len(postings)))
                for cid in postings:
                    candidate_scores[cid] += w

        # Adaptive Widening if candidate count is below min_candidates
        if len(candidate_scores) < self.min_candidates:
            # Fallback 1: match on any tokens without country filter
            for t in tokens:
                if len(t) >= 3 and t not in COMMON_NAME_STOPWORDS:
                    for c_try in ["", country]:
                        if (c_try, t) in self.index_name_tokens:
                            for cid in self.index_name_tokens[(c_try, t)][:20]:
                                candidate_scores[cid] += 0.8
            # Fallback 2: match on rare address tokens
            for at in addr_tokens:
                if len(at) >= 5 and (country, at) in self.index_addr_tokens:
                    for cid in self.index_addr_tokens[(country, at)][:10]:
                        candidate_scores[cid] += 0.5

        if not candidate_scores:
            return []

        # Rank candidates by accumulated channel score
        sorted_candidates = sorted(
            candidate_scores.keys(),
            key=lambda cid: candidate_scores[cid],
            reverse=True,
        )

        return sorted_candidates[: self.max_candidates]

    def block_all(
        self, s1_records: Iterable[dict]
    ) -> Dict[str, List[str]]:
        """Run blocking across all Source 1 records."""
        candidates_map = {}
        for rec in s1_records:
            eid = rec.get("entity_id", "")
            cand_list = self.block_entity(rec)
            candidates_map[eid] = cand_list
        return candidates_map
