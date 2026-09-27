"""
Data Preprocessing and Entity Normalization
Amazon ML Challenge 2026

Handles business name and address normalization, legal suffix standardization,
phonetic encoding (Soundex/Metaphone), abbreviation expansions (US, India, France/generic),
address component extraction (postal code, street number, landmarks),
and explicit missingness indicators.
"""

import re
import unicodedata
from typing import Dict, List, Optional, Set, Tuple
import jellyfish


# Pure legal entity corporate forms that can be stripped
CORPORATE_LEGAL_SUFFIXES = {
    "corporation": "corp",
    "corp": "corp",
    "incorporated": "inc",
    "inc": "inc",
    "limited liability company": "llc",
    "llc": "llc",
    "limited liability partnership": "llp",
    "llp": "llp",
    "private limited": "pvt ltd",
    "pvt ltd": "pvt ltd",
    "private ltd": "pvt ltd",
    "pvt": "pvt",
    "private": "pvt",
    "limited": "ltd",
    "ltd": "ltd",
    "company": "co",
    "co": "co",
    "sa": "sa",        # common French/European legal form
    "sarl": "sarl",    # common French legal form
    "sas": "sas",      # common French legal form
    "snc": "snc",      # common French legal form
}

# Broader mapping for standardizing vocabulary
LEGAL_SUFFIX_MAP = dict(CORPORATE_LEGAL_SUFFIXES)
LEGAL_SUFFIX_MAP.update({
    "enterprises": "enterprise",
    "enterprise": "enterprise",
    "technologies": "tech",
    "technology": "tech",
    "tech": "tech",
    "solutions": "solutions",
    "services": "services",
    "holdings": "holdings",
    "holding": "holdings",
    "group": "group",
    "industries": "industries",
    "industry": "industries",
    "international": "intl",
    "intl": "intl",
    "center": "center",
    "centre": "center",
})

# Regex to match trailing legal suffix (only corporate forms)
LEGAL_SUFFIX_PATTERN = re.compile(
    r"\b("
    + "|".join(
        re.escape(k)
        for k in sorted(CORPORATE_LEGAL_SUFFIXES.keys(), key=len, reverse=True)
    )
    + r")\b[\s\.,]*$",
    re.IGNORECASE,
)

COMMON_NAME_STOPWORDS = {
    "the", "a", "an", "of", "and", "&", "for", "in", "on", "at", "by",
    "corp", "inc", "llc", "llp", "ltd", "pvt", "co", "enterprise",
    "group", "tech", "solutions", "services", "sa", "sarl", "sas",
}


# ---------------------------------------------------------------------------
# Address Abbreviation Standardizations
# ---------------------------------------------------------------------------

ADDRESS_ABBREV_MAP = {
    "rd": "road",
    "rd.": "road",
    "st": "street",
    "st.": "street",
    "ave": "avenue",
    "ave.": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "blvd.": "boulevard",
    "dr": "drive",
    "dr.": "drive",
    "ln": "lane",
    "ln.": "lane",
    "ct": "court",
    "ct.": "court",
    "pl": "place",
    "pl.": "place",
    "pkwy": "parkway",
    "pkwy.": "parkway",
    "hwy": "highway",
    "hwy.": "highway",
    "fwy": "freeway",
    "sq": "square",
    "apt": "apartment",
    "apt.": "apartment",
    "ste": "suite",
    "ste.": "suite",
    "fl": "floor",
    "fl.": "floor",
    "bldg": "building",
    "no": "number",
    "no.": "number",
    "sec": "sector",
    "sec.": "sector",
    "opp": "opposite",
    "opp.": "opposite",
    "nr": "near",
    "nr.": "near",
    "b/h": "behind",
    "bh": "behind",
    "1st": "first",
    "2nd": "second",
    "3rd": "third",
    "4th": "fourth",
    "5th": "fifth",
    # French address terms (open-set support)
    "bd": "boulevard",
    "av.": "avenue",
    "r.": "rue",
    "rue": "rue",
    "rte": "route",
    "route": "route",
    "chemin": "chemin",
    "impasse": "impasse",
    "allée": "allee",
    "allee": "allee",
    "faubourg": "faubourg",
    "zac": "zac",
    "zi": "zi",
}

LANDMARK_PATTERN = re.compile(
    r"\b(near|opp|opposite|behind|beside|adj|adjacent|next to|close to|in front of)\s+([^,]+)",
    re.IGNORECASE,
)

# Postal code patterns
US_ZIP_PATTERN = re.compile(r"\b(\d{5})(?:-\d{4})?\b")
INDIA_PIN_PATTERN = re.compile(r"\b([1-9]\d{5})\b")
FRANCE_POSTAL_PATTERN = re.compile(r"\b((?:0[1-9]|[1-8]\d|9[0-8])\d{3})\b")
GENERIC_POSTAL_PATTERN = re.compile(r"\b(\d{4,6})\b")

STREET_NUM_PATTERN = re.compile(r"\b(?:#|no\.?|apt\.?|suite\s*)?(\d{1,5}(?:[a-zA-Z]|-\d+)?)\b", re.IGNORECASE)


def normalize_unicode(text: str) -> str:
    """Normalize unicode characters (NFKD decomposition and ASCII compatibility)."""
    if not text:
        return ""
    # NFKD normalizes accents, umlauts, e.g. Drükor -> Drukor
    norm = unicodedata.normalize("NFKD", str(text))
    return norm.encode("ascii", "ignore").decode("utf-8")


def clean_basic_text(text: str) -> str:
    """Lowercase, remove special characters, unify & to and, collapse whitespace."""
    if not text:
        return ""
    s = normalize_unicode(text).lower()
    s = s.replace("&", " and ")
    s = s.replace("@", " at ")
    s = s.replace("/", " ")
    s = s.replace("-", " ")
    s = s.replace("_", " ")
    # Normalize acronyms like s.a.r.l. -> sarl, l.l.c. -> llc, u.s. -> us
    s = re.sub(r"\b([a-z])\.", r"\1", s)
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def normalize_business_name(name_raw: str) -> Dict[str, str]:
    """Produce normalized name variants:
    - clean_name: basic cleaned name
    - legal_normalized: suffixes standardized
    - suffix_stripped: legal suffixes removed entirely
    - stem_domain: if name is a domain (e.g. maurewilliamscolombier.com)
    - soundex: phonetic Soundex
    - metaphone: phonetic Metaphone
    - tokens: space-separated significant tokens (stopwords removed)
    """
    if not name_raw:
        return {
            "name_clean": "",
            "name_legal_norm": "",
            "name_suffix_stripped": "",
            "name_tokens": "",
            "name_soundex": "",
            "name_metaphone": "",
        }

    clean = clean_basic_text(name_raw)

    # Domain name handling (e.g. example.com -> example)
    domain_match = re.search(r"\b([a-zA-Z0-9\-]+)\.(?:com|org|net|in|co\.in|io|fr|biz|info)\b", clean)
    if domain_match:
        domain_stem = domain_match.group(1).replace("-", " ")
        clean = f"{clean} {domain_stem}".strip()

    # Standardize legal suffixes
    words = clean.split()
    norm_words = []
    for w in words:
        w_strip = w.rstrip(".,")
        if w_strip in LEGAL_SUFFIX_MAP:
            norm_words.append(LEGAL_SUFFIX_MAP[w_strip])
        else:
            norm_words.append(w_strip)
    legal_norm = " ".join(norm_words)

    # Strip trailing legal suffixes
    suffix_stripped = legal_norm
    changed = True
    while changed:
        prev = suffix_stripped
        suffix_stripped = LEGAL_SUFFIX_PATTERN.sub("", suffix_stripped).strip()
        changed = (prev != suffix_stripped)

    # Significant tokens
    sig_tokens = [
        w for w in suffix_stripped.split()
        if w not in COMMON_NAME_STOPWORDS and len(w) > 1
    ]

    # Phonetics on suffix-stripped or legal-norm
    phonetic_target = suffix_stripped if suffix_stripped else legal_norm
    try:
        soundex_val = jellyfish.soundex(phonetic_target) if phonetic_target else ""
    except Exception:
        soundex_val = ""

    try:
        metaphone_val = jellyfish.metaphone(phonetic_target) if phonetic_target else ""
    except Exception:
        metaphone_val = ""

    return {
        "name_clean": clean,
        "name_legal_norm": legal_norm,
        "name_suffix_stripped": suffix_stripped,
        "name_tokens": " ".join(sig_tokens),
        "name_soundex": soundex_val,
        "name_metaphone": metaphone_val,
    }


def normalize_address(address_raw: str, country: str = "") -> Dict[str, str]:
    """Produce normalized address variants and extract address components:
    - addr_clean: basic cleaned and abbreviation-expanded address
    - postal_code: extracted PIN/Zip/Postal code
    - street_num: extracted street/unit number
    - landmark: extracted landmark phrase (if any)
    - addr_tokens: significant address tokens
    """
    if not address_raw:
        return {
            "addr_clean": "",
            "postal_code": "",
            "street_num": "",
            "landmark": "",
            "addr_tokens": "",
        }

    clean = clean_basic_text(address_raw)

    # Extract landmark
    landmark_val = ""
    lm_match = LANDMARK_PATTERN.search(clean)
    if lm_match:
        landmark_val = lm_match.group(0).strip()

    # Extract postal code based on country if known, else generic
    c_upper = country.strip().upper() if country else ""
    postal_val = ""
    if c_upper == "US":
        m = US_ZIP_PATTERN.search(address_raw)
        if m:
            postal_val = m.group(1)
    elif c_upper == "INDIA":
        m = INDIA_PIN_PATTERN.search(address_raw)
        if m:
            postal_val = m.group(1)
    elif c_upper == "FRANCE":
        m = FRANCE_POSTAL_PATTERN.search(address_raw)
        if m:
            postal_val = m.group(1)

    # Fallback if country-specific didn't match or country is unseen
    if not postal_val:
        m = GENERIC_POSTAL_PATTERN.search(address_raw)
        if m:
            postal_val = m.group(1)

    # Street number extraction
    street_num_val = ""
    s_match = STREET_NUM_PATTERN.search(clean)
    if s_match:
        street_num_val = s_match.group(1)

    # Expand abbreviations
    words = clean.split()
    expanded_words = []
    for w in words:
        w_strip = w.rstrip(".,")
        if w_strip in ADDRESS_ABBREV_MAP:
            expanded_words.append(ADDRESS_ABBREV_MAP[w_strip])
        else:
            expanded_words.append(w_strip)
    addr_expanded = " ".join(expanded_words)

    # Significant address tokens (excluding pure numbers and tiny tokens)
    addr_tokens = [w for w in addr_expanded.split() if len(w) > 2 and not w.isdigit()]

    return {
        "addr_clean": addr_expanded,
        "postal_code": postal_val,
        "street_num": street_num_val,
        "landmark": landmark_val,
        "addr_tokens": " ".join(addr_tokens),
    }


def preprocess_record(record: Dict[str, str]) -> Dict[str, str]:
    """Enrich a single raw entity record dictionary with all normalized fields."""
    name_raw = record.get("business_name", "") or ""
    addr_raw = record.get("business_address", "") or ""
    country = record.get("country", "") or ""

    name_dict = normalize_business_name(name_raw)
    addr_dict = normalize_address(addr_raw, country)

    out = {
        "entity_id": record.get("entity_id", ""),
        "business_name_raw": name_raw,
        "business_address_raw": addr_raw,
        "country": country,
    }
    out.update(name_dict)
    out.update(addr_dict)
    return out
