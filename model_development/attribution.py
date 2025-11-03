#!/usr/bin/env python3
"""
Attribution Model – PERSON-based conversion (uses precomputed people_by_row),
strong non-person filtering (no global combining), reliability weighting, and
tag-level PCA table.

Conversions are judged off the CLEANED people_by_row only (no global canonicalization).
Within each row ONLY: if a single-token surname appears and there is exactly one
matching full name with that surname in the same row, we upgrade it to the full name.

Inputs
------
  /Users/annaglass/capstone/capstone/data_storage/processed_data/processed_with_people_emotion.parquet

Outputs
-------
  data_storage/final_data/attribution_dataset.parquet                  (dimension-level credits)
  data_storage/final_data/final_dataset_with_attribution.parquet       (conversion-only rows; FINAL)
  data_storage/final_data/tagname_pca_ready.csv
  data_storage/final_data/persons_detected.csv                         (cleaned, per-name counts)
  data_storage/final_data/persons_by_row.csv                           (cleaned per-row list)

Env toggles (optional)
----------------------
  MAX_STATES_PER_DIM=800
  MIN_STATE_COUNT=8
  FAST_MODE_THRESHOLD=600
  ENABLE_TERMS=0
  MAX_KEY_TERMS=500
  TERMS_CHUNK_SIZE=150
"""

from __future__ import annotations
import os, re, gc, math
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve

# =============================================================================
# Paths & Core Columns
# =============================================================================
ROOT = Path("/Users/annaglass/capstone/capstone")
DATA_PARQUET = ROOT / "data_storage" / "processed_data" / "processed_with_people_emotion.parquet"

OUT_DIR  = ROOT / "data_storage" / "final_data"
OUT_FILE = OUT_DIR / "attribution_dataset.parquet"

KEYWORDS = OUT_DIR / "top_1000_keywords.csv"
BIGRAMS  = OUT_DIR / "top_1000_bigrams.csv"

PERSONS_FILE        = OUT_DIR / "persons_detected.csv"
PERSONS_BY_ROW_FILE = OUT_DIR / "persons_by_row.csv"
FULL_DATASET_FILE   = OUT_DIR / "final_dataset_with_attribution.parquet"
TAG_PCA_FILE        = OUT_DIR / "tagname_pca_ready.csv"

PATH_KEY   = "tag_name"
TIME_COL   = "seq_index"
WEIGHT_COL = "vipr_weight"
CONV       = "<CONV>"
OTHER_LABEL = "__OTHER__"

# =============================================================================
# Dimensions
# =============================================================================
DIM_CATEGORICAL_ALL = (
    "tag_name",
    "source_feed_name",
    "feed_name",
    "author_name",
    "source_type",
    "publication_name",
    "publisher_name",
    "sentiment_band",
    "channel_name"
)
DIM_NUMERIC_ALL = ("circulation_size", "sentiment_score")  # binned → 1..5 ints

# =============================================================================
# Speed / Memory knobs
# =============================================================================
MAX_STATES_PER_DIM = int(os.getenv("MAX_STATES_PER_DIM", "800"))
MIN_STATE_COUNT     = int(os.getenv("MIN_STATE_COUNT", "8"))
FAST_MODE_THRESHOLD = int(os.getenv("FAST_MODE_THRESHOLD", "600"))

ENABLE_TERMS         = os.getenv("ENABLE_TERMS", "0") == "1"  # default OFF
MAX_KEY_TERMS        = int(os.getenv("MAX_KEY_TERMS", "500"))
TERMS_CHUNK_SIZE     = int(os.getenv("TERMS_CHUNK_SIZE", "150"))
FORCE_FAST_FOR_TERMS = True
PRINT_EVERY          = 1

# =============================================================================
# Reliability boost
# =============================================================================
RELIABLE_TYPES = {
    "National News","Government","Wires",
    "General News","Regional News","Trade News",
}
RELIABLE_WEIGHT_BOOST = 0.25

# =============================================================================
# Utilities
# =============================================================================
def _to_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")

def _downcast_inplace(df: pd.DataFrame) -> None:
    """
    Conservative memory downcast:
      - numeric -> smaller numeric
      - object -> category, EXCEPT columns that must remain free text
    """
    DO_NOT_CATEGORY = {"headline","article_body","people_by_row","emotion_body"}
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif pd.api.types.is_object_dtype(df[c]):
            if c not in DO_NOT_CATEGORY:
                try:
                    nunique = df[c].nunique(dropna=True)
                    if nunique and nunique < 0.8 * len(df):
                        df[c] = df[c].astype("category")
                except Exception:
                    pass

def add_quantile_bins(df: pd.DataFrame, col: str, bins: int = 5) -> str:
    series = _to_numeric(df[col])
    ranks = series.rank(method="first")
    labels = [f"{col.upper()}_Q{i}" for i in range(1, bins+1)]
    new_col = f"{col}_bin"
    df[new_col] = pd.qcut(ranks, bins, labels=labels, duplicates="drop").astype("string")
    return new_col

def simplify_bin_to_int(df: pd.DataFrame, bin_col: str) -> None:
    """
    Convert label bins like 'CIRCULATION_SIZE_Q3' / 'SENTIMENT_SCORE_Q5' into plain integers 1..5 in-place.
    """
    if bin_col in df.columns:
        nums = df[bin_col].astype("string").str.extract(r"(\d+)$", expand=False)
        df[bin_col] = pd.to_numeric(nums, errors="coerce").astype("Int64")

def add_ratings(tbl: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if tbl.empty:
        return tbl.assign(rating=[], rating_pct=[])
    out = []
    for _, g in tbl.groupby(group_col, group_keys=False):
        cs = g["credit_share"].astype(float).fillna(0.0)
        pct = cs.rank(method="average", pct=True).fillna(0.0)
        try:
            qbins = pd.qcut(pct, 5, labels=[1,2,3,4,5], duplicates="drop")
            if getattr(qbins, "dtype", None) == "category" and qbins.cat.categories.size < 5:
                rating = np.ceil(pct * 5.0).astype(int).clip(1, 5)
            else:
                rating = qbins.astype(int)
        except ValueError:
            rating = np.ceil(pct * 5.0).astype(int).clip(1, 5)
        g = g.assign(rating=rating)
        g["_min"] = g.groupby("rating")["credit_share"].transform("min")
        g["_max"] = g.groupby("rating")["credit_share"].transform("max")
        denom = (g["_max"] - g["_min"]).replace(0, 1.0)
        g["rating_pct"] = ((g["credit_share"] - g["_min"]) / denom).clip(0, 1)
        g = g.drop(columns=["_min","_max"])
        out.append(g)
    return pd.concat(out, ignore_index=True)

def _row_weight_with_reliability(r: pd.Series) -> float:
    base = float(_to_numeric(r.get(WEIGHT_COL, 1.0)) or 0.0)
    stype = str(r.get("source_type", "") or "")
    mult = 1.0 + RELIABLE_WEIGHT_BOOST if stype in RELIABLE_TYPES else 1.0
    return base * mult

# =============================================================================
# People cleaning (NO global combining) + per-row surname→full upgrade + exports
# =============================================================================
EXCLUDE_LOWER = {
    # diseases / medical conditions
    "covid","covid-19","covid19","sars-cov-2","h1n1","influenza","influenza a","measles","polio","mpox",
    "fentanyl","listeria","bird flu","gonorrhea","xylazine","rubella","diarrhea","hepatitis c",
    "chikungunya","tick-borne","tick","acinetobacter","e. coli","malaria","omicron","chikv-en",
    "mosquito","marijuana","lassa","lassa fever","ebola","stargardt","culex",
    "measles outbreak","measles vaccine","measles cases","measles surge","measles alert",
    "measles confirmed","measles grows","measles-rubella","measles case","measles urgent",
    "polio vaccine","mpox vaccine",
    # medical/vaccine terms
    "vaccine","vaccines","vaccine hesitancy","vaccine changes","vaccine boycotts","vaccine choice",
    "vaccine development","vaccine hesitant","vaccine type","vaccine malaria","vaccines submit",
    "vaccines size","breakthrough therapy","gene therapy","chikungunya vaccine","chikungunya outbreak",
    "kidney diseases","allergy immunol","neurology","metabolic","hygiene","legionnaires",
    "discontinue nuzyra","enamel hypoplasia","alcohol disorder","alcohol concern","alcohol bristol",
    "alcohol springfield","alcoholics anonymous","influenza vaccine","tdap","carbapenem",
    "chi-square","chi-squared","likert","likert scale","nirsevimab",
    # companies/products
    "pfizer","moderna","jynneos","glaxosmithkline","sanofi","novo nordisk","tonix","paratek",
    "purdue pharma","zepbound","elebsiran","paxlovid","brii bio","brii limited","alpha genesis",
    "trader joe","pharma","medicines","tonix medicines","novartis","walgreens","medscape",
    "mercer","licensee mdpi","biotech","biotech hubs",
    # web/tech/urls
    "https","download","email","email copy","email alerts","email address","email facebook",
    "email print","index.php","zoom","youtube","youtube shorts","youtube channel","meta","org",
    "sci","view","photo","photo credit","photo courtesy","mobile apps","getty images","getty",
    "news-releases","issuewire","bloomberg","politico","plos",
    # generic non-persons
    "meeting","conference","summit","session","panel","committee","hearing","event","gala","forum",
    "ceremony","group","team","class","program","organization","party","association",
    "worldwide","founder","children","advocacy","escape","lung","fahrenheit","award","awards",
    "data","challenges","lifestyle","schedule","breakthrough","primary logo","threaten",
    "associate dean","preparedness","highs","progress","deaths","figure","figures","fig","figs",
    "network","science","alert","lifestyle medicine","lifestyle coaches","metadata","data-vis",
    "bird","boar","wildfires","tribal","older","africans","bangladeshi","coom","valentine",
    "wash","gracia","cook","gounder","clade","spectrum","rice","formula","schedule iii",
    # fictional/unlikely
    "superman","lex luthor","jesus","jesus christ","dad",
    # organizations
    "gavi","nyu langone","alaska native","alaska natives","disease central","mnu-topics",
    "organizations grants","scientology network","scientologists","scientolgytv","alliance melbourne",
    "alliancebernstein","without borders","unicef","investor relations","publicis groupe","meltwater",
    "refinitiv","westlaw","kantar","wbz newsradio","cloudberry","cloudberry health",
    "compare-autoinsurance","brigham","ayushman bharat","ayushman","lok sabha","twin peaks",
    "xinhua","rdc screening","healthbeat","biotech hubs",
    # places (common ones)
    "khan younis","khan yunis","malawi","goma","rafah","tamil nadu","mali","nepal","sri lanka",
    "burkina faso","mauritius","niger","san diego","rajasthan","haryana","albuquerque",
    # single letters and abbreviations (will be filtered by length check too)
    "al","j","l","g","n","y","x","w","z","b","t","s","h","a",
    "tel","en","res","sl","cv","mar","ma","bin","prep","drs","dr","mm","clin","nutr","dl","mou",
    "li","kefir",
    # Anna's examples (lowercased)
    "appili","appili therapeutics signs","distribution","apli","adtx","appili therapeutics",
    "appili share","appili shares","appili shareholder","buyer","closing","llp","bird fu",
    "aspek promotif dan","twins","twin birth weight","isabella, valentine's day","influenza a(h5n1",
    "haemophilus influenzae","top story","damage","discusses breakthrough","fda roundup","antifungal",
    "clsi m27-a3","vaccine enabling kit","caspofungin","anchorage mega","egr6","bigdye","supplementary fig.",
    "y132f","hospital","ethics","c. auris","tbiaa","homemaker mode","pennymuster.com",
    "everyone, moutaz kotob, park ave, ste 1500","instagram","twitter","facebook","google","x",
    "sprouted mat","r- oceanside",
}

SOFT_EXCLUDES = (
    # orgs/places/things commonly embedded in tokens
    "university","college","institute","hospital","medical center","center","centre","committee",
    "department","ministry","laboratory","lab","foundation","association","academy","watch","watcher",
    "pharmacy","school","program","project","trial","policy","laboratories",
    "airport","county","province","prefecture","city","town","district","state","territory",
    "franchisees","opportunities","surveillance","sequencing","omics","genome","genomic",
    "®","™","©","http://","https://",".com",".org",".gov",".edu",".net"
)

TITLE_RE = re.compile(r'^(dr\.?|mr\.?|mrs\.?|ms\.?|prof\.?|president|pres\.?|senator|sen\.?|rep\.?|representative|gov\.?|governor)\s+', re.I)
SUFFIX_RE = re.compile(r'\s+(jr\.?|sr\.?|phd|md|rn|esq\.?)$', re.I)
PUNCT_INNER = re.compile(r"[^\w\s\-\.’']")
SPLITTER = re.compile(r"[,\|;/]+")

# possessive stripping (Biden’s / Biden ' s / Bidens’ → Biden)
POSSESSIVE_RE = re.compile(r"(?:\s*['’]\s*s|\s*s\s*['’])\s*$", re.I)
def _strip_possessive(s: str) -> str:
    if not s:
        return ""
    return POSSESSIVE_RE.sub("", str(s)).strip()

def _looks_like_url_or_code(s: str) -> bool:
    s2 = s.strip().lower()
    return s2.startswith(("http://","https://")) or any(t in s2 for t in (".com",".org",".gov",".edu",".net","js.id","®"))

def _name_core(s: str) -> str:
    """
    Reduce a raw token to 'First Last' (or single token). Safe for empties.
    Case/possessive-insensitive: strip possessives before normalization.
    """
    if not s or not str(s).strip():
        return ""
    s = _strip_possessive(str(s))
    s = TITLE_RE.sub("", s).strip()
    s = SUFFIX_RE.sub("", s).strip()
    s = PUNCT_INNER.sub(" ", s)
    s = re.sub(r"\s+", " ", s).strip()
    if not s:
        return ""
    # remove isolated middle initials (A.) → A
    s = re.sub(r"\b([A-Za-z])\.\b", r"\1", s)
    parts = s.split()
    if not parts:
        return ""
    # keep only alphabetic-ish tokens or hyphenated names
    parts = [p for p in parts if re.fullmatch(r"[A-Za-z][A-Za-z\-’.']*", p)]
    if not parts:
        return ""
    if len(parts) >= 2:
        return f"{parts[0]} {parts[-1]}"
    return parts[0]

def _is_non_person_token(tok: str) -> bool:
    if not tok or not str(tok).strip():
        return True
    t = _strip_possessive(tok.strip())
    t_low = t.lower()

    # Single letter tokens (very likely not a person)
    if len(t) == 1 and t.isalpha():
        return True
    
    # Pattern like "J. Clin", "A. Al", etc. (journal citations, not names)
    if re.match(r"^[A-Za-z]\.\s+[A-Z]", t) and len(t) < 10:
        return True
    
    # Pattern like "J. Clin.", "A. Al.", etc. with trailing period
    if re.match(r"^[A-Za-z]\.\s+[A-Z][a-z]+\.?$", t) and len(t) < 15:
        return True
    
    if _looks_like_url_or_code(t):
        return True
    if t_low in EXCLUDE_LOWER:
        return True
    if any(key in t_low for key in SOFT_EXCLUDES):
        return True
    if len(t) >= 3 and t.upper() == t and not re.search(r"[a-z]", t):  # ACRONYMS
        return True
    if re.match(r"^[•‐\-—–↑↓→←±✓✕✖️🗓˜]+", t):  # bullets/arrows
        return True
    if sum(c.isdigit() for c in t) >= 2:
        return True
    
    # Filter place names starting with "Al-" (common Arabic place prefix)
    if re.match(r"^al-", t_low) and len(t.split()) == 1:
        return True
    
    # Filter URLs/domains more aggressively
    if re.search(r"\.(com|org|gov|edu|net|php)$", t_low):
        return True
    
    return False

def _surname_lower(name: str) -> str:
    core = _name_core(name)
    toks = core.split()
    return toks[-1].lower() if toks else ""

def _canonicalize_name(name: str) -> str:
    """
    Normalize a name to a canonical form (case-insensitive).
    Maps specific surnames and case variations to canonical full names:
    - "trump" -> "Donald Trump"
    - "biden" -> "Joe Biden"
    - "kennedy" -> "Robert Kennedy"
    - "harris" -> "Kamala Harris"
    
    For other names, converts to Title Case for full names.
    """
    if not name or not str(name).strip():
        return ""
    
    name_lower = name.strip().lower()
    
    # Specific surname mappings
    if name_lower == "trump":
        return "Donald Trump"
    elif name_lower == "biden":
        return "Joe Biden"
    elif name_lower == "kennedy":
        return "Robert Kennedy"
    elif name_lower == "harris":
        return "Kamala Harris"
    
    # For full names, normalize to Title Case
    parts = name.strip().split()
    if len(parts) >= 2:
        # Title case each part (handle hyphenated names)
        title_parts = []
        for part in parts:
            if '-' in part:
                # Handle hyphenated names like "Mary-Jane"
                hyphen_parts = [p.capitalize() for p in part.split('-')]
                title_parts.append('-'.join(hyphen_parts))
            else:
                title_parts.append(part.capitalize())
        return " ".join(title_parts)
    elif len(parts) == 1:
        # Single token: capitalize first letter
        return parts[0].capitalize()
    
    return name.strip()

def _row_clean_names(cell: str) -> list[str]:
    """
    Split -> drop non-people -> normalize to 'First Last' or single token.
    Canonicalize names (case-insensitive) and apply specific mappings.
    Deduplicate case-insensitively using canonical forms.
    Then, within the row ONLY: if a single-token surname has exactly one
    matching full name with same surname, upgrade it to that full name.
    """
    if pd.isna(cell) or not str(cell).strip():
        return []

    # 1) tokenization + strong filtering + core reduction
    parts = [p.strip() for p in SPLITTER.split(str(cell)) if p.strip()]
    prelim, seen_lower = [], set()
    for p in parts:
        if _is_non_person_token(p):
            continue
        core = _name_core(p)
        if not core:
            continue
        # drop single tokens that are common non-person words
        if len(core.split()) == 1 and core.lower() in EXCLUDE_LOWER:
            continue
        
        # Canonicalize the name (case-insensitive, applies specific mappings)
        canonical = _canonicalize_name(core)
        if not canonical:
            continue
        
        # Check if canonicalized name is still a non-person (case-insensitive)
        canonical_lower = canonical.lower()
        if canonical_lower in EXCLUDE_LOWER:
            continue
        if any(key in canonical_lower for key in SOFT_EXCLUDES):
            continue
        
        # Deduplicate case-insensitively using canonical form
        key = canonical_lower
        if key not in seen_lower:
            seen_lower.add(key)
            prelim.append(canonical)

    if not prelim:
        return []

    # 2) per-row surname→full upgrade (no global combining)
    #    build surname -> set(full names) present in row
    surname_to_full = {}
    for n in prelim:
        toks = n.split()
        if len(toks) >= 2:
            sn = toks[-1].lower()
            surname_to_full.setdefault(sn, set()).add(n)

    upgraded = []
    for n in prelim:
        toks = n.split()
        if len(toks) == 1:
            sn = toks[0].lower()
            candidates = surname_to_full.get(sn, set())
            if len(candidates) == 1:
                # upgrade single surname to the unique full name in THIS row
                n = next(iter(candidates))
        upgraded.append(n)

    # final de-dup (preserve order) after upgrades, case-insensitive
    outs, seen_lower2 = [], set()
    for n in upgraded:
        k = n.lower()
        if k not in seen_lower2:
            seen_lower2.add(k)
            outs.append(n)

    return outs

def detect_persons_and_flag_conversion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean 'people_by_row' with case-insensitive canonicalization.
    Applies specific mappings (trump->Donald Trump, biden->Joe Biden, kennedy->Robert Kennedy).
    Set has_person / is_conversion directly from cleaned+upgraded list.
    Write persons CSVs.
    """
    if "people_by_row" not in df.columns:
        raise ValueError("people_by_row column is required but not found in the input dataset.")

    cleaned_lists = df["people_by_row"].apply(_row_clean_names)
    df["people_by_row"] = cleaned_lists.apply(lambda lst: ", ".join(lst)).astype("string")
    df["has_person"] = cleaned_lists.apply(lambda x: len(x) > 0)
    df["is_conversion"] = df["has_person"]

    # Exports based on cleaned lists
    flat = [p for lst in cleaned_lists for p in lst]
    if flat:
        agg = (pd.Series(flat, dtype="string")
               .value_counts()
               .rename_axis("person")
               .reset_index(name="count"))
    else:
        agg = pd.DataFrame(columns=["person","count"])
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(PERSONS_FILE, index=False)

    persons_by_row = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "tag_name": df["tag_name"].astype("string") if "tag_name" in df.columns else pd.Series([""]*len(df), dtype="string"),
        "persons": df["people_by_row"],
        "has_person": df["has_person"].astype(int)
    })
    persons_by_row.to_csv(PERSONS_BY_ROW_FILE, index=False)

    print(f"[conv] PERSON rows (cleaned) = {int(df['has_person'].sum())} / {len(df)}")
    print(f"[conv] Saved unique persons -> {PERSONS_FILE}")
    print(f"[conv] Saved per-row persons -> {PERSONS_BY_ROW_FILE}")
    return df

# =============================================================================
# Markov core
# =============================================================================
def _row_normalized_transition(paths: List[List[str]], weights: np.ndarray, states: List[str]) -> csr_matrix:
    idx = {s: i for i, s in enumerate(states)}
    START = len(states)
    CONV_ID = len(states) + 1
    n_all = len(states) + 2

    src, dst, wt = [], [], []
    for seq, w in zip(paths, weights):
        if not seq:
            continue
        first = seq[0]
        a_id = START
        b_id = idx.get(first, CONV_ID if first == CONV else None)
        if b_id is not None:
            src.append(a_id); dst.append(b_id); wt.append(w)
        for a, b in zip(seq[:-1], seq[1:]):
            a_id = idx.get(a, CONV_ID if a == CONV else None)
            b_id = idx.get(b, CONV_ID if b == CONV else None)
            if a_id is None or b_id is None:
                continue
            src.append(a_id); dst.append(b_id); wt.append(w)
    T = coo_matrix((wt, (src, dst)), shape=(n_all, n_all)).tocsr()
    rs = np.asarray(T.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return T.multiply(1.0 / rs[:, None]).tocsr()

def markov_from_paths(paths: List[List[str]], weights: np.ndarray,
                      max_states: int = MAX_STATES_PER_DIM,
                      min_state_count: int = MIN_STATE_COUNT,
                      force_fast: bool = True) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    # frequency prune/merge to OTHER
    freq = {}
    for seq in paths:
        for s in seq:
            if s == CONV: continue
            freq[s] = freq.get(s, 0) + 1

    states = [s for s,c in freq.items() if s != CONV and c >= min_state_count]
    rare = [s for s,c in freq.items() if s != CONV and c <  min_state_count]

    if rare:
        rare_set = set(rare)
        new_paths = []
        for seq in paths:
            mapped = [OTHER_LABEL if (t in rare_set) else t for t in seq]
            collapsed, last = [], None
            for t in mapped:
                if t != last:
                    collapsed.append(t); last = t
            new_paths.append(collapsed)
        paths = new_paths
        # recompute freq quickly
        freq = {}
        for seq in paths:
            for s in seq:
                if s == CONV: continue
                freq[s] = freq.get(s, 0) + 1
        states = [s for s in freq.keys() if s != CONV]

    if len(states) > max_states:
        states_sorted = sorted(states, key=lambda s: freq[s], reverse=True)
        keep = set(states_sorted[:max_states-1])  # leave slot for OTHER
        keep.add(OTHER_LABEL)
        new_paths = []
        for seq in paths:
            mapped = [t if (t in keep or t == CONV) else OTHER_LABEL for t in seq]
            collapsed, last = [], None
            for t in mapped:
                if t != last:
                    collapsed.append(t); last = t
            new_paths.append(collapsed)
        paths = new_paths
        states = sorted(list(keep - {OTHER_LABEL})) + [OTHER_LABEL]

    states = sorted([s for s in set(states) if s != CONV])
    if not states:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    P = _row_normalized_transition(paths, weights, states)
    START = len(states); CONV_ID = len(states) + 1

    order = np.r_[ [START], np.arange(len(states)), [CONV_ID] ]
    P = P[order][:, order].tocsr()

    t_slice = slice(1, 1 + len(states))
    a_slice = slice(1 + len(states), None)
    Q = P[t_slice, t_slice].tocsr()
    R = P[t_slice, a_slice].tocsr()
    P0 = P[0, t_slice].tocsr()

    n = Q.shape[0]
    I_csr = identity(n, format="csr")
    A_csc = (I_csr - Q).tocsc()
    rhs_dense = np.asarray(P0.T.toarray()).ravel()
    y = spsolve(A_csc.T, rhs_dense)          # visit prob

    if force_fast or (len(states) >= FAST_MODE_THRESHOLD):
        r_col = np.asarray(R.toarray()).ravel() if R.shape[1] == 1 else np.asarray(R[:,0].toarray()).ravel()
        credits = np.maximum(0.0, y * r_col)
        total = credits.sum()
        share = credits / total if total > 0 else np.zeros_like(credits)
        return (
            pd.DataFrame({"state": states, "credit": credits, "credit_share": share})
              .sort_values("credit", ascending=False, ignore_index=True)
        )

    # Exact path (rare; tiny problems)
    credits = np.zeros(len(states), dtype=float)
    baseline = float((csr_matrix(y.reshape(1, -1)) @ R).toarray()[0, 0])
    for i in range(Q.shape[0]):
        Q2 = Q.copy(); R2 = R.copy()
        if Q2.indptr[i] != Q2.indptr[i+1]:
            Q2.data[Q2.indptr[i]:Q2.indptr[i+1]] = 0.0
        if R2.indptr[i] != R2.indptr[i+1]:
            R2.data[R2.indptr[i]:R2.indptr[i+1]] = 0.0
        A2 = (identity(Q2.shape[0], format="csr") - Q2).tocsc()
        y2 = spsolve(A2.T, rhs_dense)
        conv2 = float((csr_matrix(y2.reshape(1, -1)) @ R2).toarray()[0, 0])
        credits[i] = max(0.0, baseline - conv2)

    total = credits.sum()
    share = credits / total if total > 0 else np.zeros_like(credits)
    return (
        pd.DataFrame({"state": states, "credit": credits, "credit_share": share})
          .sort_values("credit", ascending=False, ignore_index=True)
    )

# =============================================================================
# Path builders
# =============================================================================
def build_paths(df: pd.DataFrame, state_col: str) -> Tuple[List[List[str]], np.ndarray]:
    cols = [PATH_KEY, TIME_COL, WEIGHT_COL, "source_type", "is_conversion"]
    if state_col not in cols:
        cols.append(state_col)
    dff = df[cols].dropna(subset=[PATH_KEY]).copy()
    dff = dff.sort_values([PATH_KEY, TIME_COL])

    paths: List[List[str]] = []
    weights = []
    for _, g in dff.groupby(PATH_KEY, sort=False):
        seq, last, wsum = [], None, 0.0
        vals = g[state_col].astype("string").to_numpy()
        convs = g["is_conversion"].to_numpy()

        base = pd.to_numeric(g[WEIGHT_COL], errors="coerce").to_numpy(float)
        stype = g["source_type"].astype("string").to_numpy()
        mult = np.where(np.isin(stype, list(RELIABLE_TYPES)), 1.0 + RELIABLE_WEIGHT_BOOST, 1.0)
        wts = np.nan_to_num(base, nan=0.0) * mult

        for v, cv, wt in zip(vals, convs, wts):
            if v != last:
                seq.append(v); last = v
            wsum += float(wt)
            if cv:
                seq.append(CONV)
                last = None
        if seq:
            paths.append(seq); weights.append(wsum)
    return paths, np.array(weights, dtype=float)

def build_paths_terms(df: pd.DataFrame, terms: Sequence[str]) -> Tuple[List[List[str]], np.ndarray]:
    term_re = [(t, re.compile(rf"\b{re.escape(t.lower())}\b")) for t in terms]
    cols = [PATH_KEY, TIME_COL, WEIGHT_COL, "source_type", "headline", "article_body", "is_conversion"]
    dff = df[cols].copy().sort_values([PATH_KEY, TIME_COL])

    def find_terms_row(head: str, body: str) -> List[str]:
        txt = (str(head) + " " + str(body)).lower()
        hits = []
        for t, rgx in term_re:
            if rgx.search(txt):
                h = f"TERM::{t}"
                if not hits or h != hits[-1]:
                    hits.append(h)
        return hits

    paths: List[List[str]] = []
    weights = []
    for _, g in dff.groupby(PATH_KEY, sort=False):
        seq, wsum = [], 0.0
        heads = g["headline"].astype(str).to_numpy()
        bodys = g["article_body"].astype(str).to_numpy()
        convs = g["is_conversion"].to_numpy()

        base = pd.to_numeric(g[WEIGHT_COL], errors="coerce").to_numpy(float)
        stype = g["source_type"].astype("string").to_numpy()
        mult = np.where(np.isin(stype, list(RELIABLE_TYPES)), 1.0 + RELIABLE_WEIGHT_BOOST, 1.0)
        wts = np.nan_to_num(base, nan=0.0) * mult

        for head, body, cv, wt in zip(heads, bodys, convs, wts):
            hits = find_terms_row(head, body)
            if hits:
                for h in hits:
                    if not seq or h != seq[-1]:
                        seq.append(h)
            wsum += float(wt)
            if cv:
                seq.append(CONV)
        if seq:
            paths.append(seq); weights.append(wsum)
    return paths, np.array(weights, dtype=float)

# =============================================================================
# PCA-ready tag_name table
# =============================================================================
def write_tagname_summary(df: pd.DataFrame, out_path: Path):
    cols_present = set(df.columns)
    g = df.groupby("tag_name", dropna=False)
    out = pd.DataFrame({
        "tag_name": g.size().index.astype("string"),
        "n_rows": g.size().to_numpy(),
        "n_conv": g["is_conversion"].sum().to_numpy()
    })
    out["conv_rate"] = out["n_conv"] / out["n_rows"]

    for col in ("vipr_weight", "circulation_size", "sentiment_score",
                "headline_token_count", "body_token_count", "token_count"):
        if col in cols_present:
            out[f"avg_{col}"] = g[col].apply(lambda s: _to_numeric(s).mean())

    if "sentiment_score" in cols_present:
        out["avg_abs_sentiment"] = g["sentiment_score"].apply(lambda s: _to_numeric(s).abs().mean())

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

# =============================================================================
# Main
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    print("[load] data (parquet)…")
    df = pd.read_parquet(DATA_PARQUET)
    print(f"[load] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[load] Columns: {list(df.columns)}")

    # Required columns
    required = {"headline","article_body","tag_name","vipr_weight","source_type","people_by_row"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing} in {DATA_PARQUET}")

    # Ensure emotion column persists and is string (do NOT drop later)
    if "emotion_body" in df.columns:
        df["emotion_body"] = df["emotion_body"].astype("string")

    # Numerics & downcast (preserve raw text/name/emotion columns)
    for c in ["vipr_weight","vipr_score","circulation_size","hit_strength","sentiment_score",
              "headline_token_count","body_token_count","token_count"]:
        if c in df.columns:
            df[c] = _to_numeric(df[c])
    _downcast_inplace(df)

    # Per-tag order index
    df[TIME_COL] = df.groupby(PATH_KEY).cumcount().astype(int)

    # Clean names + per-row surname→full upgrade + set conversions from cleaned list
    df = detect_persons_and_flag_conversion(df)

    # Numeric bins → create then normalize to 1..5 ints
    dims_num_present = [c for c in DIM_NUMERIC_ALL if c in df.columns]
    binned_cols = [add_quantile_bins(df, c) for c in dims_num_present]
    for bc in binned_cols:
        simplify_bin_to_int(df, bc)

    # Dimensions to score (exclude PATH_KEY itself)
    dimensions = [d for d in DIM_CATEGORICAL_ALL if d in df.columns and d != PATH_KEY] + binned_cols

    # ---- Collect all results then single write ----
    results: List[pd.DataFrame] = []

    for dim in dimensions:
        print(f"[attrib] {dim}")
        paths, w = build_paths(df, dim)
        res = markov_from_paths(paths, w, force_fast=True)
        if not res.empty:
            res = res.rename(columns={"state": "value"})
            res["kind"] = "item"
            res["dimension"] = dim
            res = res[["kind","dimension","value","credit","credit_share"]]
            res = add_ratings(res, "dimension")
            results.append(res)
        del paths, w, res
        gc.collect()

    # ---- Terms (optional; default OFF) ----
    if ENABLE_TERMS:
        terms: List[str] = []
        if KEYWORDS.exists():
            terms += pd.read_csv(KEYWORDS, header=None)[0].dropna().astype(str).str.strip().tolist()
        if BIGRAMS.exists():
            try:
                t = pd.read_csv(BIGRAMS, header=None)[0]
            except Exception:
                t = pd.read_csv(BIGRAMS).iloc[:, 0]
            terms += t.dropna().astype(str).str.strip().tolist()

        seen = set(); terms = [t for t in terms if not (t in seen or seen.add(t))]
        if MAX_KEY_TERMS and len(terms) > MAX_KEY_TERMS:
            terms = terms[:MAX_KEY_TERMS]

        if terms:
            total_terms = len(terms)
            n_chunks = int(math.ceil(total_terms / TERMS_CHUNK_SIZE))
            print(f"[terms] {total_terms} keywords/bigrams (chunk={TERMS_CHUNK_SIZE}, fast={FORCE_FAST_FOR_TERMS})")

            for i in range(0, total_terms, TERMS_CHUNK_SIZE):
                j = min(i + TERMS_CHUNK_SIZE, total_terms)
                chunk = terms[i:j]
                if PRINT_EVERY and ((i // TERMS_CHUNK_SIZE) % PRINT_EVERY == 0):
                    print(f"[terms] chunk {i//TERMS_CHUNK_SIZE + 1} / {n_chunks} -> {len(chunk)} terms")

                p, w = build_paths_terms(df, chunk)
                term_res = markov_from_paths(p, w, force_fast=FORCE_FAST_FOR_TERMS)
                if not term_res.empty:
                    term_res = term_res[term_res["state"].str.startswith("TERM::")].copy()
                    term_res["value"] = term_res["state"].str.replace("TERM::", "", regex=False)
                    term_res["kind"] = "term"
                    term_res["dimension"] = "term"
                    term_res = term_res[["kind","dimension","value","credit","credit_share"]]
                    term_res = add_ratings(term_res, "dimension")
                    results.append(term_res)

                del p, w, term_res
                gc.collect()

    # ---- Single write, no appends ----
    if results:
        final_df = pd.concat(results, ignore_index=True)
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        final_df.to_parquet(OUT_FILE, index=False)
    else:
        pd.DataFrame(columns=["kind","dimension","value","credit","credit_share","rating","rating_pct"]).to_parquet(OUT_FILE, index=False)

    # --- Save conversion-only FINAL dataset ---
    df_out = df.loc[df["is_conversion"] == True].copy()

    # Keep text-ish columns as strings
    for col in ["people_by_row","emotion_body"]:
        if col in df_out.columns:
            df_out[col] = df_out[col].astype("string")

    # Ensure *_bin columns are integers 1..5
    for c in df_out.columns:
        if c.endswith("_bin"):
            simplify_bin_to_int(df_out, c)

    # Drop requested columns from the absolute FINAL dataset
    FINAL_DROPS = ["article_body_raw","headline_raw","has_person","is_conversion"]
    df_out.drop(columns=[c for c in FINAL_DROPS if c in df_out.columns], inplace=True, errors="ignore")

    df_out.to_parquet(FULL_DATASET_FILE, index=False)
    print(f"[ok] saved conversion-only dataset -> {FULL_DATASET_FILE}")
    print(f"[ok] conversion-only shape: {df_out.shape}")
    print(f"[ok] dataset columns: {list(df_out.columns)}")

    # --- Tag-level PCA table (from ALL rows) ---
    write_tagname_summary(df, TAG_PCA_FILE)
    print(f"[ok] saved tag-level PCA table -> {TAG_PCA_FILE}")
    print(f"[ok] saved attribution analysis -> {OUT_FILE}")

if __name__ == "__main__":
    main()