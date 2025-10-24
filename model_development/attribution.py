#!/usr/bin/env python3
"""
Attribution Model – PERSON-based conversion final_deliverable/data/split/final_model_dataset_part_001.csv, reliability weighting, tag-level PCA table
FAST/LAPTOP MODE: uses state caps, pruning, and single final write

Inputs
------
  /Users/annaglass/capstone/capstone/data/processed/final_dataset_sampled.parquet

Outputs
-------
  data/processed/final_dataset_attribution.csv
  data/processed/persons_detected.csv
  data/processed/persons_by_row.csv
  data/processed/tagname_pca_ready.csv

Env toggles (optional)
----------------------
  SPACY_MODEL=en_core_web_trf | en_core_web_sm      # default: en_core_web_trf
  PERSON_NPROC=1                                      # spaCy processes
  SPACY_BATCH_SIZE=64                                 # transformer batch size
  SPACY_MIN_TEXT_LEN=60                               # skip very short rows
  SPACY_CHUNK_SIZE=10000                              # NER processing chunk size
  SKIP_NER=0                                          # 1 to skip NER (smoke test)

  MAX_STATES_PER_DIM=800                              # Markov state cap
  MIN_STATE_COUNT=8                                   # prune rare states
  FAST_MODE_THRESHOLD=600                             # always FAST beyond this
  ENABLE_TERMS=0                                      # keep off for speed
  MAX_KEY_TERMS=150
  TERMS_CHUNK_SIZE=150
"""

from __future__ import annotations
import os, re, gc, math
import time
from pathlib import Path
from typing import List, Sequence, Tuple
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from functools import lru_cache
import multiprocessing as mp

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve

# ---------------- spaCy (Transformer NER) ----------------
import spacy

# =============================================================================
# Performance Monitoring
# =============================================================================
def _log_performance(stage: str, start_time: float):
    """Log performance metrics for each stage."""
    elapsed = time.time() - start_time
    print(f"[perf] {stage}: {elapsed:.2f}s")
    return time.time()

def _load_spacy_model():
    """
    Prefer a transformer model (lowercase-tolerant).
    Fallback to small model if transformer isn't installed.
    """
    name = os.getenv("SPACY_MODEL", "en_core_web_trf")
    try:
        return spacy.load(name, disable=["parser","tagger","lemmatizer","attribute_ruler"])
    except Exception as e:
        if name != "en_core_web_sm":
            # fallback to small model
            try:
                print(f"[warn] Could not load '{name}'. Falling back to 'en_core_web_sm'. "
                      f"Install with: python -m spacy download {name}")
                return spacy.load("en_core_web_sm", disable=["parser","tagger","lemmatizer","attribute_ruler"])
            except Exception as e2:
                raise RuntimeError(
                    "No spaCy model available. Install one of:\n"
                    "  python -m spacy download en_core_web_trf\n"
                    "  python -m spacy download en_core_web_sm"
                ) from e2
        raise

NLP = _load_spacy_model()

# =============================================================================
# Paths & Core Columns
# =============================================================================
ROOT = Path("/Users/annaglass/capstone/capstone")
DATA_PARQUET = ROOT / "data_storage" / "processed_data" / "sampled_data.parquet" 

OUT_DIR  = ROOT / "data_storage" / "final_data"
OUT_FILE = OUT_DIR / "attribution_analysis_results.parquet"  # Attribution analysis only
COMPLETE_DATASET_FILE = OUT_DIR / "final_dataset.parquet"  # All data + attribution

KEYWORDS = OUT_DIR / "top_1000_keywords.csv"
BIGRAMS  = OUT_DIR / "top_1000_bigrams.csv"

PERSONS_FILE        = OUT_DIR / "persons_detected.csv"
PERSONS_BY_ROW_FILE = OUT_DIR / "persons_by_row.csv"

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
)
DIM_NUMERIC_ALL = ("circulation_size", "sentiment_score")  # binned

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

# Optimized spaCy settings for better performance
SPACY_BATCH_SIZE    = int(os.getenv("SPACY_BATCH_SIZE", "128"))  # Increased batch size
SPACY_MIN_TEXT_LEN  = int(os.getenv("SPACY_MIN_TEXT_LEN", "60"))
SPACY_NPROC         = max(1, min(mp.cpu_count(), int(os.getenv("PERSON_NPROC", "1"))))  # Auto-detect cores
SPACY_CHUNK_SIZE    = int(os.getenv("SPACY_CHUNK_SIZE", "5000"))  # Smaller chunks for better memory management
SKIP_NER            = os.getenv("SKIP_NER", "0") == "1"

# Additional performance settings
USE_PARALLEL_DIMENSIONS = os.getenv("USE_PARALLEL_DIMENSIONS", "1") == "1"  # Enable parallel dimension processing
MEMORY_EFFICIENT_MODE = os.getenv("MEMORY_EFFICIENT_MODE", "1") == "1"  # Use memory-efficient data types

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
    """Optimized downcasting with vectorized operations."""
    for c in df.columns:
        if pd.api.types.is_integer_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="integer")
        elif pd.api.types.is_float_dtype(df[c]):
            df[c] = pd.to_numeric(df[c], downcast="float")
        elif pd.api.types.is_object_dtype(df[c]):
            if c not in ("headline","article_body"):
                try:
                    # Use more efficient nunique calculation
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

@lru_cache(maxsize=128)
def _get_rating_labels():
    """Cached rating labels to avoid repeated list creation."""
    return [1, 2, 3, 4, 5]

def add_ratings(tbl: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Optimized rating calculation with vectorized operations."""
    if tbl.empty:
        return tbl.assign(rating=[], rating_pct=[])
    
    # Use vectorized operations where possible
    cs = tbl["credit_share"].astype("float32").fillna(0.0)  # Use float32 to save memory
    pct = cs.rank(method="average", pct=True).fillna(0.0)
    
    # Vectorized rating calculation
    rating = np.ceil(pct * 5.0).astype("int8").clip(1, 5)  # Use int8 to save memory
    
    # Add ratings to the dataframe
    tbl = tbl.assign(rating=rating)
    
    # Calculate rating percentages more efficiently
    rating_groups = tbl.groupby("rating")["credit_share"]
    min_vals = rating_groups.transform("min")
    max_vals = rating_groups.transform("max")
    denom = (max_vals - min_vals).replace(0, 1.0)
    rating_pct = ((tbl["credit_share"] - min_vals) / denom).clip(0, 1).astype("float32")
    
    return tbl.assign(rating_pct=rating_pct)

def _row_weight_with_reliability(r: pd.Series) -> float:
    base = float(_to_numeric(r.get(WEIGHT_COL, 1.0)) or 0.0)
    stype = str(r.get("source_type", "") or "")
    mult = 1.0 + RELIABLE_WEIGHT_BOOST if stype in RELIABLE_TYPES else 1.0
    return base * mult

# =============================================================================
# PERSON-based conversion (headline + article_body) with prefilter
# =============================================================================
TEXT_COLS = ["headline", "article_body"]
_CAP_HINT = re.compile(r"\b([A-Za-z][a-z]{2,})(?:\s+[A-Za-z][a-z]{2,})?\b")

def _looks_like_person_text(t: str) -> bool:
    if not t or len(t) < SPACY_MIN_TEXT_LEN:
        return False
    # Lowercase-friendly: allow name-like bigrams even if not capitalized
    # (Transformer models are robust, but prefilter still saves cycles)
    return bool(_CAP_HINT.search(t))

def _process_text_chunk(texts_and_indices: Tuple[List[str], List[int]]) -> Tuple[List[int], List[List[str]]]:
    """Process a chunk of texts and return indices and person lists."""
    texts, indices = texts_and_indices
    person_lists = []
    has_person_indices = []
    
    # Use a local NLP model to avoid multiprocessing issues
    try:
        local_nlp = spacy.load(os.getenv("SPACY_MODEL", "en_core_web_trf"), 
                              disable=["parser","tagger","lemmatizer","attribute_ruler"])
    except:
        local_nlp = spacy.load("en_core_web_sm", disable=["parser","tagger","lemmatizer","attribute_ruler"])
    
    docs = local_nlp.pipe(texts, batch_size=SPACY_BATCH_SIZE)
    
    for doc in docs:
        if doc and doc.ents:
            persons = [ent.text.strip() for ent in doc.ents if ent.label_ == "PERSON" and ent.text.strip()]
            person_lists.append(persons)
            if persons:
                has_person_indices.append(len(person_lists) - 1)
        else:
            person_lists.append([])
    
    return has_person_indices, person_lists

def detect_persons_and_flag_conversion(df: pd.DataFrame) -> pd.DataFrame:
    print(f"[conv] NER on: {TEXT_COLS} | model={NLP.meta.get('name','?')} | n_process={SPACY_NPROC} | batch={SPACY_BATCH_SIZE}")

    # Build combined text once - optimize with vectorized operations
    heads = df["headline"].astype("string").fillna("")
    bodys = df["article_body"].astype("string").fillna("")
    texts = (heads + " " + bodys).str.strip()
    
    # Pre-filter candidates more efficiently
    text_lengths = texts.str.len()
    candidate_mask = (text_lengths >= SPACY_MIN_TEXT_LEN) & texts.str.contains(_CAP_HINT, regex=True, na=False)
    cand_idx = np.where(candidate_mask)[0].tolist()
    
    print(f"[conv] rows={len(texts):,} | NER candidates={len(cand_idx):,} ({len(cand_idx)/max(1,len(texts)):.1%})")

    has_person = np.zeros(len(df), dtype=bool)
    person_lists: List[List[str]] = [[] for _ in range(len(df))]

    if SKIP_NER or len(cand_idx) == 0:
        df["has_person"] = has_person
        df["is_conversion"] = has_person
    else:
        # Optimize chunk processing with parallel execution
        chunk_size = SPACY_CHUNK_SIZE
        total_chunks = (len(cand_idx) + chunk_size - 1) // chunk_size
        
        print(f"[conv] Processing {len(cand_idx):,} candidates in {total_chunks} chunks of {chunk_size:,}")
        
        # Prepare chunks for parallel processing
        chunks = []
        for chunk_num in range(total_chunks):
            start_idx = chunk_num * chunk_size
            end_idx = min(start_idx + chunk_size, len(cand_idx))
            chunk_cand_idx = cand_idx[start_idx:end_idx]
            chunk_texts = [texts.iloc[i] for i in chunk_cand_idx]
            chunks.append((chunk_texts, chunk_cand_idx))
        
        # Process chunks in parallel if multiple processes are available
        if SPACY_NPROC > 1 and len(chunks) > 1:
            print(f"[conv] Using {SPACY_NPROC} processes for parallel NER processing")
            with ProcessPoolExecutor(max_workers=min(SPACY_NPROC, len(chunks))) as executor:
                results = list(executor.map(_process_text_chunk, chunks))
        else:
            # Sequential processing
            results = [_process_text_chunk(chunk) for chunk in chunks]
        
        # Combine results
        for chunk_num, (has_person_indices, chunk_person_lists) in enumerate(results):
            chunk_cand_idx = chunks[chunk_num][1]
            for i, (orig_idx, person_list) in enumerate(zip(chunk_cand_idx, chunk_person_lists)):
                person_lists[orig_idx] = person_list
                if person_list:
                    has_person[orig_idx] = True
            
            print(f"[conv] Completed chunk {chunk_num + 1}/{total_chunks}")
        
        df["has_person"] = has_person
        df["is_conversion"] = has_person

    # Optimize aggregation with vectorized operations
    flat_persons = [p for plist in person_lists for p in plist]
    if flat_persons:
        # Use pandas value_counts which is optimized
        agg = (pd.Series(flat_persons, dtype="string")
                 .value_counts()
                 .rename_axis("person")
                 .reset_index(name="count"))
    else:
        agg = pd.DataFrame(columns=["person","count"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(PERSONS_FILE, index=False)

    # Optimize persons_by_row creation
    persons_by_row = pd.DataFrame({
        "row_index": np.arange(len(df)),
        "tag_name": df["tag_name"].astype("string"),
        "persons": [", ".join(pl) if pl else "" for pl in person_lists],
        "has_person": has_person.astype("int8"),  # Use int8 to save memory
    })
    persons_by_row.to_csv(PERSONS_BY_ROW_FILE, index=False)

    print(f"[conv] PERSON rows = {int(has_person.sum())} / {len(df)}")
    print(f"[conv] Saved unique persons -> {PERSONS_FILE}")
    print(f"[conv] Saved per-row persons -> {PERSONS_BY_ROW_FILE}")
    return df

# =============================================================================
# Markov – ALWAYS FAST for medium/large; exact only when trivially small
# =============================================================================
def _row_normalized_transition(paths: List[List[str]], weights: np.ndarray, states: List[str]) -> csr_matrix:
    """Optimized transition matrix construction with pre-allocated arrays."""
    idx = {s: i for i, s in enumerate(states)}
    START = len(states)
    CONV_ID = len(states) + 1
    n_all = len(states) + 2

    # Pre-allocate arrays for better performance
    max_transitions = sum(len(seq) + 1 for seq in paths)  # Upper bound
    src = np.empty(max_transitions, dtype=np.int32)
    dst = np.empty(max_transitions, dtype=np.int32)
    wt = np.empty(max_transitions, dtype=np.float32)
    
    pos = 0
    for seq, w in zip(paths, weights):
        if not seq:
            continue
            
        # First transition from START
        first = seq[0]
        a_id = START
        b_id = idx.get(first, CONV_ID if first == CONV else None)
        if b_id is not None:
            src[pos] = a_id
            dst[pos] = b_id
            wt[pos] = w
            pos += 1
            
        # Subsequent transitions
        for a, b in zip(seq[:-1], seq[1:]):
            a_id = idx.get(a, CONV_ID if a == CONV else None)
            b_id = idx.get(b, CONV_ID if b == CONV else None)
            if a_id is None or b_id is None:
                continue
            src[pos] = a_id
            dst[pos] = b_id
            wt[pos] = w
            pos += 1
    
    # Trim arrays to actual size
    src = src[:pos]
    dst = dst[:pos]
    wt = wt[:pos]
    
    # Build sparse matrix more efficiently
    T = coo_matrix((wt, (src, dst)), shape=(n_all, n_all)).tocsr()
    
    # Row normalization with vectorized operations
    rs = np.asarray(T.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    return T.multiply(1.0 / rs[:, None]).tocsr()

def markov_from_paths(paths: List[List[str]], weights: np.ndarray,
                      max_states: int = MAX_STATES_PER_DIM,
                      min_state_count: int = MIN_STATE_COUNT,
                      force_fast: bool = True) -> pd.DataFrame:
    """Optimized Markov chain attribution calculation."""
    if not paths:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    # Optimized frequency counting with Counter
    from collections import Counter
    freq = Counter()
    for seq in paths:
        for s in seq:
            if s != CONV:
                freq[s] += 1

    # Filter states more efficiently
    states = [s for s, c in freq.items() if c >= min_state_count]
    rare = [s for s, c in freq.items() if c < min_state_count]

    # Optimize path processing with vectorized operations where possible
    if rare:
        rare_set = frozenset(rare)  # Use frozenset for faster lookup
        new_paths = []
        for seq in paths:
            # Use list comprehension for better performance
            mapped = [OTHER_LABEL if t in rare_set else t for t in seq]
            # Collapse consecutive identical states
            collapsed = [mapped[0]] if mapped else []
            for t in mapped[1:]:
                if t != collapsed[-1]:
                    collapsed.append(t)
            new_paths.append(collapsed)
        paths = new_paths
        
        # Recompute frequency efficiently
        freq = Counter()
        for seq in paths:
            for s in seq:
                if s != CONV:
                    freq[s] += 1
        states = [s for s in freq.keys() if s != CONV]

    # Handle state limit more efficiently
    if len(states) > max_states:
        # Sort by frequency and keep top states
        states_sorted = sorted(states, key=freq.get, reverse=True)
        keep = frozenset(states_sorted[:max_states-1]) | {OTHER_LABEL}
        
        new_paths = []
        for seq in paths:
            mapped = [t if (t in keep or t == CONV) else OTHER_LABEL for t in seq]
            collapsed = [mapped[0]] if mapped else []
            for t in mapped[1:]:
                if t != collapsed[-1]:
                    collapsed.append(t)
            new_paths.append(collapsed)
        paths = new_paths
        states = sorted(list(keep - {OTHER_LABEL})) + [OTHER_LABEL]

    # Build transition matrix
    states = sorted([s for s in set(states) if s != CONV])
    if not states:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    P = _row_normalized_transition(paths, weights, states)
    START = len(states)
    CONV_ID = len(states) + 1

    # Optimize matrix operations
    order = np.r_[[START], np.arange(len(states)), [CONV_ID]]
    P = P[order][:, order].tocsr()

    t_slice = slice(1, 1 + len(states))
    a_slice = slice(1 + len(states), None)
    Q = P[t_slice, t_slice].tocsr()
    R = P[t_slice, a_slice].tocsr()
    P0 = P[0, t_slice].tocsr()

    # Solve linear system more efficiently
    n = Q.shape[0]
    I_csr = identity(n, format="csr")
    A_csc = (I_csr - Q).tocsc()
    rhs_dense = np.asarray(P0.T.toarray()).ravel()
    y = spsolve(A_csc.T, rhs_dense)
    
    # Always use FAST mode for performance
    if force_fast or (len(states) >= FAST_MODE_THRESHOLD):
        r_col = np.asarray(R.toarray()).ravel() if R.shape[1] == 1 else np.asarray(R[:,0].toarray()).ravel()
        credits = np.maximum(0.0, y * r_col)
        total = credits.sum()
        share = credits / total if total > 0 else np.zeros_like(credits)
        
        # Use more efficient DataFrame creation
        result_df = pd.DataFrame({
            "state": states, 
            "credit": credits.astype("float32"), 
            "credit_share": share.astype("float32")
        })
        return result_df.sort_values("credit", ascending=False, ignore_index=True)

    # Exact calculation (rarely used)
    credits = np.zeros(len(states), dtype="float32")
    baseline_conv = float((csr_matrix(y.reshape(1, -1)) @ R).toarray()[0, 0])
    
    for i in range(Q.shape[0]):
        Q2 = Q.copy()
        R2 = R.copy()
        if Q2.indptr[i] != Q2.indptr[i+1]:
            Q2.data[Q2.indptr[i]:Q2.indptr[i+1]] = 0.0
        if R2.indptr[i] != R2.indptr[i+1]:
            R2.data[R2.indptr[i]:R2.indptr[i+1]] = 0.0
        A2 = (I_csr - Q2).tocsc()
        y2 = spsolve(A2.T, rhs_dense)
        conv2 = float((csr_matrix(y2.reshape(1, -1)) @ R2).toarray()[0, 0])
        credits[i] = max(0.0, baseline_conv - conv2)

    total = credits.sum()
    share = credits / total if total > 0 else np.zeros_like(credits)
    
    result_df = pd.DataFrame({
        "state": states, 
        "credit": credits, 
        "credit_share": share
    })
    return result_df.sort_values("credit", ascending=False, ignore_index=True)

# =============================================================================
# Path builders
# =============================================================================
def build_paths(df: pd.DataFrame, state_col: str) -> Tuple[List[List[str]], np.ndarray]:
    """Optimized path building with vectorized operations."""
    cols = [PATH_KEY, TIME_COL, WEIGHT_COL, "source_type", "is_conversion"]
    if state_col not in cols:
        cols.append(state_col)
    
    # Use more efficient data types
    dff = df[cols].dropna(subset=[PATH_KEY]).copy()
    dff = dff.sort_values([PATH_KEY, TIME_COL])

    paths: List[List[str]] = []
    weights = []
    
    # Pre-compute reliable types set for faster lookup
    reliable_types_set = set(RELIABLE_TYPES)
    
    for _, g in dff.groupby(PATH_KEY, sort=False):
        seq, last, wsum = [], None, 0.0
        
        # Convert to numpy arrays for faster processing
        vals = g[state_col].astype("string").to_numpy()
        convs = g["is_conversion"].to_numpy()
        
        # Vectorized weight calculation
        base = pd.to_numeric(g[WEIGHT_COL], errors="coerce").to_numpy("float32")
        stype = g["source_type"].astype("string").to_numpy()
        
        # Use set membership for faster lookup
        mult = np.where(np.isin(stype, reliable_types_set), 1.0 + RELIABLE_WEIGHT_BOOST, 1.0)
        wts = np.nan_to_num(base, nan=0.0) * mult

        # Process sequences more efficiently
        for v, cv, wt in zip(vals, convs, wts):
            if v != last:
                seq.append(v)
                last = v
            wsum += float(wt)
            if cv:
                seq.append(CONV)
                last = None
                
        if seq:
            paths.append(seq)
            weights.append(wsum)
            
    return paths, np.array(weights, dtype="float32")  # Use float32 to save memory

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
                if not hits or hits[-1] != h:
                    hits.append(h)
        return hits

    paths: List[List[str]] = []
    weights = []
    # weights vector as above
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
    """Optimized main function with better memory management and parallel processing."""
    start_time = time.time()
    print(f"[start] Attribution analysis started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[config] CPU cores: {mp.cpu_count()}, Parallel dimensions: {USE_PARALLEL_DIMENSIONS}, Memory efficient: {MEMORY_EFFICIENT_MODE}")
    
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        OUT_FILE.unlink()

    stage_start = time.time()
    print("[load] data (parquet)…")
    # Load data with optimized dtypes
    df = pd.read_parquet(DATA_PARQUET)
    print(f"[load] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[load] Columns: {list(df.columns)}")
    stage_start = _log_performance("Data loading", stage_start)

    required = {"headline","article_body","tag_name","vipr_weight","source_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Optimize numeric columns with better dtypes
    numeric_cols = ["vipr_weight","vipr_score","circulation_size","hit_strength","sentiment_score",
                    "headline_token_count","body_token_count","token_count"]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = _to_numeric(df[c]).astype("float32")  # Use float32 to save memory
    
    _downcast_inplace(df)

    # per-tag order index
    df[TIME_COL] = df.groupby(PATH_KEY).cumcount().astype("int16")  # Use int16 to save memory
    stage_start = _log_performance("Data preprocessing", stage_start)

    # PERSON conversion
    stage_start = time.time()
    df = detect_persons_and_flag_conversion(df)
    stage_start = _log_performance("PERSON detection", stage_start)

    # Numeric bins
    dims_num_present = [c for c in DIM_NUMERIC_ALL if c in df.columns]
    binned_cols = [add_quantile_bins(df, c) for c in dims_num_present]

    # Dimensions to score
    dimensions = [d for d in DIM_CATEGORICAL_ALL if d in df.columns and d != PATH_KEY] + binned_cols

    # ---- Collect all results with parallel processing ----
    results: List[pd.DataFrame] = []
    
    # Process dimensions in parallel if enabled and there are multiple CPU cores available
    max_workers = min(mp.cpu_count(), len(dimensions)) if USE_PARALLEL_DIMENSIONS else 1
    
    if max_workers > 1 and len(dimensions) > 1 and USE_PARALLEL_DIMENSIONS:
        print(f"[attrib] Processing {len(dimensions)} dimensions in parallel using {max_workers} workers")
        
        def process_dimension(dim):
            print(f"[attrib] {dim}")
            paths, w = build_paths(df, dim)
            res = markov_from_paths(paths, w, force_fast=True)
            if not res.empty:
                res = res.rename(columns={"state": "value"})
                res["kind"] = "item"
                res["dimension"] = dim
                res = res[["kind","dimension","value","credit","credit_share"]]
                res = add_ratings(res, "dimension")
                return res
            return None
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            dimension_results = list(executor.map(process_dimension, dimensions))
        
        # Filter out None results
        results = [res for res in dimension_results if res is not None]
    else:
        # Sequential processing
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

        # Remove duplicates efficiently
        seen = set()
        terms = [t for t in terms if not (t in seen or seen.add(t))]
        
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
        del final_df
    else:
        pd.DataFrame(columns=["kind","dimension","value","credit","credit_share","rating","rating_pct"]).to_parquet(OUT_FILE, index=False)

    # --- Save COMPLETE dataset with ALL columns and attribution analysis ---
    
    # Save ALL rows (not just conversions) with ALL original columns plus attribution columns
    df_out = df.copy()  # Keep ALL rows and ALL columns
    
    # Add attribution analysis results as additional columns
    if results:
        attribution_df = pd.concat(results, ignore_index=True)
        
        # Create a mapping of dimension -> top attribution results for each row
        for dim in dimensions:
            dim_results = attribution_df[attribution_df["dimension"] == dim].copy()
            if not dim_results.empty:
                # Get top 3 attributions for this dimension
                top_3 = dim_results.head(3)
                
                # Add columns for top 3 attributions for this dimension
                for i, (_, row) in enumerate(top_3.iterrows(), 1):
                    df_out[f"{dim}_attribution_{i}_value"] = row["value"]
                    df_out[f"{dim}_attribution_{i}_credit"] = row["credit"]
                    df_out[f"{dim}_attribution_{i}_credit_share"] = row["credit_share"]
                    df_out[f"{dim}_attribution_{i}_rating"] = row["rating"]
    
    df_out.to_parquet(COMPLETE_DATASET_FILE, index=False)
    print(f"[ok] saved COMPLETE dataset with ALL columns -> {COMPLETE_DATASET_FILE}")
    print(f"[ok] Complete dataset shape: {df_out.shape}")
    print(f"[ok] Complete dataset columns ({len(df_out.columns)}): {list(df_out.columns)}")
    
    # Also save conversion-only subset for backward compatibility
    CONVERSION_ONLY_FILE = OUT_DIR / "conversion_only_dataset.parquet"
    df_conversion = df.loc[df["is_conversion"] == True].copy()
    df_conversion.to_parquet(CONVERSION_ONLY_FILE, index=False)
    print(f"[ok] saved conversion-only subset -> {CONVERSION_ONLY_FILE}")
    print(f"[ok] Conversion-only shape: {df_conversion.shape}")

    # --- Tag-level PCA table ---
    TAG_PCA_FILE = OUT_DIR / "tagname_pca_ready.csv"
    write_tagname_summary(df, TAG_PCA_FILE)
    print(f"[ok] saved tag-level PCA table -> {TAG_PCA_FILE}")
    print(f"[ok] saved attribution analysis -> {OUT_FILE}")
    
    # Final performance summary
    total_time = time.time() - start_time
    print(f"[perf] Total execution time: {total_time:.2f}s ({total_time/60:.1f} minutes)")
    print(f"[perf] Average processing rate: {len(df)/total_time:.0f} rows/second")
    print(f"[done] Attribution analysis completed at {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()