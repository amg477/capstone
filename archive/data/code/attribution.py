#!/usr/bin/env python3
"""
Attribution Model – unified output, auto-tuned conversion, reliability weighting, and fast term processing
========================================================================================================

What this script does
---------------------
1) **Builds journeys (paths)** grouped by `tag_name`, ordered by `load_date`. Each path contains
   states drawn from a given dimension (e.g., publisher_name, author_name), plus an absorbing
   `<CONV>` state when a row qualifies as a conversion. It also builds **term** paths from
   `processed_headline` + `processed_body` (keywords/bigrams).

2) **Defines conversion** using an **auto-tuned k-of-signals rule** so you hit at least a target
   number of conversions with minimal overshoot. Signals per row:
      • reliable source type (`source_type_name` in a curated set)
      • high circulation (≥ q-quantile)
      • high vipr_weight (≥ q-quantile)
      • strong absolute sentiment (≥ threshold)
   Conversion fires if **≥ k** signals are present *and* the row is at/after a minimum step index
   within its tag’s timeline. The tuner scans strict→loose combos and picks the first that meets
   the target with minimal overshoot.

3) **Reliability weighting**: rows from reliable `source_type_name` get a +25% weight bump (tunable).
   This nudges attribution toward trusted sources without zeroing others.

4) **Markov removal-effect attribution**: credit(state) = drop in conversion probability when that
   state is removed. To stay fast/memory-safe:
   - Very rare states are merged into `__OTHER__`.
   - Max states per dimension is capped.
   - **Hybrid engine**:
       • EXACT (row-removal linear solves) for small/medium cardinalities.
       • FAST approximation (visit_prob × conv_prob_from_state) when state count is large or forced.

5) **Unified output**: Writes one CSV `data/processed/attribution_all_scored.csv` combining
   all item dimensions and term results. Adds a **1–5 rating** per dimension, robust to ties, plus
   `rating_pct` (0..1 within-band).

How to run
----------
    python attribution.py

Output
------
  data/processed/attribution_all_scored.csv
Columns:
  kind ∈ {"item","term"}, dimension, value, credit, credit_share, rating, rating_pct

Main knobs (speed / reliability / volume)
-----------------------------------------
- Speed/memory:
    MAX_STATES_PER_DIM, MIN_STATE_COUNT, FAST_MODE_THRESHOLD
- Terms speed:
    MAX_KEY_TERMS, TERMS_CHUNK_SIZE, FORCE_FAST_FOR_TERMS, PRINT_EVERY
- Reliability bias:
    RELIABLE_TYPES, RELIABLE_WEIGHT_BOOST
- Conversion target:
    TARGET_CONV_COUNT  (auto-tuner scans strict→loose grid)
"""

from __future__ import annotations
import re, gc
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve

# =============================================================================
# Paths & Core Columns
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]    # .../capstone/capstone
DATA_CSV   = ROOT / "data" / "final_dataset_sampled.csv"
OUT_DIR    = ROOT / "data" / "processed"
OUT_FILE   = OUT_DIR / "attribution_all_scored.csv"
KEYWORDS   = OUT_DIR / "top_1000_keywords.csv"
BIGRAMS    = OUT_DIR / "top_1000_bigrams.csv"

PATH_KEY   = "tag_name"
TIME_COL   = "load_date"
WEIGHT_COL = "vipr_weight"
CONV       = "<CONV>"
OTHER_LABEL = "__OTHER__"

# =============================================================================
# Dimensions (categorical + numeric bins)
# =============================================================================
DIM_CATEGORICAL = (
    "tag_name",              # skipped (path key)
    "source_feed_name",
    "feed_name",
    "author_name",
    "source_type_name",
    "channel_name",
    "genre",
    "publisher_name",
    "publication_name",
    "source_type",
    "sentiment_band",
)
DIM_NUMERIC = ("circulation_size", "sentiment_score")  # will be binned

# =============================================================================
# Speed / Memory knobs
# =============================================================================
MAX_STATES_PER_DIM = 3000   # cap unique states per dimension (others -> __OTHER__)
MIN_STATE_COUNT     = 3     # states with freq < this are merged into __OTHER__ first
FAST_MODE_THRESHOLD = 2500  # use FAST approximation when #states >= this

# ===== Terms speed knobs =====
MAX_KEY_TERMS        = 400     # lower values speed up runs (e.g., 300/250)
TERMS_CHUNK_SIZE     = 200     # process keywords in batches
FORCE_FAST_FOR_TERMS = True    # force FAST approximation for terms
PRINT_EVERY          = 1       # print each chunk progress (raise to reduce logging)

# =============================================================================
# Reliability boost (affects all dimensions via path weights)
# =============================================================================
RELIABLE_TYPES = {
    "National News","Government","Wires",
    "General News","Regional News","Trade News",
}
RELIABLE_WEIGHT_BOOST = 0.25   # +25%

# =============================================================================
# Conversion target & path behavior
# =============================================================================
TARGET_CONV_COUNT   = 20000
STOP_AT_FIRST_CONV  = False  # keep collecting states after conversion (better coverage)

# =============================================================================
# Utilities
# =============================================================================
def add_quantile_bins(df: pd.DataFrame, col: str, bins: int = 5) -> str:
    series = pd.to_numeric(df[col], errors="coerce")
    ranks = series.rank(method="first")
    labels = [f"{col.upper()}_Q{i}" for i in range(1, bins+1)]
    new_col = f"{col}_bin"
    df[new_col] = pd.qcut(ranks, bins, labels=labels).astype("string")
    return new_col

def add_ratings(tbl: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """
    Add 1–5 ratings per dimension, robust to heavy ties (duplicate bin edges).
    - Primary: quantile bins on percentile ranks (5 bins) with duplicates='drop'.
    - Fallback: percentile→ceil(5*pct).
    - rating_pct: normalized 0..1 within each rating band.
    """
    if tbl.empty:
        return tbl.assign(rating=[], rating_pct=[])

    out = []
    for _, g in tbl.groupby(group_col, group_keys=False):
        cs = g["credit_share"].astype(float).fillna(0.0)

        if (len(g) < 5) or (cs.nunique() < 5):
            pct = cs.rank(method="average", pct=True).fillna(0.0)
            rating = np.ceil(pct * 5.0).astype(int).clip(1, 5)
            g = g.assign(rating=rating)
        else:
            pct = cs.rank(method="average", pct=True).fillna(0.0)
            try:
                qbins = pd.qcut(pct, 5, labels=[1,2,3,4,5], duplicates="drop")
                if qbins.dtype == "category" and qbins.cat.categories.size < 5:
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
    base = float(pd.to_numeric(r.get(WEIGHT_COL, 1.0), errors="coerce") or 0.0)
    stype = str(r.get("source_type_name", "") or "")
    mult = 1.0 + RELIABLE_WEIGHT_BOOST if stype in RELIABLE_TYPES else 1.0
    return base * mult

# =============================================================================
# Conversion (auto-tuned, minimal overshoot) – k-of-4 signals
# =============================================================================
def _compute_conv_flag_kofn(
    df: pd.DataFrame,
    q: float,
    sent_thr: int,
    min_steps: int,
    k_required: int
) -> pd.Series:
    circ   = pd.to_numeric(df.get("circulation_size", 0), errors="coerce").fillna(0)
    vipr   = pd.to_numeric(df.get("vipr_weight", 0), errors="coerce").fillna(0)
    sscore = pd.to_numeric(df.get("sentiment_score", 0), errors="coerce").fillna(0)
    stype  = df.get("source_type_name", pd.Series(index=df.index, dtype=object)).astype("string")

    q_circ = circ.quantile(q) if len(circ) else np.inf
    q_vipr = vipr.quantile(q) if len(vipr) else np.inf

    reliable    = stype.isin(RELIABLE_TYPES)
    high_circ   = circ >= q_circ
    high_vipr   = vipr >= q_vipr
    strong_sent = sscore.abs() >= sent_thr

    signals = np.vstack([
        reliable.values.astype(int),
        high_circ.values.astype(int),
        high_vipr.values.astype(int),
        strong_sent.values.astype(int),
    ])
    kofn = signals.sum(axis=0) >= int(k_required)
    conv_raw = pd.Series(kofn, index=df.index, dtype=bool)

    if (TIME_COL in df.columns) and (PATH_KEY in df.columns):
        tmp = df[[PATH_KEY, TIME_COL]].copy()
        tmp[TIME_COL] = pd.to_datetime(tmp[TIME_COL], errors="coerce")
        steps = tmp.groupby(PATH_KEY)[TIME_COL].rank(method="first")
        conv = conv_raw & (steps >= min_steps)
    else:
        conv = conv_raw

    return conv.fillna(False)

def autotune_conversion(df: pd.DataFrame) -> pd.Series:
    grid = [
        (3, 0.80, 18, 3),
        (3, 0.75, 15, 3),
        (2, 0.80, 18, 3),
        (2, 0.75, 15, 3),
        (2, 0.70, 12, 2),
        (2, 0.65, 12, 1),
        (2, 0.60, 10, 1),
        (2, 0.55, 10, 1),
        (2, 0.50,  8, 0),
        (1, 0.75, 15, 3),
        (1, 0.65, 12, 1),
        (1, 0.50,  8, 0),
    ]
    best_conv = None
    best_count = -1
    chosen = None
    overshoot = []
    for k_required, q, thr, m in grid:
        conv = _compute_conv_flag_kofn(df, q=q, sent_thr=thr, min_steps=m, k_required=k_required)
        n = int(conv.sum())
        print(f"[conv] k={k_required} q={q:.2f} sent_thr={thr} min_steps={m} -> {n} rows")
        best_conv, best_count, chosen = conv, n, (k_required, q, thr, m)
        if n >= TARGET_CONV_COUNT:
            overshoot.append((n, (k_required, q, thr, m), conv))
    if overshoot:
        overshoot.sort(key=lambda x: x[0])  # minimal overshoot
        n, params, conv = overshoot[0]
        k_required, q, thr, m = params
        print(f"[conv] selected (minimal overshoot): k={k_required}, q={q:.2f}, sent_thr={thr}, min_steps={m} -> {n} rows (>= {TARGET_CONV_COUNT})")
        return conv
    k_required, q, thr, m = chosen
    print(f"[conv] fallback (max achieved {best_count} < {TARGET_CONV_COUNT}) with k={k_required}, q={q:.2f}, sent_thr={thr}, min_steps={m}")
    return best_conv

# =============================================================================
# Markov – hybrid (FAST for big, EXACT for small) and memory-safe
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
                      force_fast: bool = False) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    # ---- frequency prune/merge to OTHER ----
    freq = {}
    for seq in paths:
        for s in seq:
            if s == CONV: continue
            freq[s] = freq.get(s, 0) + 1

    states = [s for s,c in freq.items() if c >= min_state_count]
    rare = [s for s,c in freq.items() if c <  min_state_count]
    if rare:
        rare_set = set(rare)
        new_paths = []
        for seq in paths:
            mapped = [ (OTHER_LABEL if (t in rare_set) else t) for t in seq ]
            collapsed = []
            last = None
            for t in mapped:
                if t != last:
                    collapsed.append(t); last = t
            new_paths.append(collapsed)
        paths = new_paths
        # recompute freq
        freq = {}
        for seq in paths:
            for s in seq:
                if s == CONV: continue
                freq[s] = freq.get(s, 0) + 1
        states = list(freq.keys())

    if len(states) > max_states:
        states_sorted = sorted(states, key=lambda s: freq[s], reverse=True)
        keep = set(states_sorted[:max_states-1])  # hold 1 slot for OTHER
        if OTHER_LABEL not in keep:
            keep.add(OTHER_LABEL)
        new_paths = []
        for seq in paths:
            mapped = [ (t if (t in keep or t == CONV) else OTHER_LABEL) for t in seq ]
            collapsed, last = [], None
            for t in mapped:
                if t != last:
                    collapsed.append(t); last = t
            new_paths.append(collapsed)
        paths = new_paths
        states = sorted(list(keep - {OTHER_LABEL})) + ([OTHER_LABEL] if OTHER_LABEL in keep else [])

    # Build P, partition to Q,R,P0
    states = sorted([s for s in set(states) if s != CONV])
    P = _row_normalized_transition(paths, weights, states)
    START = len(states); CONV_ID = len(states) + 1

    order = np.r_[ [START], np.arange(len(states)), [CONV_ID] ]
    P = P[order][:, order].tocsr()

    t_slice = slice(1, 1 + len(states))
    a_slice = slice(1 + len(states), None)
    Q = P[t_slice, t_slice].tocsr()
    R = P[t_slice, a_slice].tocsr()   # (T, 1)
    P0 = P[0, t_slice].tocsr()        # (1, T)

    # ---- One RHS solve for baseline and visit probabilities ----
    n = Q.shape[0]
    I_csr = identity(n, format="csr")
    A_csc = (I_csr - Q).tocsc()
    rhs = P0.T.tocsc()
    rhs_dense = np.asarray(rhs.toarray()).ravel()
    y = spsolve(A_csc.T, rhs_dense)         # shape (T,)
    y_row = csr_matrix(y.reshape(1, -1))    # (1, T)

    # Baseline conversion probability
    baseline = float((y_row @ R).toarray()[0, 0])

    # ---- FAST path for large state spaces OR forced fast ----
    if force_fast or (len(states) >= FAST_MODE_THRESHOLD):
        # Approx: credit_i ≈ visit_prob(i) * P(convert | i)
        r_col = np.asarray(R.toarray()).ravel() if R.shape[1] == 1 else np.asarray(R[:,0].toarray()).ravel()
        credits = np.maximum(0.0, y * r_col)
        total = credits.sum()
        share = credits / total if total > 0 else np.zeros_like(credits)
        return (
            pd.DataFrame({"state": states, "credit": credits, "credit_share": share})
              .sort_values("credit", ascending=False, ignore_index=True)
        )

    # ---- EXACT path for small/medium dims (row removal with single-RHS solves) ----
    credits = np.zeros(len(states), dtype=float)
    for i in range(Q.shape[0]):
        Q2 = Q.copy()
        R2 = R.copy()
        if Q2.indptr[i] != Q2.indptr[i+1]:
            Q2.data[Q2.indptr[i]:Q2.indptr[i+1]] = 0.0
        if R2.indptr[i] != R2.indptr[i+1]:
            R2.data[R2.indptr[i]:R2.indptr[i+1]] = 0.0
        A2 = (I_csr - Q2).tocsc()
        rhs2 = P0.T.tocsc()
        rhs2_dense = np.asarray(rhs2.toarray()).ravel()
        y2 = spsolve(A2.T, rhs2_dense)
        y2_row = csr_matrix(y2.reshape(1, -1))
        conv2 = float((y2_row @ R2).toarray()[0, 0])
        credits[i] = max(0.0, baseline - conv2)

    total = credits.sum()
    share = credits / total if total > 0 else np.zeros_like(credits)
    return (
        pd.DataFrame({"state": states, "credit": credits, "credit_share": share})
          .sort_values("credit", ascending=False, ignore_index=True)
    )

# =============================================================================
# Path builders (use reliability-aware weights)
# =============================================================================
def build_paths(df: pd.DataFrame, state_col: str) -> Tuple[List[List[str]], np.ndarray]:
    cols = [PATH_KEY, TIME_COL, WEIGHT_COL, "source_type_name", "is_conversion"]
    if state_col not in cols:
        cols.append(state_col)
    dff = df[cols].dropna(subset=[PATH_KEY, TIME_COL]).copy()
    dff[TIME_COL] = pd.to_datetime(dff[TIME_COL], errors="coerce")
    dff = dff.dropna(subset=[TIME_COL]).sort_values([PATH_KEY, TIME_COL])

    paths, weights = [], []
    for _, g in dff.groupby(PATH_KEY, sort=False):
        seq, last, wsum = [], None, 0.0
        for _, r in g.iterrows():
            val = str(r[state_col]) if state_col in r else str(r[PATH_KEY])
            if val != last:
                seq.append(val); last = val
            wsum += _row_weight_with_reliability(r)
            if r["is_conversion"]:
                seq.append(CONV)
                if STOP_AT_FIRST_CONV:
                    break
                else:
                    last = None
        if seq:
            paths.append(seq); weights.append(wsum)
    return paths, np.array(weights, dtype=float)

def build_paths_terms(df: pd.DataFrame, terms: Sequence[str]) -> Tuple[List[List[str]], np.ndarray]:
    term_re = [(t, re.compile(rf"\b{re.escape(t.lower())}\b")) for t in terms]
    cols = [PATH_KEY, TIME_COL, WEIGHT_COL, "source_type_name", "processed_headline", "processed_body", "is_conversion"]
    dff = df[cols].copy()
    dff[TIME_COL] = pd.to_datetime(dff[TIME_COL], errors="coerce")
    dff = dff.dropna(subset=[TIME_COL]).sort_values([PATH_KEY, TIME_COL])

    def find_terms(r) -> List[str]:
        h = r.get("processed_headline", ""); b = r.get("processed_body", "")
        h = "" if (h is None or pd.isna(h)) else str(h)
        b = "" if (b is None or pd.isna(b)) else str(b)
        txt = (h + " " + b).lower()
        return [f"TERM::{t}" for t, rgx in term_re if rgx.search(txt)]

    paths, weights = [], []
    for _, g in dff.groupby(PATH_KEY, sort=False):
        seq, wsum = [], 0.0
        for _, r in g.iterrows():
            hits = find_terms(r)
            if hits:
                for h in hits:
                    if not seq or h != seq[-1]:
                        seq.append(h)
            wsum += _row_weight_with_reliability(r)
            if r["is_conversion"]:
                seq.append(CONV)
                if STOP_AT_FIRST_CONV:
                    break
        if seq:
            paths.append(seq); weights.append(wsum)
    return paths, np.array(weights, dtype=float)

# =============================================================================
# Main (incremental writing + chunked terms)
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if OUT_FILE.exists():
        OUT_FILE.unlink()  # fresh run

    print("[load] data…")
    df = pd.read_csv(DATA_CSV, low_memory=False)

    # Deduplicate duplicate header names
    if df.columns.duplicated().any():
        counts, new_cols = {}, []
        for c in df.columns:
            if c not in counts: counts[c] = 0; new_cols.append(c)
            else: counts[c] += 1; new_cols.append(f"{c}_{counts[c]}")
        df.columns = new_cols

    # Core dtypes and safe text
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    for c in set(DIM_CATEGORICAL) | {PATH_KEY}:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for col in ["processed_headline", "processed_body"]:
        if col in df.columns: df[col] = df[col].fillna("").astype(str)
        else: df[col] = ""

    # Conversion (auto-tuned)
    df["is_conversion"] = autotune_conversion(df)
    print(f"[conv] final count = {int(df['is_conversion'].sum())}")

    # Numeric bins
    binned_cols = [add_quantile_bins(df, c) for c in DIM_NUMERIC if c in df.columns]
    dimensions = [d for d in DIM_CATEGORICAL if d in df.columns and d != PATH_KEY] + binned_cols

    # Prepare unified CSV
    header_written = False

    # ---- Items (incremental) ----
    for dim in dimensions:
        print(f"[attrib] {dim}")
        paths, w = build_paths(df, dim)
        res = markov_from_paths(paths, w)  # exact for small/medium; fast for very large
        res = res.rename(columns={"state": "value"})
        res["kind"] = "item"
        res["dimension"] = dim
        res = res[["kind","dimension","value","credit","credit_share"]]
        res = add_ratings(res, "dimension")

        # write/append
        mode = "w" if not header_written else "a"
        res.to_csv(OUT_FILE, mode=mode, index=False, header=not header_written)
        header_written = True

        # free memory between dimensions
        del paths, w, res
        gc.collect()

    # ---- Terms (incremental, chunked) ----
    terms: List[str] = []
    if KEYWORDS.exists():
        terms += pd.read_csv(KEYWORDS, header=None)[0].dropna().astype(str).str.strip().tolist()
    if BIGRAMS.exists():
        try:
            t = pd.read_csv(BIGRAMS, header=None)[0]
        except Exception:
            t = pd.read_csv(BIGRAMS).iloc[:, 0]
        terms += t.dropna().astype(str).str.strip().tolist()

    # unique, preserve order
    seen = set(); terms = [t for t in terms if not (t in seen or seen.add(t))]

    # Optional cap for speed this run
    if MAX_KEY_TERMS and len(terms) > MAX_KEY_TERMS:
        terms = terms[:MAX_KEY_TERMS]

    if terms:
        total_terms = len(terms)
        n_chunks = int(np.ceil(total_terms / TERMS_CHUNK_SIZE))
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

                term_res.to_csv(OUT_FILE, mode=("a" if header_written else "w"),
                                index=False, header=not header_written)
                header_written = True

            # free memory per chunk
            del p, w, term_res
            gc.collect()

    print(f"[ok] saved -> {OUT_FILE}")

if __name__ == "__main__":
    main()