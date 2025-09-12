#!/usr/bin/env python3
"""
Attribution Model – Unified Output with Reliability Boost & Auto-Tuned Conversion
-------------------------------------------------------------------------------
This script computes Markov removal-effect attribution for:
  • Items across multiple dimensions (publishers, authors, binned numerics, etc.)
  • Terms (keywords/bigrams) from processed_headline + processed_body

Key features:
- Conversion auto-tuned to reach at least TARGET_CONV_COUNT rows.
- Paths DO NOT stop at first conversion (collects more states).
- Reliable news types get a small weight boost (+25%) but others still count.
- Writes a single CSV: data/processed/attribution_all_scored.csv

Run:
  python attribution.py
"""

from __future__ import annotations
import re
from pathlib import Path
from typing import List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csc_matrix, csr_matrix, identity
from scipy.sparse.linalg import spsolve

# =============================================================================
# Configuration & Paths
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]    # .../capstone/capstone
DATA_CSV   = ROOT / "data" / "final_dataset_sampled.csv"
OUT_DIR    = ROOT / "data" / "processed"
KEYWORDS   = OUT_DIR / "top_1000_keywords.csv"
BIGRAMS    = OUT_DIR / "top_1000_bigrams.csv"

PATH_KEY   = "tag_name"        # journey key
TIME_COL   = "load_date"
WEIGHT_COL = "vipr_weight"
CONV       = "<CONV>"

# Categorical dimensions (skip PATH_KEY to avoid self-attribution)
DIM_CATEGORICAL = (
    "tag_name",              # will be excluded below
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
# Numeric to bin (quantiles)
DIM_NUMERIC = ("circulation_size", "sentiment_score")

# Reliable types (get a small weight boost in attribution)
RELIABLE_TYPES = {
    "National News","Government","Wires",
    "General News","Regional News","Trade News",
}
RELIABLE_WEIGHT_BOOST = 0.25   # +25% weight for reliable news types

# Conversion target & path behavior
TARGET_CONV_COUNT = 20000      # aim for >= 20k conversion rows
STOP_AT_FIRST_CONV = False     # keep collecting states after conversion

# =============================================================================
# Helpers
# =============================================================================
def add_quantile_bins(df: pd.DataFrame, col: str, bins: int = 5) -> str:
    series = pd.to_numeric(df[col], errors="coerce")
    ranks = series.rank(method="first")
    labels = [f"{col.upper()}_Q{i}" for i in range(1, bins+1)]
    new_col = f"{col}_bin"
    df[new_col] = pd.qcut(ranks, bins, labels=labels).astype("string")
    return new_col

def add_ratings(tbl: pd.DataFrame, group_col: str) -> pd.DataFrame:
    if tbl.empty:
        return tbl.assign(rating=[], rating_pct=[])
    out = []
    for _, g in tbl.groupby(group_col, group_keys=False):
        cs = g["credit_share"].fillna(0.0)
        if cs.nunique() <= 1:
            g = g.assign(rating=3, rating_pct=0.5)
        else:
            qbins = pd.qcut(cs.rank(method="average"), 5, labels=[1,2,3,4,5])
            g = g.assign(rating=qbins.astype(int))
            g["_min"] = g.groupby("rating")["credit_share"].transform("min")
            g["_max"] = g.groupby("rating")["credit_share"].transform("max")
            denom = (g["_max"] - g["_min"]).replace(0, 1.0)
            g["rating_pct"] = ((g["credit_share"] - g["_min"]) / denom).clip(0,1)
            g = g.drop(columns=["_min","_max"])
        out.append(g)
    return pd.concat(out, ignore_index=True)

# ---- Reliability-aware row weight ------------------------------------------------
def _row_weight_with_reliability(r: pd.Series) -> float:
    base = float(pd.to_numeric(r.get(WEIGHT_COL, 1.0), errors="coerce") or 0.0)
    stype = str(r.get("source_type_name", "") or "")
    mult = 1.0 + RELIABLE_WEIGHT_BOOST if stype in RELIABLE_TYPES else 1.0
    return base * mult

# =============================================================================
# Conversion logic (auto-tuned to hit >= TARGET_CONV_COUNT)
# =============================================================================
def _compute_conv_flag(df: pd.DataFrame, q: float, sent_thr: int, min_steps: int) -> pd.Series:
    circ = pd.to_numeric(df.get("circulation_size", 0), errors="coerce").fillna(0)
    vipr = pd.to_numeric(df.get("vipr_weight", 0), errors="coerce").fillna(0)
    sscore = pd.to_numeric(df.get("sentiment_score", 0), errors="coerce").fillna(0)
    stype = df.get("source_type_name", pd.Series(index=df.index, dtype=object)).astype("string")

    q_circ = circ.quantile(q) if len(circ) else np.inf
    q_vipr = vipr.quantile(q) if len(vipr) else np.inf

    reliable    = stype.isin(RELIABLE_TYPES)
    high_circ   = circ >= q_circ
    high_vipr   = vipr >= q_vipr
    strong_sent = sscore.abs() >= sent_thr

    # Looser logic: convert if ANY major signal OR reliable
    conv_raw = reliable | high_circ | high_vipr | strong_sent

    # Minimum steps within each PATH_KEY before conversion can trigger
    if (TIME_COL in df.columns) and (PATH_KEY in df.columns):
        tmp = df[[PATH_KEY, TIME_COL]].copy()
        tmp[TIME_COL] = pd.to_datetime(tmp[TIME_COL], errors="coerce")
        order = tmp.groupby(PATH_KEY)[TIME_COL].rank(method="first")
        conv = conv_raw & (order >= min_steps)
    else:
        conv = conv_raw

    return conv.fillna(False)

def autotune_conversion(df: pd.DataFrame) -> pd.Series:
    """
    Try progressively looser settings until we meet TARGET_CONV_COUNT.
    Returns the chosen Boolean conversion series.
    """
    grid = [
        (0.75, 15, 3),
        (0.70, 12, 2),
        (0.65, 12, 1),
        (0.60, 10, 1),
        (0.55, 10, 1),
        (0.50,  8, 0),
        (0.40,  8, 0),
        (0.30,  6, 0),
    ]
    best = None
    for q, thr, m in grid:
        conv = _compute_conv_flag(df, q=q, sent_thr=thr, min_steps=m)
        n = int(conv.sum())
        print(f"[conv] q={q:.2f} sent_thr={thr} min_steps={m} -> {n} rows")
        best = conv
        if n >= TARGET_CONV_COUNT:
            print(f"[conv] selected: q={q:.2f}, sent_thr={thr}, min_steps={m} (>= {TARGET_CONV_COUNT})")
            return conv
    print(f"[conv] fallback (max achieved {int(best.sum())} < {TARGET_CONV_COUNT})")
    return best

# =============================================================================
# Markov engine (CSR for row ops, CSC for solves) – removal effect
# =============================================================================
def markov_from_paths(paths: List[List[str]], weights: np.ndarray) -> pd.DataFrame:
    if not paths:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    # Build state index
    states = sorted({s for seq in paths for s in seq if s != CONV})
    idx = {s: i for i, s in enumerate(states)}
    START = len(states)
    CONV_ID = len(states) + 1
    n_all = len(states) + 2

    # Transition counts (weighted)
    src, dst, wt = [], [], []
    for seq, w in zip(paths, weights):
        if not seq:
            continue
        # START -> first
        first = seq[0]
        src.append(START); dst.append(idx.get(first, CONV_ID if first == CONV else None)); wt.append(w)
        # internal transitions
        for a, b in zip(seq[:-1], seq[1:]):
            a_id = idx.get(a, CONV_ID if a == CONV else None)
            b_id = idx.get(b, CONV_ID if b == CONV else None)
            if a_id is None or b_id is None:
                continue
            src.append(a_id); dst.append(b_id); wt.append(w)

    # Row-normalized transition P (CSR)
    T = coo_matrix((wt, (src, dst)), shape=(n_all, n_all)).tocsr()
    rs = np.asarray(T.sum(axis=1)).ravel()
    rs[rs == 0] = 1.0
    P = T.multiply(1.0 / rs[:, None]).tocsr()

    # Reorder: START | transient | absorbing(CONV)
    transient = np.array([i for i in range(len(states))], dtype=int)
    order = np.r_[ [START], transient, [CONV_ID] ]
    P = P[order][:, order].tocsr()

    t_slice = slice(1, 1 + len(transient))
    a_slice = slice(1 + len(transient), None)

    # Use CSR for row zeroing, CSC for linear solves
    Q_csr = P[t_slice, t_slice].tocsr()
    R_csr = P[t_slice, a_slice].tocsr()
    P0_csr = P[0, t_slice].tocsr()

    I_csr = identity(Q_csr.shape[0], format="csr")
    A_csc  = (I_csr - Q_csr).tocsc()
    I_csc  = identity(Q_csr.shape[0], format="csc")
    N = spsolve(A_csc, I_csc)  # baseline fundamental matrix (CSC solve)

    baseline = float((P0_csr.tocsc() @ N @ R_csr.tocsc()).A.ravel()[0])

    # Removal effect: zero **row i** in Q and R (CSR row ops), then solve in CSC
    credits = []
    for i in range(Q_csr.shape[0]):
        Q2 = Q_csr.copy()
        R2 = R_csr.copy()
        # zero row i in Q2
        if Q2.indptr[i] != Q2.indptr[i+1]:
            Q2.data[Q2.indptr[i]:Q2.indptr[i+1]] = 0.0
        # zero row i in R2
        if R2.indptr[i] != R2.indptr[i+1]:
            R2.data[R2.indptr[i]:R2.indptr[i+1]] = 0.0

        A2_csc = (I_csr - Q2).tocsc()
        N2 = spsolve(A2_csc, I_csc)
        conv2 = float((P0_csr.tocsc() @ N2 @ R2.tocsc()).A.ravel()[0])
        credits.append(max(0.0, baseline - conv2))

    credits = np.array(credits, dtype=float)
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
    """Build paths of `state_col` per PATH_KEY, optionally continue after conversion."""
    cols = [PATH_KEY, TIME_COL, WEIGHT_COL, "is_conversion", "source_type_name"]
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
                    last = None  # allow immediate repeats after CONV
        if seq:
            paths.append(seq); weights.append(wsum)
    return paths, np.array(weights, dtype=float)

def build_paths_terms(df: pd.DataFrame, terms: Sequence[str]) -> Tuple[List[List[str]], np.ndarray]:
    """Build paths of TERM::<keyword> hits per PATH_KEY, optionally continue after conversion."""
    term_re = [(t, re.compile(rf"\b{re.escape(t.lower())}\b")) for t in terms]

    dff = df[[PATH_KEY, TIME_COL, WEIGHT_COL, "source_type_name",
              "processed_headline", "processed_body", "is_conversion"]].copy()
    dff[TIME_COL] = pd.to_datetime(dff[TIME_COL], errors="coerce")
    dff = dff.dropna(subset=[TIME_COL]).sort_values([PATH_KEY, TIME_COL])

    def find_terms(r) -> List[str]:
        h = r.get("processed_headline", "")
        b = r.get("processed_body", "")
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
# Main
# =============================================================================
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print("[load] data…")
    df = pd.read_csv(DATA_CSV, low_memory=False)

    # Deduplicate duplicate header names if any
    if df.columns.duplicated().any():
        counts, new_cols = {}, []
        for c in df.columns:
            if c not in counts:
                counts[c] = 0; new_cols.append(c)
            else:
                counts[c] += 1; new_cols.append(f"{c}_{counts[c]}")
        df.columns = new_cols

    # Ensure core dtypes + safe text columns
    df[TIME_COL] = pd.to_datetime(df[TIME_COL], errors="coerce")
    for c in set(DIM_CATEGORICAL) | {PATH_KEY}:
        if c in df.columns:
            df[c] = df[c].astype("string")
    for col in ["processed_headline", "processed_body"]:
        if col in df.columns:
            df[col] = df[col].fillna("").astype(str)
        else:
            df[col] = ""

    # Conversion flag (auto-tuned to hit >= target)
    df["is_conversion"] = autotune_conversion(df)
    print(f"[conv] final count = {int(df['is_conversion'].sum())}")

    # Numeric bins
    binned_cols = [add_quantile_bins(df, c) for c in DIM_NUMERIC if c in df.columns]

    # Dimensions (skip PATH_KEY itself)
    dimensions = [d for d in DIM_CATEGORICAL if d in df.columns and d != PATH_KEY] + binned_cols

    # ---------- Item attribution ----------
    item_frames = []
    for dim in dimensions:
        print(f"[attrib] {dim}")
        paths, w = build_paths(df, dim)
        res = markov_from_paths(paths, w)
        res = res.rename(columns={"state": "value"})
        res["kind"] = "item"
        res["dimension"] = dim
        item_frames.append(res[["kind","dimension","value","credit","credit_share"]])

    items_df = pd.concat(item_frames, ignore_index=True) if item_frames else pd.DataFrame(
        columns=["kind","dimension","value","credit","credit_share"]
    )
    items_df = add_ratings(items_df, "dimension") if not items_df.empty else items_df.assign(rating=[], rating_pct=[])

    # ---------- Term attribution ----------
    terms: List[str] = []
    if KEYWORDS.exists():
        terms += pd.read_csv(KEYWORDS, header=None)[0].dropna().astype(str).str.strip().tolist()
    if BIGRAMS.exists():
        try:
            t = pd.read_csv(BIGRAMS, header=None)[0]
        except Exception:
            t = pd.read_csv(BIGRAMS).iloc[:, 0]
        terms += t.dropna().astype(str).str.strip().tolist()
    # de-dupe preserving order
    seen = set(); terms = [t for t in terms if not (t in seen or seen.add(t))]

    term_df = pd.DataFrame(columns=["kind","dimension","value","credit","credit_share","rating","rating_pct"])
    if terms:
        print(f"[terms] {len(terms)} keywords/bigrams")
        p, w = build_paths_terms(df, terms)
        term_res = markov_from_paths(p, w)
        term_res = term_res[term_res["state"].str.startswith("TERM::")].copy()
        term_res["value"] = term_res["state"].str.replace("TERM::", "", regex=False)
        term_res["kind"] = "term"
        term_res["dimension"] = "term"
        term_df = term_res[["kind","dimension","value","credit","credit_share"]]
        term_df = add_ratings(term_df, "dimension")

    # ---------- Unified output ----------
    all_attr = pd.concat([items_df, term_df], ignore_index=True)
    out_all = OUT_DIR / "attribution_all_scored.csv"
    all_attr.to_csv(out_all, index=False)
    print(f"[ok] saved -> {out_all}")

if __name__ == "__main__":
    main()