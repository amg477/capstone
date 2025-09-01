# markov_attribution.py
from __future__ import annotations
import numpy as np
import pandas as pd
import re
from typing import Iterable, List, Optional, Tuple, Dict, Callable

START = "<START>"
CONV = "<CONV>"

# ---------------------------
# Helpers
# ---------------------------
def _to_datetime(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True).dt.tz_localize(None)

def _extract_keywords(text: str, keywords: Iterable[str]) -> List[str]:
    if not isinstance(text, str):
        return []
    t = text.lower()
    found = []
    for k in keywords:
        if re.search(rf"\b{re.escape(k.lower())}\b", t):
            found.append(k)
    return found

def _ensure_nonneg_weights(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    if (w < 0).any():
        w = (w - w.min()).clip(lower=0)
    return w

def _collapse_consecutive(seq: List[str]) -> List[str]:
    out = []
    last = None
    for x in seq:
        if x != last:
            out.append(x)
            last = x
    return out

# ---------------------------
# Path builder
# ---------------------------
def build_paths(
    df: pd.DataFrame,
    path_key: str,                  # e.g., "tag_name" (grouping for a sequence)
    time_col: str,                  # e.g., "load_date"
    state_col: str,                 # e.g., "publication_name" (or "source_type_name" / "author_name")
    weight_col: Optional[str] = None,  # e.g., "vipr_weight" to weight paths
    keyword_mode: bool = False,     # if True, states become publication::keyword (see below)
    keywords: Optional[Iterable[str]] = None,
    article_text_col: str = "article_body",
    filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
    drop_na_states: bool = True,
) -> pd.DataFrame:
    """
    Returns a tidy DataFrame with columns:
      - path: list[str]  (sequence of states)
      - weight: float     (path weight; default 1.0 or sum of weights within the group)

    Notes
    -----
    - Each path groups rows by `path_key` and orders them by `time_col`.
    - States normally come from `state_col`. If `keyword_mode=True`, states become
      f"{state_col_value}::{keyword}", repeating an item for each matched keyword.
    - `filter_fn` can subset df before building paths (e.g. only certain authors or channels).
    """
    data = df.copy()
    data[time_col] = _to_datetime(data[time_col])

    if filter_fn is not None:
        data = filter_fn(data)

    if drop_na_states:
        data = data[~data[state_col].isna()]

    if keyword_mode:
        if not keywords:
            raise ValueError("keyword_mode=True requires a non-empty `keywords` list.")
        # create exploded rows per matched keyword
        data["_keywords"] = data[article_text_col].apply(lambda t: _extract_keywords(t, keywords))
        data = data.explode("_keywords", ignore_index=False)
        data = data.dropna(subset=["_keywords"])
        data["_state"] = data[state_col].astype(str) + "::" + data["_keywords"].astype(str)
    else:
        data["_state"] = data[state_col].astype(str)

    # order within each path_key by time
    data = data.sort_values([path_key, time_col])
    # weight per row (used to aggregate path-level weight)
    if weight_col and (weight_col in data.columns):
        data["_row_w"] = _ensure_nonneg_weights(data[weight_col])
    else:
        data["_row_w"] = 1.0

    # group into paths
    paths = []
    for g_key, g in data.groupby(path_key, sort=False):
        seq = g["_state"].tolist()
        seq = [s for s in seq if isinstance(s, str) and len(s)]
        seq = _collapse_consecutive(seq)  # avoid self-loops inflated by duplicates
        if not seq:
            continue
        w = float(g["_row_w"].sum())  # path weight = sum of row weights in that sequence
        paths.append({"path_key": g_key, "path": seq, "weight": w})

    if not paths:
        return pd.DataFrame(columns=["path_key", "path", "weight"])

    return pd.DataFrame(paths)


# ---------------------------
# Markov Attribution (Removal Effect)
# ---------------------------

def _transition_counts(
    paths_df: pd.DataFrame,
    var_path: str = "path",
    var_weight: str = "weight",
) -> Dict[Tuple[str, str], float]:
    """
    Count weighted transitions, including START->first and last->CONV.
    """
    counts: Dict[Tuple[str, str], float] = {}
    for _, row in paths_df.iterrows():
        path: List[str] = row[var_path]
        w: float = float(row[var_weight])

        if not path:
            # empty path -> START->CONV directly
            counts[(START, CONV)] = counts.get((START, CONV), 0.0) + w
            continue

        # START -> first
        counts[(START, path[0])] = counts.get((START, path[0]), 0.0) + w
        # internal transitions
        for a, b in zip(path[:-1], path[1:]):
            counts[(a, b)] = counts.get((a, b), 0.0) + w
        # last -> CONV
        counts[(path[-1], CONV)] = counts.get((path[-1], CONV), 0.0) + w

    return counts

def _normalize_transition_matrix(counts: Dict[Tuple[str, str], float]):
    """
    Build row-stochastic transition matrix (P), along with state index mapping.
    """
    states = sorted(set([a for (a, b) in counts] + [b for (a, b) in counts]))
    idx = {s: i for i, s in enumerate(states)}
    P = np.zeros((len(states), len(states)), dtype=float)

    # row sums by origin
    row_sums: Dict[str, float] = {}
    for (a, b), c in counts.items():
        row_sums[a] = row_sums.get(a, 0.0) + c

    for (a, b), c in counts.items():
        if row_sums[a] > 0:
            P[idx[a], idx[b]] += c / row_sums[a]

    return P, states, idx

def _absorbing_conversion_probability(P: np.ndarray, states: List[str], start_state: str = START, conv_state: str = CONV) -> float:
    """
    Compute probability of absorption in CONV when starting from START.
    Uses the standard absorbing Markov chain formulation with fundamental matrix N = (I - Q)^(-1).
    """
    # identify absorbing vs transient
    absorbing = []
    transient = []
    for i, s in enumerate(states):
        if np.isclose(P[i].sum(), 1.0) and np.isclose(P[i, i], 1.0):
            # perfectly absorbing self-loop (not typical here)
            absorbing.append(i)
        elif s == conv_state:
            absorbing.append(i)
        else:
            transient.append(i)

    if not transient:
        return 0.0

    # Build Q (transient->transient) and R (transient->absorbing)
    Q = P[np.ix_(transient, transient)]
    R = P[np.ix_(transient, absorbing)]

    I = np.eye(Q.shape[0])
    try:
        N = np.linalg.inv(I - Q)
    except np.linalg.LinAlgError:
        # fallback: pseudo-inverse if ill-conditioned
        N = np.linalg.pinv(I - Q)

    # starting vector: 1 at START (must be transient), else return 0
    try:
        start_idx_full = states.index(start_state)
    except ValueError:
        return 0.0
    if start_idx_full not in transient:
        return 0.0

    start_idx = transient.index(start_idx_full)
    e_start = np.zeros((1, len(transient)))
    e_start[0, start_idx] = 1.0

    # probability of absorption into each absorbing state
    B = e_start @ N @ R  # shape (1, n_absorb)
    # find the absorbing column for CONV
    try:
        conv_abs_idx_full = states.index(conv_state)
    except ValueError:
        return 0.0
    if conv_abs_idx_full not in absorbing:
        return 0.0

    conv_abs_idx = absorbing.index(conv_abs_idx_full)
    p_conv = float(B[0, conv_abs_idx])
    # clip numerical noise
    return float(np.clip(p_conv, 0.0, 1.0))

def _remove_state_from_paths(paths_df: pd.DataFrame, state_to_remove: str,
                             var_path: str = "path", var_weight: str = "weight") -> pd.DataFrame:
    """
    Remove a state from every path, collapsing duplicates, dropping paths
    that become empty (they will count as START->CONV when counting transitions).
    """
    new_rows = []
    for _, row in paths_df.iterrows():
        seq = [s for s in row[var_path] if s != state_to_remove]
        seq = _collapse_consecutive(seq)
        new_rows.append({var_path: seq, var_weight: row[var_weight]})
    return pd.DataFrame(new_rows)

def markov_attribution_removal_effect(
    paths_df: pd.DataFrame,
    var_path: str = "path",
    var_weight: str = "weight",
) -> pd.DataFrame:
    """
    Computes removal-effect attribution for every state:
      credit(state) = P_conv_baseline - P_conv_without_state

    Returns tidy DataFrame with:
      state, credit, credit_share
    """
    if paths_df.empty:
        return pd.DataFrame(columns=["state", "credit", "credit_share"])

    # Baseline conversion probability
    counts = _transition_counts(paths_df, var_path, var_weight)
    P, states, _ = _normalize_transition_matrix(counts)
    baseline = _absorbing_conversion_probability(P, states, START, CONV)

    # Evaluate removal effect for each state except START/CONV
    credits = {}
    for s in states:
        if s in (START, CONV):
            continue
        mod_paths = _remove_state_from_paths(paths_df, s, var_path, var_weight)
        mod_counts = _transition_counts(mod_paths, var_path, var_weight)
        Pm, Sm, _ = _normalize_transition_matrix(mod_counts)
        p_conv = _absorbing_conversion_probability(Pm, Sm, START, CONV)
        credits[s] = max(0.0, baseline - p_conv)  # non-negative

    if not credits:
        return pd.DataFrame(columns=["state", "credit", "credit_share"])

    out = pd.DataFrame({"state": list(credits.keys()), "credit": list(credits.values())})
    total = out["credit"].sum()
    if total > 0:
        out["credit_share"] = out["credit"] / total
    else:
        out["credit_share"] = 0.0
    out = out.sort_values("credit", ascending=False).reset_index(drop=True)
    return out

# ---------------------------
# Convenience runner
# ---------------------------

def run_markov_attribution(
    df: pd.DataFrame,
    path_key: str = "tag_name",
    time_col: str = "load_date",
    state_col: str = "publication_name",    # swap to "source_type_name" or "author_name" easily
    weight_col: Optional[str] = "vipr_weight",
    keyword_mode: bool = False,
    keywords: Optional[Iterable[str]] = None,
    filter_fn: Optional[Callable[[pd.DataFrame], pd.DataFrame]] = None,
) -> pd.DataFrame:
    """
    One-call helper: build paths -> compute removal-effect attribution.
    """
    paths = build_paths(
        df=df,
        path_key=path_key,
        time_col=time_col,
        state_col=state_col,
        weight_col=weight_col,
        keyword_mode=keyword_mode,
        keywords=keywords,
        article_text_col="article_body",
        filter_fn=filter_fn,
    )
    return markov_attribution_removal_effect(paths)