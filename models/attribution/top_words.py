# keywords_from_processed.py
import pandas as pd
from collections import Counter, defaultdict
from typing import Optional
from pathlib import Path

def _nonneg_weights(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    if (w < 0).any():
        w = (w - w.min()).clip(lower=0.0)
    return w

def extract_top_terms_from_processed(
    df: pd.DataFrame,
    text_col: str = "processed_text",
    weight_col: Optional[str] = None,  # <-- None (not "None")
    top_k: int = 300,
    min_len: int = 3,
    include_bigrams: bool = False,
) -> pd.DataFrame:
    """
    Extract top terms from a preprocessed text column (space-separated tokens).
    Returns: columns word, count, doc_freq, [weighted_count]
    """
    if text_col not in df.columns:
        raise KeyError(f"'{text_col}' not in DataFrame")

    texts = df[text_col].fillna("").astype(str)

    # Optional weights (non-negative)
    if weight_col and (weight_col in df.columns):
        w = _nonneg_weights(df[weight_col])
    else:
        w = pd.Series(1.0, index=df.index)

    freq = Counter()
    wfreq = Counter()
    docfreq = defaultdict(int)

    for txt, wt in zip(texts.values, w.values):
        toks = [t for t in txt.split() if len(t) >= min_len]
        if not toks:
            continue

        # unigrams
        freq.update(toks)
        if weight_col and (weight_col in df.columns):
            for t in toks:
                wfreq[t] += float(wt)

        # document frequency: unique terms in this doc
        seen = set(toks)

        # bigrams (optional)
        if include_bigrams and len(toks) > 1:
            bigs = [f"{a} {b}" for a, b in zip(toks[:-1], toks[1:])]
            freq.update(bigs)
            if weight_col and (weight_col in df.columns):
                for b in bigs:
                    wfreq[b] += float(wt)
            seen.update(bigs)

        for t in seen:
            docfreq[t] += 1

    if not freq:
        return pd.DataFrame(columns=["word", "count", "doc_freq", "weighted_count"])

    words = list(freq.keys())
    data = {
        "word": words,
        "count": [freq[w] for w in words],
        "doc_freq": [docfreq[w] for w in words],
    }
    if weight_col and (weight_col in df.columns):
        data["weighted_count"] = [wfreq[w] for w in words]

    out = pd.DataFrame(data)

    # Sort preference: weighted_count (if present) -> count -> doc_freq
    sort_cols = ["weighted_count", "count", "doc_freq"] if "weighted_count" in out.columns else ["count", "doc_freq"]
    out = out.sort_values(sort_cols, ascending=False, kind="mergesort").head(top_k).reset_index(drop=True)
    return out

if __name__ == "__main__":
    # Resolve paths from project root (…/capstone/capstone)
    ROOT = Path(__file__).resolve().parents[2]  # models/attribution -> models -> capstone
    IN_CSV  = ROOT / "data" / "processed" / "text_processed_data.csv"
    OUT_DIR = ROOT / "data" / "processed"

    df = pd.read_csv(IN_CSV)

    # --- Top 300 unigrams (weighted by vipr_weight) ---
    top_uni = extract_top_terms_from_processed(
        df,
        text_col="processed_text",
        weight_col="vipr_weight",   # set to None for pure frequency
        top_k=300,
        min_len=3,
        include_bigrams=False
    )
    top_uni.to_csv(OUT_DIR / "top_300_unigrams.csv", index=False)

    # --- (Optional) Top 300 including bigrams ---
    top_uni_bi = extract_top_terms_from_processed(
        df,
        text_col="processed_text",
        weight_col="vipr_weight",
        top_k=300,
        min_len=3,
        include_bigrams=True
    )
    top_uni_bi.to_csv(OUT_DIR / "top_300_terms_unigrams_bigrams.csv", index=False)

    # Plain list for attribution keyword mode (unigrams only)
    pd.Series(top_uni["word"].tolist()).to_csv(
        OUT_DIR / "top_keywords_300_list.csv", index=False, header=False
    )
    print(f"Saved keyword files in {OUT_DIR}")