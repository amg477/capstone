import pandas as pd
from collections import Counter, defaultdict
from typing import Optional, Sequence
from pathlib import Path

def _nonneg_weights(w: pd.Series) -> pd.Series:
    w = pd.to_numeric(w, errors="coerce").fillna(0.0)
    if (w < 0).any():
        w = (w - w.min()).clip(lower=0.0)
    return w

def extract_top_terms_from_processed(
    df: pd.DataFrame,
    text_cols: Sequence[str] = ("processed_headline", "processed_body"),
    weight_col: Optional[str] = None,
    top_k: int = 1000,
    min_len: int = 3,
    include_bigrams: bool = False,
) -> pd.DataFrame:
    """
    Extract top terms from one or more preprocessed text columns.

    Parameters
    ----------
    text_cols : list/tuple of str
        Column names whose contents will be concatenated per row.
    weight_col : str or None
        Optional column of non-negative weights.
    top_k : int
        Return the top_k most frequent/weighted terms.
    min_len : int
        Minimum token length to keep.
    include_bigrams : bool
        Whether to include bigram counts as well.

    Returns
    -------
    DataFrame with columns:
        word, count, doc_freq, [weighted_count]
    """
    # Combine specified text columns into one string per row
    missing = [c for c in text_cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns: {missing}")
    texts = (
        df[list(text_cols)]
        .fillna("")
        .astype(str)
        .agg(" ".join, axis=1)
    )

    # Optional weights
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

        freq.update(toks)
        if weight_col and (weight_col in df.columns):
            for t in toks:
                wfreq[t] += float(wt)

        # document frequency
        seen = set(toks)

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
    out = pd.DataFrame({
        "word": words,
        "count": [freq[w] for w in words],
        "doc_freq": [docfreq[w] for w in words],
    })
    if weight_col and (weight_col in df.columns):
        out["weighted_count"] = [wfreq[w] for w in words]

    sort_cols = (
        ["weighted_count", "count", "doc_freq"]
        if "weighted_count" in out.columns
        else ["count", "doc_freq"]
    )
    return out.sort_values(sort_cols, ascending=False, kind="mergesort").head(top_k).reset_index(drop=True)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parents[2]
    IN_CSV  = ROOT / "data" / "final_dataset_sampled.csv"
    OUT_DIR = ROOT / "data" / "processed"

    df = pd.read_csv(IN_CSV)

    # --- Top 1000 unigrams from headline + body (weighted by vipr_weight) ---
    top_terms = extract_top_terms_from_processed(
        df,
        text_cols=("processed_headline", "processed_body"),
        weight_col="vipr_weight",
        top_k=1000,
        min_len=3,
        include_bigrams=False
    )
    top_terms.to_csv(OUT_DIR / "top_1000_words.csv", index=False)

    # --- Optional: include bigrams ---
    top_terms_bi = extract_top_terms_from_processed(
        df,
        text_cols=("processed_headline", "processed_body"),
        weight_col="vipr_weight",
        top_k=1000,
        min_len=3,
        include_bigrams=True
    )
    top_terms_bi.to_csv(OUT_DIR / "top_1000_bigrams.csv", index=False)

    # Keyword list for attribution
    pd.Series(top_terms["word"]).to_csv(
        OUT_DIR / "top_1000_keywords.csv", index=False, header=False
    )
    print(f"Saved keyword files in {OUT_DIR}")