"""
Unified data processing (lean):
- Fill NAs in text cols (+optional extra cols)
- Log1p: vipr_weight, hit_strength, circulation_size (if present)
- Clip vipr_score to 1st–99th pct (if present)
- NLTK clean: headline + article_body -> processed_* and combined processed_text
- Token counts
- Save CSV
"""

import re, sys
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from typing import Optional, List  # add this import at the top

def process_and_save(
    df: pd.DataFrame,
    body_col: str = "article_body",
    headline_col: str = "headline",
    out_csv: str = "processed_text.csv",
    force_fill_cols: Optional[List[str]] = None,
    verbose: bool = True,
) -> pd.DataFrame:

    STOP = set(stopwords.words("english"))
    LEMM = WordNetLemmatizer()
    NONLETTERS = re.compile(r"[^a-z\s]")
    out = df.copy()

    # Fill NAs in text columns
    text_cols = out.select_dtypes(include=["object", "string"]).columns
    if len(text_cols): out[text_cols] = out[text_cols].fillna("Unknown")
    if force_fill_cols:
        cols = [c for c in force_fill_cols if c in out.columns]
        if cols: out[cols] = out[cols].fillna("Unknown")

    # Numeric transforms
    for col in ("vipr_weight","hit_strength","circulation_size"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out[f"{col}_log"] = np.log1p(out[col])
    if "vipr_score" in out.columns:
        q1, q99 = out["vipr_score"].quantile([0.01, 0.99])
        out["vipr_score_clip"] = out["vipr_score"].clip(q1, q99)

    # 3) Text cleaning
    if body_col not in out.columns:
        raise KeyError(f"Missing required column: {body_col}")

    def clean(x: str) -> str:
        if not isinstance(x, str): return ""
        s = NONLETTERS.sub(" ", x.lower())
        toks = (t for t in s.split() if len(t) > 2 and t not in STOP)
        return " ".join(LEMM.lemmatize(t) for t in toks)

    out["processed_headline"] = out[headline_col].fillna("").apply(clean) if headline_col in out.columns else ""
    out["processed_body"]     = out[body_col].fillna("").apply(clean)
    out["processed_text"]     = (out["processed_headline"].str.strip() + " " + out["processed_body"].str.strip()).str.strip()

    # Token counts (fast regex)
    tok_pat = r"\S+"
    out["headline_token_count"] = out["processed_headline"].str.count(tok_pat)
    out["body_token_count"]     = out["processed_body"].str.count(tok_pat)
    out["token_count"]          = out["processed_text"].str.count(tok_pat)

    # 4) Save
    out.to_csv(out_csv, index=False)
    if verbose:
        created = [c for c in out.columns if c.endswith("_log") or c.endswith("_clip") or c.startswith("processed_")]
        print("Created:", created)
        print(f"Saved -> {out_csv} (rows: {len(out)})")
    return out

if __name__ == "__main__":
    in_csv  = "data/processed/processed_data.csv"
    out_csv = "data/processed/text_processed_data.csv"

    print("Starting data processing...")
    df = pd.read_csv(in_csv)

    process_and_save(
        df,
        out_csv=out_csv,
        force_fill_cols=["genre"],
        verbose=True)

    print(f"Data processing complete. Rows: {len(df)} | Saved to: {out_csv}")