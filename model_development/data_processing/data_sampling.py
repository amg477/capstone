#!/usr/bin/env python3
"""
Simple Stratified Sampler (auto-run)
-----------------------------------
Reads processed_data.parquet, performs proportional stratified sampling
to a target N (default 100,000), and writes final_dataset_sampled.parquet.

Usage:
    python data_sampling.py                  # uses default 100k
    python data_sampling.py --n 100000       # specify target rows
    python data_sampling.py --seed 2025      # set RNG seed
"""

from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Config (paths + defaults)
# ---------------------------------------------------------------------
ROOT = Path("/Users/annaglass/capstone/capstone")
RAW_DIR       = ROOT / "data_storage" / "raw_data"
INPUT_PATH  = ROOT / "data_storage" / "processed_data" / "processed_data.parquet"
OUTPUT_PATH = ROOT / "data_storage" / "processed_data" / "sampled_data.parquet"

DEFAULT_TARGET_N = 100_000
DEFAULT_SEED     = 2025
DEFAULT_STRATA   = ["tag_name", "source_type", "sentiment_band"]

# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def pick_strata(df: pd.DataFrame, candidates) -> list[str]:
    return [c for c in candidates if c in df.columns]

def stratified_sample(df: pd.DataFrame, strata_cols: list[str], n: int, random_state: int = 42) -> pd.DataFrame:
    if n <= 0 or len(df) <= n or not strata_cols:
        return df.sample(n=min(n, len(df)), random_state=random_state)

    g = df.groupby(strata_cols, dropna=False, observed=True).size().rename("size").reset_index()
    total = float(g["size"].sum())
    raw = g["size"] / total * n
    floor_q = np.floor(raw).astype(int)
    remainder = n - int(floor_q.sum())

    if remainder > 0:
        frac = raw - floor_q
        add_idx = np.argsort(-frac.values)[:remainder]
        floor_q.iloc[add_idx] += 1

    floor_q = floor_q.clip(upper=g["size"])

    samples = []
    for i, row in g.iterrows():
        quota = int(floor_q.iloc[i])
        if quota <= 0:
            continue
        mask = np.ones(len(df), dtype=bool)
        for c in strata_cols:
            mask &= (df[c] == row[c])
        grp = df.loc[mask]
        if len(grp) == 0:
            continue
        samples.append(grp.sample(n=min(quota, len(grp)), random_state=random_state))

    sampled = pd.concat(samples, ignore_index=False) if samples else df.head(0)

    short = n - len(sampled)
    if short > 0:
        remaining = df.drop(index=sampled.index, errors="ignore")
        if len(remaining) > 0:
            extra = remaining.sample(n=min(short, len(remaining)), random_state=random_state)
            sampled = pd.concat([sampled, extra], ignore_index=False)

    return sampled.sample(frac=1.0, random_state=random_state)

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=DEFAULT_TARGET_N, help="target number of rows")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help="random seed")
    args = ap.parse_args()

    print(f"[load] {INPUT_PATH}")
    df = pd.read_parquet(INPUT_PATH)
    print(f"[info] {len(df):,} total rows")

    strata_cols = pick_strata(df, DEFAULT_STRATA)
    print(f"[strata] using {strata_cols}" if strata_cols else "[strata] none found; simple random sample")

    target_n = min(args.n, len(df))
    print(f"[sample] target = {target_n:,} rows (seed={args.seed})")

    df_sampled = stratified_sample(df, strata_cols, n=target_n, random_state=args.seed)
    print(f"[result] {len(df_sampled):,} sampled rows")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] parquet -> {OUTPUT_PATH}")
    df_sampled.to_parquet(OUTPUT_PATH, index=False)
    print("sampled has:", {"article_id": "article_id" in df_sampled.columns,
                       "load_date": "load_date" in df_sampled.columns})
    print("[ok] done.")

if __name__ == "__main__":
    main()