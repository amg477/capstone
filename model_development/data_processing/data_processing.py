#!/usr/bin/env python3
"""
data_processing.py

EXECUTION ORDER: Step 1 of 5 in the data processing pipeline.

This script performs the initial data processing steps:
  Stage 1: Combines multiple Excel files, applies filters (language='en', country='US'),
           and unifies column schemas → combined_data.parquet
  Stage 2: Performs proportional stratified sampling → sampled_data.parquet

INPUT FILES:
  - data_storage/raw_data/penta_raw_1.xlsx through penta_raw_9.xlsx

OUTPUT FILES:
  - data_storage/processed_data/combined_data.parquet
  - data_storage/processed_data/sampled_data.parquet

USAGE:
    python data_processing.py

ENVIRONMENT VARIABLES (optional):
  SAMPLE_N=100000      # Target number of rows for sampling (default: 50000)
  SAMPLE_SEED=2025     # Random seed for reproducible sampling (default: 2025)

NOTES:
  - No spaCy or NLTK dependencies required at this stage
  - Reads only needed columns from Excel files for speed
  - Ensures headline/body columns exist but does NOT modify their contents
  - Text processing and person extraction happen in subsequent steps

NEXT STEP:
  After this script completes, run: python names_then_text.py
"""

import os
import sys
import gc
from pathlib import Path
from typing import List, Tuple, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# ---------------- Configuration ----------------
ROOT = Path(__file__).resolve().parents[2]  # .../capstone/capstone

RAW_DIR       = ROOT / "data_storage" / "raw_data"
PROCESSED_DIR = ROOT / "data_storage" / "processed_data"
FINAL_DIR     = ROOT / "data_storage" / "final_data"       # not used here, kept for compatibility

COMBINED_OUT  = PROCESSED_DIR / "combined_data.parquet"
SAMPLED_OUT   = PROCESSED_DIR / "sampled_data.parquet"

EXCEL_FILES = [
    RAW_DIR / "penta_raw_1.xlsx",
    RAW_DIR / "penta_raw_2.xlsx",
    RAW_DIR / "penta_raw_3.xlsx",
    RAW_DIR / "penta_raw_4.xlsx",
    RAW_DIR / "penta_raw_5.xlsx",
    RAW_DIR / "penta_raw_6.xlsx",
    RAW_DIR / "penta_raw_7.xlsx",
    RAW_DIR / "penta_raw_8.xlsx",
    RAW_DIR / "penta_raw_9.xlsx",
]

# Drop iso_language_code only AFTER filtering to 'en'
DROP_COLS = [
    "publisher_site_url",
    "page",
    "article_url",
    "source_unique_id",
    "source_type_name",  # dupe of source_type
    "region",            # not clean; sub_region is more reliable
    "load_datetime",
    "load_date",
    "engagement",
    "genre",
    "iso_language_code",
    # "country" is optionally dropped after US filter (see FILTER_US_ONLY)
]

HEADLINE_COL = "headline"
BODY_COL     = "article_body"

# Categorical text columns you want preserved (not normalized here)
CAT_TEXT_COLS = [
    "tag_name", "source_feed_name", "feed_name", "author_name", "source_type",
    "sentiment_band", "sub_region", "country", "channel_name", "channel",
    "publisher_name", "publication_name",
]

ID_PASS_THRU = {"article_id"}

# Early row-filters
FILTER_US_ONLY = True
NUMERIC_REQUIRED = ["vipr_weight", "circulation", "sentiment"]  # drop NA if present

# Sampling config
DEFAULT_TARGET_N = int(os.getenv("SAMPLE_N", "50000"))
DEFAULT_SEED     = int(os.getenv("SAMPLE_SEED", "2025"))
DEFAULT_STRATA   = ["tag_name", "source_type", "sentiment_band"]

def _sizeof(path: Path) -> str:
    try:
        b = path.stat().st_size
    except FileNotFoundError:
        return "0 B"
    units = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while b >= 1024 and i < len(units) - 1:
        b /= 1024.0
        i += 1
    return f"{b:.1f} {units[i]}"

def _canonicalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Integers → nullable Int64
    int_like = df.select_dtypes(include=["int64", "int32", "int16", "int8", "uint8", "uint16", "uint32"]).columns
    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Floats → nullable Float32
    float_like = df.select_dtypes(include=["float64", "float32"]).columns
    for c in float_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float32")

    # Objects → Arrow strings
    obj_like = df.select_dtypes(include=["object"]).columns
    for c in obj_like:
        df[c] = df[c].astype("string[pyarrow]")
    return df

############################
# Stage 1: Combine & Clean (columns only)
############################
def _prescan_header_union(excel_files: List[Path]) -> Set[str]:
    union_cols: Set[str] = set()
    for p in excel_files:
        if not p.exists():
            raise FileNotFoundError(f"Missing Excel file: {p}")
        df0 = pd.read_excel(p, engine="openpyxl", nrows=0)
        union_cols.update(df0.columns.tolist())
    return union_cols

def combine_and_align(excel_files: List[Path], out_parquet: Path) -> Tuple[Path, int]:
    union_cols = _prescan_header_union(excel_files)

    final_drop = set(DROP_COLS)
    if FILTER_US_ONLY:
        final_drop = final_drop | {"country"}  # drop after filter
    output_cols = sorted([c for c in union_cols if c not in final_drop])

    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    if out_parquet.exists():
        print(f"Removing existing file to avoid schema conflicts: {out_parquet}")
        out_parquet.unlink()

    # Read only needed columns + filter keys
    read_cols = sorted(set(output_cols) | {"iso_language_code", "country"})

    parquet_writer: pq.ParquetWriter | None = None
    target_schema: pa.Schema | None = None
    kept_total = 0
    scanned_total = 0

    print("── Stage 1: Combine & Align (union schema) ──")
    print(f"Target output columns: {len(output_cols)} columns")

    for p in excel_files:
        print(f"Reading {p.name} ...")
        df = pd.read_excel(p, engine="openpyxl", usecols=read_cols)
        scanned_total += len(df)

        # Language filter
        if "iso_language_code" in df.columns:
            before = len(df)
            df = df[df["iso_language_code"] == "en"]
            print(f"  Language filter (en): {before:,} → {len(df):,}")
        else:
            print("  iso_language_code not found; skipping language filter.")

        # Geography filter (US)
        if FILTER_US_ONLY and "country" in df.columns:
            before = len(df)
            df = df[df["country"] == "United States"]
            print(f"  Country filter (US):  {before:,} → {len(df):,}")
        elif FILTER_US_ONLY:
            print("  country column not found; US filter skipped.")

        # Required numeric non-NA
        req_present = [c for c in NUMERIC_REQUIRED if c in df.columns]
        if req_present:
            before = len(df)
            df = df.dropna(subset=req_present)
            if len(df) != before:
                print(f"  Drop NA in {req_present}: {before:,} → {len(df):,}")

        # Drop metadata columns (after filters)
        drop_now = [c for c in DROP_COLS if c in df.columns]
        if FILTER_US_ONLY and "country" in df.columns:
            drop_now.append("country")
        if drop_now:
            df = df.drop(columns=drop_now, errors="ignore")
            print(f"Dropped columns: {', '.join(sorted(set(drop_now)))}")

        # Align to union schema
        missing = [c for c in output_cols if c not in df.columns]
        for c in missing:
            df[c] = pd.NA
        df = df[output_cols]

        # Keep date types gentle (parse if present)
        DATE_COL = "published_datetime"
        if DATE_COL in df.columns:
            print(f"  Parsing {DATE_COL} as datetime ...")
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)

        # Ensure headline/body exist (no cleaning)
        if HEADLINE_COL not in df.columns:
            df[HEADLINE_COL] = pd.Series(pd.array([None] * len(df), dtype="string[pyarrow]"))
        if BODY_COL not in df.columns:
            df[BODY_COL] = pd.Series(pd.array([None] * len(df), dtype="string[pyarrow]"))

        # Canonical dtypes
        df = _canonicalize_dtypes(df)

        # Append to Parquet with locked schema
        table = pa.Table.from_pandas(df, preserve_index=False)
        if parquet_writer is None:
            target_schema = table.schema
            parquet_writer = pq.ParquetWriter(
                out_parquet,
                target_schema,
                compression="zstd",
                use_dictionary=True
            )
        else:
            if table.schema != target_schema:
                table = table.cast(target_schema)
        parquet_writer.write_table(table)

        kept_total += len(df)
        del df, table
        gc.collect()

    if parquet_writer is not None:
        parquet_writer.close()

    print(f"Stage 1 saved: {out_parquet} [{_sizeof(out_parquet)}]")
    print(f"Rows scanned (all files): {scanned_total:,}")
    print(f"Rows kept (post-filters): {kept_total:,}")
    return out_parquet, kept_total

############################
# Stage 2: Stratified Sampling
############################
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

def sample_combined(input_path: Path, output_path: Path,
                    target_n: int = DEFAULT_TARGET_N, seed: int = DEFAULT_SEED) -> Path:
    print(f"[load] {input_path}")
    df = pd.read_parquet(input_path)
    print(f"[info] {len(df):,} total rows")

    strata_cols = pick_strata(df, DEFAULT_STRATA)
    print(f"[strata] using {strata_cols}" if strata_cols else "[strata] none found; simple random sample")

    target_n = min(target_n, len(df))
    print(f"[sample] target = {target_n:,} rows (seed={seed})")

    sampled = stratified_sample(df, strata_cols, n=target_n, random_state=seed)
    print(f"[result] {len(sampled):,} sampled rows")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[write] parquet -> {output_path}")
    sampled.to_parquet(output_path, index=False)
    print("[ok] sampled parquet written.")
    return output_path

############################
# Main
############################
def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_DIR.mkdir(parents=True, exist_ok=True)

    combined_path, n1 = combine_and_align(EXCEL_FILES, COMBINED_OUT)

    sample_n    = DEFAULT_TARGET_N
    sample_seed = DEFAULT_SEED
    sampled_path = sample_combined(combined_path, SAMPLED_OUT, target_n=sample_n, seed=sample_seed)

    print("\nDone.")
    print(f"Rows after Stage 1 (combined): {n1:,}")
    print(f"Combined data: {combined_path}")
    print(f"Sampled data: {sampled_path}")

if __name__ == "__main__":
    sys.exit(main())