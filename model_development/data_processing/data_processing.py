#!/usr/bin/env python3
"""
data_processing.py

Cleans and prepares the raw Penta datasets for modeling and analysis.

Steps:
1. Combine and filter nine Excel files from `/data/raw/`, keeping only English-language rows.
2. Drop redundant metadata columns (IDs, URLs, timestamps, etc.).
3. Normalize categorical columns (lowercase, depunctuate, replace spaces/dashes with underscores).
   Canonicalize variations of unknown values (e.g., "uncredited", "other", "n/a") → "unknown".
4. Clean unstructured text (`headline`, `article_body`) using NLTK:
   lowercase, remove non-letters, drop stopwords, lemmatize.
5. Output Parquet files:
   - `/data/processed/combined_data.parquet` — combined, filtered dataset.
   - `/data/processed/processed_data.parquet` — fully cleaned dataset.
"""

#######################
# Setup
#######################
import os
import sys
from pathlib import Path
from typing import List, Tuple, Set

import re
import gc
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

import nltk
nltk.download("stopwords"); nltk.download("wordnet"); nltk.download("punkt")

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ---------------- Configuration ----------------
from pathlib import Path

# Resolve project root robustly from this file's location:
# .../capstone/capstone/model_development/data_processing/data_processing.py
# parents[2] => .../capstone/capstone
ROOT = Path(__file__).resolve().parents[2]

RAW_DIR       = ROOT / "data_storage" / "raw_data"
PROCESSED_DIR = ROOT / "data_storage" / "processed_data"
FINAL_DIR     = ROOT / "data_storage" / "final_data"

COMBINED_OUT = str(PROCESSED_DIR / "combined_data.parquet")   # after Stage 1
FINAL_OUT    = str(PROCESSED_DIR / "processed_data.parquet")  # after Stage 2

EXCEL_FILES = [
    str(RAW_DIR / "penta_raw_1.xlsx"),
    str(RAW_DIR / "penta_raw_2.xlsx"),
    str(RAW_DIR / "penta_raw_3.xlsx"),
    str(RAW_DIR / "penta_raw_4.xlsx"),
    str(RAW_DIR / "penta_raw_5.xlsx"),
    str(RAW_DIR / "penta_raw_6.xlsx"),
    str(RAW_DIR / "penta_raw_7.xlsx"),
    str(RAW_DIR / "penta_raw_8.xlsx"),
    str(RAW_DIR / "penta_raw_9.xlsx"),
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
    # country is optionally dropped after US filter (see FILTER_US_ONLY)
]

HEADLINE_COL = "headline"
BODY_COL     = "article_body"

# Categorical text columns to normalize (lowercase, depunctuate, spaces/dashes -> _)
CAT_TEXT_COLS = [
    "tag_name", "source_feed_name", "feed_name", "author_name", "source_type",
    "sentiment_band", "sub_region", "country", "channel",
    "publisher_name", "publication_name",
]

# Add pass-through IDs (no normalization)
ID_PASS_THRU = {"article_id"}

# Optional early row-filters
FILTER_US_ONLY = True  # set False if you don't want US-only
NUMERIC_REQUIRED = ["vipr_weight", "circulation", "sentiment"]  # drop rows with NA in these, if present

# Stage 2 batch size (tune down if memory is tight)
STAGE2_BATCH_ROWS = 50_000

# ---------------- Utils ----------------
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

# Canonical dtype helpers (consistent across files)
def _canonicalize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    # Integers → nullable Int32 (avoids int8/int16 drift across files)
    int_like = df.select_dtypes(include=["int64", "int32", "int16", "int8", "uint8", "uint16", "uint32"]).columns
    for c in int_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")

    # Floats → nullable Float32 (uniform)
    float_like = df.select_dtypes(include=["float64", "float32"]).columns
    for c in float_like:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("Float32")

    # Strings → Arrow-backed strings
    obj_like = df.select_dtypes(include=["object"]).columns
    for c in obj_like:
        df[c] = df[c].astype("string[pyarrow]")

    return df

############################
# Stage 1: Combine & Clean
############################
def _prescan_header_union(excel_files: List[str]) -> Set[str]:
    """
    Read only headers (nrows=0) from all Excel files to build a union set of columns.
    Ensures the combined output is NOT limited by any single file's schema.
    """
    union_cols: Set[str] = set()
    for f in excel_files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Missing Excel file: {p}")
        df0 = pd.read_excel(p, engine="openpyxl", nrows=0)
        union_cols.update(df0.columns.tolist())
    return union_cols


def combine_and_clean(excel_files: List[str], out_parquet: str) -> Tuple[str, int]:
    """
    Read Excel files one-by-one (low memory), apply early row filters,
    drop metadata columns, and write a single Parquet with the UNION of columns
    across all files (minus the drop list). Returns (out_path, row_count).
    """
    # 0) Union of all columns across files (so we don't limit what's loaded)
    union_cols = _prescan_header_union(excel_files)

    # Columns that will appear in the final combined dataset
    final_drop = set(DROP_COLS)
    if FILTER_US_ONLY:
        final_drop = final_drop | {"country"}  # drop after filter
    output_cols = sorted([c for c in union_cols if c not in final_drop])

    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Removing existing file to avoid schema conflicts: {out_path}")
        out_path.unlink()

    parquet_writer: pq.ParquetWriter | None = None
    target_schema: pa.Schema | None = None
    kept_total = 0
    scanned_total = 0

    print("── Stage 1: Combine & Clean (pandas; per-file, union schema) ──")
    print(f"Target output columns: {len(output_cols)} columns")

    for f in excel_files:
        p = Path(f)
        if not p.exists():
            raise FileNotFoundError(f"Missing Excel file: {p}")
        print(f"Reading {p.name} ...")

        # Read full sheet (no usecols) so nothing is limited
        df = pd.read_excel(p, engine="openpyxl")
        scanned_total += len(df)

        # --- Early row filters ---
        # Language filter
        if "iso_language_code" in df.columns:
            before = len(df)
            df = df[df["iso_language_code"] == "en"]
            print(f"  Language filter (en): {before:,} → {len(df):,}")
        else:
            print("  iso_language_code not found; skipping language filter for this file.")

        # Geography filter (US) - NOTE: COULD BE TAKEN OUT WITH MORE CPU RESOURCES
        if FILTER_US_ONLY and "country" in df.columns:
            before = len(df)
            df = df[df["country"] == "United States"]
            print(f"  Country filter (US):  {before:,} → {len(df):,}")
        elif FILTER_US_ONLY:
            print("  country column not found; US filter skipped for this file.")

        # Required numeric non-NA
        req_present = [c for c in NUMERIC_REQUIRED if c in df.columns]
        if req_present:
            before = len(df)
            df = df.dropna(subset=req_present)
            if len(df) != before:
                print(f"  Drop NA in {req_present}: {before:,} → {len(df):,}")

        # --- Drop metadata columns (after filters) ---
        drop_now = [c for c in DROP_COLS if c in df.columns]
        if FILTER_US_ONLY and "country" in df.columns:
            drop_now.append("country")
        if drop_now:
            df = df.drop(columns=drop_now, errors="ignore")
            print(f"Dropped columns: {', '.join(sorted(set(drop_now)))}")

        # --- Align to union-based output schema ---
        missing = [c for c in output_cols if c not in df.columns]
        for c in missing:
            df[c] = pd.NA
        df = df[output_cols]  # stable order

        # --- Canonicalize dtypes uniformly across files ---
        df = _canonicalize_dtypes(df)

        # --- Date column normalization ---
        DATE_COL = "published_datetime"
        if DATE_COL in df.columns:
            print(f"  Parsing {DATE_COL} as datetime ...")
            df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce", utc=True)

        # Ensure headline/body exist (so Stage 2 doesn't blow up later)
        if HEADLINE_COL not in df.columns:
            df[HEADLINE_COL] = pd.Series(pd.array([None] * len(df), dtype="string[pyarrow]"))
        if BODY_COL not in df.columns:
            df[BODY_COL] = pd.Series(pd.array([None] * len(df), dtype="string[pyarrow]"))

        # --- Append to Parquet with locked schema ---
        table = pa.Table.from_pandas(df, preserve_index=False)

        if parquet_writer is None:
            target_schema = table.schema
            parquet_writer = pq.ParquetWriter(
                out_path,
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

    print(f"Stage 1 saved: {out_path} [{_sizeof(out_path)}]")
    print(f"Rows scanned (all files): {scanned_total:,}")
    print(f"Rows kept (post-filters): {kept_total:,}")

    return str(out_path), kept_total

############################
# Stage 2: Text Processing (STREAMING, LOW-MEM)
############################
def process_text(in_parquet: str, out_parquet: str) -> Tuple[str, int]:
    """
    Streaming implementation to avoid OOM:
    - Reads the combined parquet in batches via pyarrow.dataset
    - Normalizes categorical columns per batch
    - Cleans headline/article_body per batch (stopwords + lemmatization)
    - Appends to final parquet with a locked schema

    Returns (out_path, row_count).
    """
    import pyarrow.dataset as pds

    STOP = set(stopwords.words("english"))
    LEMM = WordNetLemmatizer()
    NONLETTERS = re.compile(r"[^a-z\s]")
    TOK = re.compile(r"\S+")

    UNKNOWN_SYNONYMS_NORM = {
        "uncredited", "unknown", "nan", "other", "na", "n/a", "none", "null",
        "unknown/other", "other_unknown", ""
    }

    def normalize_categorical(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        key = s.str.strip().str.lower()
        key = key.str.replace(r"\s*/\s*", "/", regex=True).str.replace(r"\s+", " ", regex=True)

        na_like = s.isna() | key.isin({"na", "n/a", "nan", "none", "null", ""})
        to_unknown = na_like | key.isin(UNKNOWN_SYNONYMS_NORM)

        norm = s[~to_unknown].str.lower().str.strip()
        norm = norm.str.replace(r"[^\w\s-]", "", regex=True)   # drop punctuation except _ and -
        norm = norm.str.replace(r"[\s-]+", "_", regex=True)    # spaces/dashes -> underscore
        norm = norm.str.strip("_")

        out = s.copy()
        out[to_unknown | (key == "other/unknown")] = "unknown"
        out[~to_unknown & (key != "other/unknown")] = norm
        out = out.fillna("")
        out = out.mask(out == "", "unknown")
        return out

    def clean_text_series(s: pd.Series) -> pd.Series:
        # Lower + keep letters/spaces, remove nonletters, drop stopwords, lemmatize
        s = s.fillna("").astype(str).str.lower()
        s = s.apply(lambda x: NONLETTERS.sub(" ", x))
        def _proc(x: str) -> str:
            toks = (t for t in x.split() if len(t) >= 3 and t not in STOP)
            return " ".join(LEMM.lemmatize(t) for t in toks)
        return s.apply(_proc)

    # Prepare output writer (overwrite if exists to avoid schema conflicts)
    out_path = Path(out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists():
        print(f"Removing existing file to avoid schema conflicts: {out_path}")
        out_path.unlink()

    # Build dataset and select just the columns we need to touch
    ds = pds.dataset(in_parquet, format="parquet")

    # --- add near other config ---
    DATE_COLS = {"published_datetime"}  # any other date cols you want to keep

    # inside process_text(...), after ds = pds.dataset(...):
    ds_cols = set(ds.schema.names)

    NUMERIC_PASS_THRU = {
        "circulation_size",
        "sentiment_score",
        "hit_strength",
        "vipr_weight",
        "vipr_score",
    }

    needed_cols = (
    set(CAT_TEXT_COLS)
    | {HEADLINE_COL, BODY_COL, "article_id"} 
    | NUMERIC_PASS_THRU
    | (DATE_COLS & ds_cols)
    )

    present_cols = sorted([c for c in needed_cols if c in ds_cols])

    if BODY_COL not in present_cols:
        raise KeyError(f"Missing required column: {BODY_COL}")
    if HEADLINE_COL not in present_cols:
        present_cols.append(HEADLINE_COL)  # we'll fill missing later if truly absent

    writer: pq.ParquetWriter | None = None
    target_schema: pa.Schema | None = None
    total_rows = 0
    batch_idx = 0

    print("Streaming Stage 2 in batches...")
    for batch in ds.to_batches(columns=present_cols, batch_size=STAGE2_BATCH_ROWS):
        batch_idx += 1
        pdf = batch.to_pandas(types_mapper=pd.ArrowDtype)

        # Keep article_id as integer if present
        if "article_id" in pdf.columns:
            pdf["article_id"] = pd.to_numeric(pdf["article_id"], errors="coerce").astype("Int64")

        # Ensure headline column exists
        if HEADLINE_COL not in pdf.columns:
            pdf[HEADLINE_COL] = ""

        # Categorical normalization
        for col in CAT_TEXT_COLS:
            if col in pdf.columns:
                pdf[col] = normalize_categorical(pdf[col])

        # Text cleaning (headline/body)
        pdf[HEADLINE_COL] = clean_text_series(pdf[HEADLINE_COL])
        pdf[BODY_COL]     = clean_text_series(pdf[BODY_COL])

        # Token counts
        pdf["headline_token_count"] = pdf[HEADLINE_COL].apply(lambda s: len(TOK.findall(s)))
        pdf["body_token_count"]     = pdf[BODY_COL].apply(lambda s: len(TOK.findall(s)))
        pdf["token_count"]          = (pdf["headline_token_count"] + pdf["body_token_count"]).astype("Int32")

        # Keep stable dtypes and Arrow-backed strings
        pdf = _canonicalize_dtypes(pdf)

        # Append to Parquet with locked schema
        table = pa.Table.from_pandas(pdf, preserve_index=False)

        if writer is None:
            target_schema = table.schema
            writer = pq.ParquetWriter(
                out_path,
                target_schema,
                compression="zstd",
                use_dictionary=True
            )
        else:
            if table.schema != target_schema:
                table = table.cast(target_schema)

        writer.write_table(table)
        rows_this = batch.num_rows  # or: len(pdf)
        total_rows += len(pdf)

        # Free memory after we’ve recorded counts
        del pdf, table, batch
        gc.collect()

        print(f"  Wrote batch {batch_idx:,} (+{rows_this} rows). Total: {total_rows:,}")

    if writer is not None:
        writer.close()

    print(f"Stage 2 saved: {out_path} [{_sizeof(out_path)}]")
    return str(out_path), total_rows

############################
# Main
############################
def main():
    Path(PROCESSED_DIR).mkdir(parents=True, exist_ok=True)
    Path(FINAL_DIR).mkdir(parents=True, exist_ok=True)

    print("── Stage 1: Combine & Clean (pandas) ──")
    combined_path, n1 = combine_and_clean(EXCEL_FILES, COMBINED_OUT)

    combined_path = COMBINED_OUT
    print("── Stage 2: Text Processing (headline & body) ──")
    final_path, n2 = process_text(combined_path, FINAL_OUT)

    print("\nDone.")
    print(f"Rows after Stage 1: {n1:,}")
    print(f"Rows after Stage 2: {n2:,}")
    print(f"Combined data: {combined_path}")
    print(f"Processed data: {final_path}")

if __name__ == "__main__":
    sys.exit(main())