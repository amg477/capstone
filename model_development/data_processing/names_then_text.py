#!/usr/bin/env python3
"""
names_then_text.py

Pipeline on the *sampled* dataset:
  Stage 1: PERSON extraction (spaCy) on sampled_data.parquet → sampled_with_people.parquet
  Stage 2: Text processing + categorical normalization on sampled_with_people.parquet
           → processed_with_people.parquet
  Stage 3: Emotion of article_body (cleaned) → processed_with_people_emotion.parquet

Guarantees:
- Stage 1 writes raw PERSON strings to people_by_row (comma-space separated, e.g., "Jane Doe, John Smith").
- Stage 2 adds people_by_row_clean (deduped, accent-folded, title-cased; comma-space separated).
- Set OVERWRITE_PEOPLE_NAMES=1 to overwrite people_by_row with the cleaned version during Stage 2.
"""

import os
import sys
import gc
import re
import hashlib
import unicodedata
from pathlib import Path
from typing import Tuple, Optional

import pandas as pd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

# ---------- NLTK: download only if missing ----------
import nltk
from nltk.data import find as nltk_find
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

def _ensure_nltk_pkg(pkg):
    try:
        nltk_find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_ensure_nltk_pkg("stopwords")
_ensure_nltk_pkg("wordnet")
_ensure_nltk_pkg("punkt")

# ---------- Config ----------
ROOT = Path(__file__).resolve().parents[2]  # .../capstone/capstone
PROCESSED_DIR = ROOT / "data_storage" / "processed_data"

SAMPLED_IN   = PROCESSED_DIR / "sampled_data.parquet"
PEOPLE_OUT   = PROCESSED_DIR / "sampled_with_people.parquet"
FINAL_OUT    = PROCESSED_DIR / "processed_with_people.parquet"
EMO_OUT      = PROCESSED_DIR / "processed_with_people_emotion.parquet"

HEADLINE_COL = "headline"
BODY_COL     = "article_body"

# Columns whose string values you want normalized (case/depunc/underscores)
CAT_TEXT_COLS = [
    "tag_name", "source_feed_name", "feed_name", "author_name", "source_type",
    "sentiment_band", "sub_region", "country", "channel_name", "channel",
    "publisher_name", "publication_name",
]

DATE_COLS_KEEP = {"published_datetime"}
NUMERIC_PASS_THRU = {"circulation_size", "sentiment_score", "hit_strength", "vipr_weight", "vipr_score"}

# ---- Stage 1 (NER) envs ----
SPACY_MODEL   = os.getenv("SPACY_MODEL", "en_core_web_sm")
SPACY_NPROC   = int(os.getenv("SPACY_NPROC", "4"))
SPACY_BATCH   = int(os.getenv("SPACY_BATCH_SIZE", "512"))
NER_SOURCE    = os.getenv("NER_SOURCE", "full")  # "headline" or "full"

# ---- Stage 2 (text) envs ----
USE_LEMMATIZATION = bool(int(os.getenv("USE_LEMMATIZATION", "0")))
STAGE2_BATCH_ROWS = int(os.getenv("STAGE2_BATCH_ROWS", "200000"))
OVERWRITE_PEOPLE  = bool(int(os.getenv("OVERWRITE_PEOPLE_NAMES", "0")))  # 1 to overwrite people_by_row with cleaned

# ---- Stage 3 (emotion) envs ----
# Default to a solid zero-shot emotion classifier (no sentencepiece dependency)
EMOTION_MODEL_NAME = os.getenv("EMOTION_MODEL", "j-hartmann/emotion-english-distilroberta-base")
EMOTION_BATCH_SIZE = int(os.getenv("EMOTION_BATCH_SIZE", "64"))
EMOTION_MAX_LEN    = int(os.getenv("EMOTION_MAX_LEN", "512"))      # tokenizer truncation length
EMOTION_COL_NAME   = os.getenv("EMOTION_COL_NAME", "emotion_body") # output column name

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

# ---------- Person-name cleaning helpers ----------
HONORIFICS = {
    "mr","mrs","ms","miss","mx","dr","prof","sir","dame","lord","lady",
    "president","pres","gov","sen","rep","amb","sec","chancellor","pm",
    "pres.","gov.","sen.","rep.","dr.","prof.","mr.","mrs.","ms.","mx."
}
TRAILING_SUFFIXES = {"jr","sr","ii","iii","iv","phd","md","esq","jr.","sr.","ph.d.","m.d."}
GENERIC_NONNAMES = {"anonymous","unknown","staff","editor","guest","author","reporter"}

def _strip_accents(s: str) -> str:
    if not s:
        return s
    return unicodedata.normalize("NFKD", s).encode("ASCII", "ignore").decode("ASCII")

def _titlecase_name(s: str) -> str:
    if not s:
        return s
    parts = s.split()
    lower_particles = {"de","del","da","dos","das","van","der","den","von","la","le","di","du","al","bin"}
    fixed = []
    for p in parts:
        if p.lower() in lower_particles:
            fixed.append(p.lower())
        else:
            sub = "-".join(seg.capitalize() for seg in p.split("-"))
            sub = "'".join(seg.capitalize() for seg in sub.split("'"))
            fixed.append(sub)
    return " ".join(fixed)

def _clean_person_name(raw: str) -> Optional[str]:
    """Lightweight canonicalizer for PERSON entity text."""
    if not isinstance(raw, str):
        return None
    s = raw.strip().strip("›«»“”\"'`·•")
    s = re.sub(r"\s+", " ", s)
    s = _strip_accents(s)
    # Remove leading/trailing punctuation
    s = re.sub(r"^[,.\-–—;:()$begin:math:display$$end:math:display${}]+|[,.\-–—;:()$begin:math:display$$end:math:display${}]+$", "", s).strip()
    if not s:
        return None

    toks = [t for t in s.split() if t]
    if not toks:
        return None

    # Drop honorifics at start
    while toks and toks[0].lower().rstrip(".") in HONORIFICS:
        toks = toks[1:]
    # Drop suffixes at end
    while toks and toks[-1].lower().rstrip(".") in TRAILING_SUFFIXES:
        toks = toks[:-1]
    if not toks:
        return None

    # Filter out single-letter tokens without dot
    kept = []
    for t in toks:
        tt = t.strip()
        if re.fullmatch(r"[A-Za-z]", tt):
            continue
        kept.append(tt)
    toks = kept
    if not toks:
        return None

    cand = " ".join(toks).strip()
    if cand.lower() in GENERIC_NONNAMES:
        return None

    cand = _titlecase_name(cand)
    if len(cand) < 2:
        return None
    return cand

def _clean_people_series(s: pd.Series, keep_delim: str = ", ") -> pd.Series:
    """
    Clean a comma-separated PERSON list per row and return the cleaned list,
    deduped (case-insensitive) and rejoined with ', '.
    """
    def _clean_row(val: str) -> str:
        if not isinstance(val, str) or not val.strip():
            return ""
        names = [n.strip() for n in val.split(",") if n.strip()]
        cleaned = []
        seen = set()
        for nm in names:
            c = _clean_person_name(nm)
            if not c:
                continue
            key = c.lower()
            if key not in seen:
                seen.add(key)
                cleaned.append(c)
        return keep_delim.join(cleaned)
    out = s.fillna("").astype(str).apply(_clean_row)
    return pd.Series(out, dtype="string[pyarrow]")

# ---------- Stage 1: PERSON extraction on sampled ----------
def _load_spacy(model: str):
    try:
        import spacy
    except ImportError:
        print("ERROR: spaCy not installed. `pip install spacy` and download a model.")
        sys.exit(1)
    try:
        nlp = spacy.load(model, disable=["parser","tagger","attribute_ruler","lemmatizer"])
        print(f"[nlp] Loaded spaCy model: {model}")
        return nlp
    except Exception as e:
        print(f"[nlp] Could not load '{model}': {e}")
        if model != "en_core_web_sm":
            print("[nlp] Falling back to en_core_web_sm...")
            try:
                nlp = spacy.load("en_core_web_sm", disable=["parser","tagger","attribute_ruler","lemmatizer"])
                print("[nlp] Loaded spaCy model: en_core_web_sm")
                return nlp
            except Exception as e2:
                print("[nlp] FATAL: cannot load spaCy model:", e2)
                sys.exit(1)
        sys.exit(1)

def stage1_people(input_path: Path, output_path: Path) -> Path:
    if not input_path.exists():
        print(f"ERROR: Missing input parquet {input_path}")
        sys.exit(1)

    print(f"[Stage 1: NER] load {input_path}")
    df = pd.read_parquet(input_path)
    nrows = len(df)
    print(f"[info] sampled rows: {nrows:,}")

    # Choose text source
    if NER_SOURCE == "headline":
        texts = df.get(HEADLINE_COL, pd.Series([""] * nrows)).fillna("").astype(str)
        print("[ner] Using HEADLINE ONLY.")
    else:
        head = df.get(HEADLINE_COL, pd.Series([""] * nrows)).fillna("").astype(str)
        body = df.get(BODY_COL, pd.Series([""] * nrows)).fillna("").astype(str)
        texts = (head + " " + body)
        print("[ner] Using FULL TEXT (headline + body).")

    nlp = _load_spacy(SPACY_MODEL)

    # Deduplicate identical texts to save compute
    hashes = texts.apply(lambda t: hashlib.md5(t.encode("utf-8", "ignore")).hexdigest())
    uniq_map = {}
    uniq_texts = []
    for idx, h in enumerate(hashes):
        if h not in uniq_map:
            uniq_map[h] = len(uniq_texts)
            uniq_texts.append(texts.iloc[idx])

    persons_by_uniq = [""] * len(uniq_texts)
    print(f"[ner] Unique texts: {len(uniq_texts):,} (of {nrows:,})")

    for start in range(0, len(uniq_texts), SPACY_BATCH):
        batch = uniq_texts[start:start+SPACY_BATCH]
        docs = nlp.pipe(batch, batch_size=SPACY_BATCH, n_process=SPACY_NPROC)
        for j, doc in enumerate(docs):
            names = []
            for ent in getattr(doc, "ents", []):
                if ent.label_ == "PERSON":
                    nm = ent.text.strip()  # preserve original casing
                    if nm and nm not in names:
                        names.append(nm)
            # EXACT delimiter: comma + space
            persons_by_uniq[start + j] = ", ".join(names)
        print(f"  processed {min(start+SPACY_BATCH, len(uniq_texts)):,} / {len(uniq_texts):,}")

    # Map back to all rows and write
    df["people_by_row"] = pd.Series([persons_by_uniq[uniq_map[h]] for h in hashes], dtype="string[pyarrow]")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Stage 1: NER] write -> {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"[ok] {output_path} [{_sizeof(output_path)}]")
    return output_path

# ---------- Stage 2: Text processing on the people-added set ----------
def stage2_text_processing(input_path: Path, output_path: Path) -> Tuple[Path, int]:
    """
    Streaming batch text cleaning + categorical normalization.
    - Preserves raw text to headline_raw/article_body_raw
    - Cleans headline/article_body
    - Adds token counts
    - Adds people_by_row_clean (cleaned, deduped, title-cased; comma-space separated)
    - Keeps people_by_row raw unless OVERWRITE_PEOPLE_NAMES=1
    """
    import pyarrow.dataset as pds

    STOP = set(stopwords.words("english"))
    NONLETTERS = re.compile(r"[^a-z\s]")
    TOK = re.compile(r"\S+")
    LEMM = WordNetLemmatizer() if USE_LEMMATIZATION else None

    def normalize_categorical(series: pd.Series) -> pd.Series:
        s = series.astype("string")
        key = s.str.strip().str.lower()
        key = key.str.replace(r"\s*/\s*", "/", regex=True).str.replace(r"\s+", " ", regex=True)
        UNKNOWN_SYNONYMS_NORM = {
            "uncredited", "unknown", "nan", "other", "na", "n/a", "none", "null",
            "unknown/other", "other_unknown", ""
        }
        na_like = s.isna() | key.isin({"na", "n/a", "nan", "none", "null", ""})
        to_unknown = na_like | key.isin(UNKNOWN_SYNONYMS_NORM)

        norm = s[~to_unknown].str.lower().str.strip()
        norm = norm.str.replace(r"[^\w\s-]", "", regex=True)
        norm = norm.str.replace(r"[\s-]+", "_", regex=True).str.strip("_")

        out = s.copy()
        out[to_unknown | (key == "other/unknown")] = "unknown"
        out[~to_unknown & (key != "other/unknown")] = norm
        out = out.fillna("").mask(out == "", "unknown")
        return out

    def clean_text_series(s: pd.Series) -> pd.Series:
        s = s.fillna("").astype(str).str.lower()
        s = s.str.replace(r"[^a-z\s]", " ", regex=True)
        if LEMM is None:
            return s.apply(lambda x: " ".join(t for t in x.split() if len(t) >= 3 and t not in STOP))
        else:
            return s.apply(lambda x: " ".join(LEMM.lemmatize(t) for t in x.split() if len(t) >= 3 and t not in STOP))

    # Prepare output writer
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        print(f"[Stage 2] Removing existing file to avoid schema conflicts: {output_path}")
        output_path.unlink()

    ds = pds.dataset(input_path, format="parquet")
    ds_cols = set(ds.schema.names)

    needed_cols = (
        set(CAT_TEXT_COLS)
        | {HEADLINE_COL, BODY_COL, "article_id", "people_by_row"}
        | (DATE_COLS_KEEP & ds_cols)
        | (NUMERIC_PASS_THRU & ds_cols)
    )
    present_cols = sorted([c for c in needed_cols if c in ds_cols])

    if BODY_COL not in present_cols:
        raise KeyError(f"Missing required column: {BODY_COL}")
    if HEADLINE_COL not in present_cols:
        present_cols.append(HEADLINE_COL)

    writer: Optional[pq.ParquetWriter] = None
    target_schema: Optional[pa.Schema] = None
    total_rows = 0
    batch_idx = 0

    print("[Stage 2] Streaming text processing...")
    for batch in ds.to_batches(columns=present_cols, batch_size=STAGE2_BATCH_ROWS):
        batch_idx += 1
        pdf = batch.to_pandas(types_mapper=pd.ArrowDtype)

        # Preserve raw text before cleaning
        pdf["headline_raw"] = pdf[HEADLINE_COL].astype("string[pyarrow]")
        pdf["article_body_raw"] = pdf[BODY_COL].astype("string[pyarrow]")

        # Normalize categoricals (people_by_row is intentionally NOT included)
        for col in CAT_TEXT_COLS:
            if col in pdf.columns:
                pdf[col] = normalize_categorical(pdf[col])

        # --- People names: cleaned companion column + optional overwrite ---
        if "people_by_row" in pdf.columns:
            pdf["people_by_row"] = pdf["people_by_row"].astype("string[pyarrow]")
            pdf["people_by_row_clean"] = _clean_people_series(pdf["people_by_row"])
            if OVERWRITE_PEOPLE:
                pdf["people_by_row"] = pdf["people_by_row_clean"]

        # Clean text
        pdf[HEADLINE_COL] = clean_text_series(pdf[HEADLINE_COL])
        pdf[BODY_COL]     = clean_text_series(pdf[BODY_COL])

        # Token counts
        pdf["headline_token_count"] = pdf[HEADLINE_COL].apply(lambda s: len(re.findall(r"\S+", s))).astype("Int32")
        pdf["body_token_count"]     = pdf[BODY_COL].apply(lambda s: len(re.findall(r"\S+", s))).astype("Int32")
        pdf["token_count"]          = (pdf["headline_token_count"] + pdf["body_token_count"]).astype("Int32")

        # Dtypes
        pdf = _canonicalize_dtypes(pdf)

        table = pa.Table.from_pandas(pdf, preserve_index=False)
        if writer is None:
            target_schema = table.schema
            writer = pq.ParquetWriter(
                output_path,
                target_schema,
                compression="zstd",
                use_dictionary=True
            )
        else:
            if table.schema != target_schema:
                table = table.cast(target_schema)
        writer.write_table(table)

        total_rows += len(pdf)
        del pdf, table, batch
        gc.collect()
        print(f"  wrote batch {batch_idx:,} — total {total_rows:,}")

    if writer is not None:
        writer.close()

    print(f"[Stage 2] saved: {output_path} [{_sizeof(output_path)}]")
    return output_path, total_rows

# ---------- Stage 3: Emotion of article_body (cleaned) ----------
def _load_emotion_pipeline():
    try:
        from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    except Exception as e:
        print("[emotion] transformers not available:", e)
        return None

    try:
        tok = AutoTokenizer.from_pretrained(EMOTION_MODEL_NAME)
        model = AutoModelForSequenceClassification.from_pretrained(EMOTION_MODEL_NAME)
        clf = pipeline(
            "text-classification",
            model=model,
            tokenizer=tok,
            top_k=None,            # return all scores (we'll argmax)
            truncation=True
        )
        print(f"[emotion] Loaded model: {EMOTION_MODEL_NAME}")
        return clf
    except Exception as e:
        print(f"[emotion] failed to load '{EMOTION_MODEL_NAME}':", e)
        return None

def _predict_emotion_batch(clf, texts: list[str]) -> list[str | None]:
    """
    Run classifier pipeline over a list of texts and return label strings (argmax).
    Returns None if classifier missing or text empty.
    """
    if clf is None:
        return [None] * len(texts)
    # Hugging Face pipeline returns list[ list[ {label, score}, ... ] ]
    out = []
    for i in range(0, len(texts), EMOTION_BATCH_SIZE):
        batch = texts[i:i+EMOTION_BATCH_SIZE]
        # Normalize empty
        inputs = [t if (t and isinstance(t, str)) else "" for t in batch]
        preds = clf(inputs, truncation=True, max_length=EMOTION_MAX_LEN)
        for res in preds:
            if isinstance(res, list) and len(res):
                best = max(res, key=lambda x: x.get("score", 0.0))
                out.append(best.get("label"))
            elif isinstance(res, dict) and "label" in res:
                out.append(res["label"])
            else:
                out.append(None)
    return out

def stage3_emotion(input_path: Path, output_path: Path) -> Path:
    """
    Appends EMOTION_COL_NAME for the cleaned article_body to the processed dataset.
    Reads the entire file once; if it's huge, feel free to convert to batched Arrow reads later.
    """
    print(f"[Stage 3] load {input_path}")
    if not input_path.exists():
        print(f"ERROR: {input_path} not found.")
        sys.exit(1)

    df = pd.read_parquet(input_path)
    print(f"[Stage 3] rows: {len(df):,}")

    if BODY_COL not in df.columns:
        raise KeyError(f"{BODY_COL} not found in {input_path.name}")

    clf = _load_emotion_pipeline()
    if clf is None:
        print("[emotion] Skipping (transformers/model not available). Filling with None.")
        df[EMOTION_COL_NAME] = pd.Series([None] * len(df), dtype="object")
    else:
        texts = df[BODY_COL].fillna("").astype(str).tolist()
        preds = _predict_emotion_batch(clf, texts)
        df[EMOTION_COL_NAME] = preds

    # Ensure people_by_row still proper dtype
    if "people_by_row" in df.columns:
        df["people_by_row"] = df["people_by_row"].astype("string[pyarrow]")
    if "people_by_row_clean" in df.columns:
        df["people_by_row_clean"] = df["people_by_row_clean"].astype("string[pyarrow]")

    # Write out
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[Stage 3] write -> {output_path}")
    df.to_parquet(output_path, index=False)
    print(f"[ok] {output_path} [{_sizeof(output_path)}]")
    return output_path

# ---------- Main ----------
def main():
    # Stage 1: NER on sampled set
    people_path = stage1_people(SAMPLED_IN, PEOPLE_OUT)

    # Stage 2: Text processing on people-added set
    final_path, n = stage2_text_processing(people_path, FINAL_OUT)

    # Stage 3: Emotion on cleaned article_body
    emo_path = stage3_emotion(final_path, EMO_OUT)

    print("\nDone.")
    print(f"People-added file:  {people_path}")
    print(f"Processed file:     {final_path} (rows: {n:,})")
    print(f"With emotion file:  {emo_path}")

if __name__ == "__main__":
    sys.exit(main())