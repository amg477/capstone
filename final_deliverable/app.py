# app.py — Attribution Explorer (single view + selector)
# ------------------------------------------------------
# What this app does
# - Loads:
#     /Users/annaglass/capstone/capstone/data/final_model_dataset.csv
#     /Users/annaglass/capstone/capstone/data/attribution_all_scored.csv
# - Lets you (via a single view selector):
#     • Item Lookup — influence by any item dimension/value
#     • Term Lookup — search keywords/bigrams + see hits
#     • Browse — scan attribution tables
# - Global sidebar filters apply everywhere (date, sentiment, etc.)
#
# Notes
# - Read-only: does NOT mutate files.
# - Built for fast exploration on large (~300K+) datasets.

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import streamlit as st

# ---------- Optional secret (unused but kept for future use) ----------
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")

# ---------- Paths ----------
ROOT = Path("/Users/annaglass/capstone/capstone")
DATA_FP = ROOT / "data" / "final_model_dataset.csv"
ATTR_FP = ROOT / "data" / "attribution_all_scored.csv"

# ---------- Page config ----------
st.set_page_config(page_title="Attribution Explorer", layout="wide")
st.title("Attribution Explorer")

# ---------- Loaders (cached) ----------
@st.cache_data(show_spinner=True)
def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(DATA_FP, low_memory=False)

    # Parse dates safely
    if "load_date" in df.columns:
        df["load_date"] = pd.to_datetime(df["load_date"], errors="coerce", utc=True).dt.tz_localize(None)
    else:
        df["load_date"] = pd.NaT

    attr = pd.read_csv(ATTR_FP, low_memory=False)

    # Normalize expected columns
    for col in ["kind", "dimension", "value", "credit", "credit_share", "rating"]:
        if col not in attr.columns:
            attr[col] = np.nan

    return df, attr

df, attr = load_data()

# Split attribution tables
ITEM_ATTR = attr.query("kind == 'item'").copy()
TERM_ATTR = attr.query("kind == 'term'").copy()

# Available item dimensions present in both (attr + df)
available_dims = sorted([d for d in ITEM_ATTR["dimension"].dropna().unique() if d in df.columns])

# ---------- Sidebar: global filters ----------
st.sidebar.header("Global Filters")

# Date range
if df["load_date"].notna().any():
    min_date = pd.to_datetime(df["load_date"]).min()
    max_date = pd.to_datetime(df["load_date"]).max()
    date_range = st.sidebar.date_input(
        "Load date range",
        value=(
            min_date.date() if pd.notna(min_date) else None,
            max_date.date() if pd.notna(max_date) else None,
        )
    )
else:
    date_range = None

# Sentiment band filter
sent_bands = sorted(df["sentiment_band"].dropna().unique().tolist()) if "sentiment_band" in df.columns else []
sel_bands = st.sidebar.multiselect("Sentiment band", sent_bands, default=sent_bands)

# Publication quick filter
pubs = sorted(df["publication_name"].dropna().unique().tolist()) if "publication_name" in df.columns else []
sel_pubs = st.sidebar.multiselect("Publication (optional)", pubs, default=[])

# Min influence thresholds (from model table if present on df)
min_pub_credit = st.sidebar.slider(
    "Min pub_credit_share",
    0.0,
    float(df.get("pub_credit_share", pd.Series([0.0])).max() or 0.0),
    0.0,
    0.01,
)
min_term_credit = st.sidebar.slider(
    "Min max_term_credit",
    0.0,
    float(df.get("max_term_credit", pd.Series([0.0])).max() or 0.0),
    0.0,
    0.01,
)

# Row limit
row_limit = st.sidebar.number_input(
    "Rows to display (for speed)",
    min_value=50,
    max_value=50000,
    value=2000,
    step=50
)

def apply_global_filters(_df: pd.DataFrame) -> pd.DataFrame:
    out = _df.copy()
    # Date range
    if date_range and len(date_range) == 2:
        start, end = date_range
        if start and end and pd.notna(start) and pd.notna(end) and "load_date" in out.columns:
            # Ensure datetime dtype
            if not np.issubdtype(out["load_date"].dtype, np.datetime64):
                out["load_date"] = pd.to_datetime(out["load_date"], errors="coerce")
            mask = (out["load_date"].dt.date >= start) & (out["load_date"].dt.date <= end)
            out = out[mask]

    # Sentiment
    if sel_bands and "sentiment_band" in out.columns:
        out = out[out["sentiment_band"].isin(sel_bands)]

    # Publications
    if sel_pubs and "publication_name" in out.columns:
        out = out[out["publication_name"].isin(sel_pubs)]

    # Numeric thresholds
    if "pub_credit_share" in out.columns:
        out = out[out["pub_credit_share"] >= min_pub_credit]
    if "max_term_credit" in out.columns:
        out = out[out["max_term_credit"] >= min_term_credit]

    return out

# ---------- Helpers ----------
def safe_text_series(df_in: pd.DataFrame, col: str) -> pd.Series:
    """Return a safe string Series for a text column (empty string if missing)."""
    if col in df_in.columns:
        return df_in[col].fillna("").astype(str)
    return pd.Series("", index=df_in.index, dtype=str)

def download_button_for_df(df_in: pd.DataFrame, label: str, fname: str):
    csv_bytes = df_in.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_bytes, file_name=fname, mime="text/csv")

# ---------- View selector (replaces tabs) ----------
view = st.selectbox(
    "View",
    ["🔎 Item Lookup", "🔎 Term Lookup", "📚 Browse Attribution"],
    index=0
)

# ==========================
# VIEW — ITEM LOOKUP
# ==========================
if view == "🔎 Item Lookup":
    st.subheader("Item Lookup (dimensions & values)")

    if not available_dims:
        st.info("No item dimensions available for lookup.")
    else:
        col1, col2 = st.columns([1, 2], vertical_alignment="top")

        with col1:
            default_idx = available_dims.index("publication_name") if "publication_name" in available_dims else 0
            dim = st.selectbox("Dimension", available_dims, index=default_idx)

            dim_values = sorted(
                ITEM_ATTR.query("dimension == @dim")["value"].dropna().unique().tolist()
            )
            value = st.selectbox("Value", dim_values)

        with col2:
            score_df = (
                ITEM_ATTR.query("dimension == @dim and value == @value")
                [["dimension", "value", "credit", "credit_share", "rating"]]
                .sort_values("credit_share", ascending=False)
            )
            st.write("**Attribution Score**")
            st.dataframe(score_df, use_container_width=True)

        # Matching articles
        if dim in df.columns:
            filtered = apply_global_filters(df[df[dim] == value])
            st.write(f"**Matching Articles** ({len(filtered):,} rows; showing up to {row_limit:,})")
            st.dataframe(filtered.head(int(row_limit)), use_container_width=True)

            fname = f"{dim}__{re.sub(r'[^A-Za-z0-9_-]+','_', str(value))}.csv"
            download_button_for_df(filtered, "⬇️ Download filtered rows (CSV)", fname)
        else:
            st.warning(f"`{dim}` not found in the final dataset columns.")

# ==========================
# VIEW — TERM LOOKUP
# ==========================
elif view == "🔎 Term Lookup":
    st.subheader("Term Lookup (keywords & bigrams)")

    # Controls
    term_input = st.text_input("Type a term to search (exact or substring)", "")
    whole_word = st.checkbox(r"Whole word match (\bterm\b)", value=True)

    # Top terms table
    topN = st.number_input(
        "Show top N terms by credit_share",
        min_value=10, max_value=2000, value=100, step=10
    )
    top_terms = (
        TERM_ATTR[["value", "credit", "credit_share", "rating"]]
        .drop_duplicates()
        .sort_values("credit_share", ascending=False)
        .head(int(topN))
    )
    st.write("**Top Terms by Credit Share**")
    st.dataframe(top_terms, use_container_width=True)

    # Selected term details + article hits
    if term_input:
        tscore = TERM_ATTR.query("value == @term_input")[["value", "credit", "credit_share", "rating"]]
        if tscore.empty:
            st.info("Term not found in attribution table; showing substring matches in articles only.")
        else:
            st.write("**Term Attribution**")
            st.dataframe(tscore, use_container_width=True)

        work = apply_global_filters(df)
        text = (
            safe_text_series(work, "processed_headline") + " " +
            safe_text_series(work, "processed_body")
        )

        pattern = r"\b{}\b".format(re.escape(term_input)) if whole_word else re.escape(term_input)
        rx = re.compile(pattern, flags=re.IGNORECASE)

        hits = work[text.str.contains(rx, na=False)]
        st.write(f"**Articles containing “{term_input}”** ({len(hits):,}; showing up to {row_limit:,})")
        st.dataframe(hits.head(int(row_limit)), use_container_width=True)

        fname = f"term__{re.sub(r'[^A-Za-z0-9_-]+','_', term_input)}.csv"
        download_button_for_df(hits, f"⬇️ Download rows with '{term_input}' (CSV)", fname)

# ==========================
# VIEW — BROWSE ATTRIBUTION
# ==========================
else:
    st.subheader("Browse All Attribution")

    # Items
    st.markdown("### Items (dimensions/values)")
    if available_dims:
        dim_browse = st.selectbox("Dimension to browse", available_dims)
        items_view = (
            ITEM_ATTR.query("dimension == @dim_browse")[["dimension", "value", "credit", "credit_share", "rating"]]
            .sort_values(["credit_share", "value"], ascending=[False, True])
        )
        st.dataframe(items_view.head(int(row_limit)), use_container_width=True)
    else:
        st.info("No item dimensions available for browsing.")

    # Terms
    st.markdown("### Terms")
    terms_view = (
        TERM_ATTR[["value", "credit", "credit_share", "rating"]]
        .drop_duplicates()
        .sort_values("credit_share", ascending=False)
    )
    st.dataframe(terms_view.head(int(row_limit)), use_container_width=True)

# ---------- Footer ----------
st.caption(
    "Tip: Use the sidebar to filter by date & sentiment. "
    "Switch the view selector to move between Item Lookup, Term Lookup, and Browse."
)