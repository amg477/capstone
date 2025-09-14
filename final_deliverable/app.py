# app.py — Attribution Explorer (single view + selector)
from __future__ import annotations

import io, re
from typing import Tuple
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from azure.storage.blob import BlobServiceClient

# ---------------------------
# Page config
# ---------------------------
st.set_page_config(page_title="Attribution Explorer", layout="wide")
st.title("Attribution Explorer")

# ---------------------------
# Data loader (Azure, cached)
# ---------------------------
@st.cache_data(show_spinner=True)
def load_both() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load final dataset and attribution table from Azure Blob using secrets."""
    try:
        conn_str   = st.secrets["AZURE_STORAGE_CONNECTION_STRING"]
        container  = st.secrets["CONTAINER_NAME"]
        final_path = st.secrets["FINAL_DATA_BLOB_PATH"]
        attr_path  = st.secrets["ATTR_DATA_BLOB_PATH"]
    except KeyError as e:
        st.error(f"Missing secret: {e}. Add it in your Streamlit app Secrets.")
        st.stop()

    svc = BlobServiceClient.from_connection_string(conn_str)

    def _read_csv_from_blob(path: str) -> pd.DataFrame:
        blob = svc.get_blob_client(container=container, blob=path)
        buf = io.BytesIO()
        blob.download_blob().readinto(buf)
        buf.seek(0)
        return pd.read_csv(buf, low_memory=False)

    df   = _read_csv_from_blob(final_path)
    attr = _read_csv_from_blob(attr_path)

    # Parse dates safely
    if "load_date" in df.columns:
        df["load_date"] = pd.to_datetime(df["load_date"], errors="coerce", utc=True).dt.tz_localize(None)

    # Ensure expected attr columns exist
    for col in ["kind", "dimension", "value", "credit", "credit_share", "rating"]:
        if col not in attr.columns:
            attr[col] = np.nan

    # Provide processed text fallbacks for term search
    if "processed_headline" not in df.columns and "headline" in df.columns:
        df["processed_headline"] = df["headline"].astype(str)
    if "processed_body" not in df.columns and "article_body" in df.columns:
        df["processed_body"] = df["article_body"].astype(str)

    return df, attr

df, attr = load_both()

st.caption(
    f"Loaded data blobs: `{st.secrets['FINAL_DATA_BLOB_PATH']}` and `{st.secrets['ATTR_DATA_BLOB_PATH']}`"
)

# ---------------------------
# Prep attribution splits
# ---------------------------
ITEM_ATTR = attr.query("kind == 'item'").copy()
TERM_ATTR = attr.query("kind == 'term'").copy()
available_dims = sorted([d for d in ITEM_ATTR["dimension"].dropna().unique() if d in df.columns])

# ---------------------------
# Sidebar: global filters
# ---------------------------
st.sidebar.header("Global Filters")

# Date range
if "load_date" in df.columns and df["load_date"].notna().any():
    min_date = pd.to_datetime(df["load_date"]).min()
    max_date = pd.to_datetime(df["load_date"]).max()
    date_range = st.sidebar.date_input(
        "Load date range",
        value=(min_date.date(), max_date.date()) if pd.notna(min_date) and pd.notna(max_date) else ()
    )
else:
    date_range = ()

# Sentiment band
sent_bands = sorted(df["sentiment_band"].dropna().unique().tolist()) if "sentiment_band" in df.columns else []
sel_bands = st.sidebar.multiselect("Sentiment band", sent_bands, default=sent_bands)

# Publication filter
pubs = sorted(df["publication_name"].dropna().unique().tolist()) if "publication_name" in df.columns else []
sel_pubs = st.sidebar.multiselect("Publication (optional)", pubs, default=[])

# Thresholds
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
    min_value=50, max_value=50000, value=2000, step=50
)

def apply_global_filters(_df: pd.DataFrame) -> pd.DataFrame:
    out = _df.copy()

    # Date range
    if date_range and len(date_range) == 2 and "load_date" in out.columns:
        start, end = date_range
        if start and end:
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

def safe_text_series(df_in: pd.DataFrame, col: str) -> pd.Series:
    return df_in[col].fillna("").astype(str) if col in df_in.columns else pd.Series("", index=df_in.index, dtype=str)

def download_button_for_df(df_in: pd.DataFrame, label: str, fname: str):
    csv_bytes = df_in.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_bytes, file_name=fname, mime="text/csv")

# ---------------------------
# View selector
# ---------------------------
view = st.selectbox("View", ["Item Lookup", "Term Lookup", "Browse Attribution"], index=0)

# ===== Item Lookup =====
if view == "Item Lookup":
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

        if dim in df.columns:
            filtered = apply_global_filters(df[df[dim] == value])
            st.write(f"**Matching Articles** ({len(filtered):,} rows; showing up to {row_limit:,})")
            st.dataframe(filtered.head(int(row_limit)), use_container_width=True)

            fname = f"{dim}__{re.sub(r'[^A-Za-z0-9_-]+','_', str(value))}.csv"
            download_button_for_df(filtered, "⬇️ Download filtered rows (CSV)", fname)
        else:
            st.warning(f"`{dim}` not found in the final dataset columns.")

# ===== Term Lookup =====
elif view == "Term Lookup":
    st.subheader("Term Lookup (keywords & bigrams)")

    term_input = st.text_input("Type a term to search (exact or substring)", "")
    whole_word = st.checkbox(r"Whole word match (\bterm\b)", value=True)

    topN = st.number_input("Show top N terms by credit_share", min_value=10, max_value=2000, value=100, step=10)
    top_terms = (
        TERM_ATTR[["value", "credit", "credit_share", "rating"]]
        .drop_duplicates()
        .sort_values("credit_share", ascending=False)
        .head(int(topN))
    )
    st.write("**Top Terms by Credit Share**")
    st.dataframe(top_terms, use_container_width=True)

    if term_input:
        tscore = TERM_ATTR.query("value == @term_input")[["value", "credit", "credit_share", "rating"]]
        if not tscore.empty:
            st.write("**Term Attribution**")
            st.dataframe(tscore, use_container_width=True)

        work = apply_global_filters(df)
        text = safe_text_series(work, "processed_headline") + " " + safe_text_series(work, "processed_body")
        pattern = r"\b{}\b".format(re.escape(term_input)) if whole_word else re.escape(term_input)
        hits = work[text.str.contains(pattern, case=False, na=False, regex=True)]

        st.write(f"**Articles containing “{term_input}”** ({len(hits):,}; showing up to {row_limit:,})")
        st.dataframe(hits.head(int(row_limit)), use_container_width=True)

        fname = f"term__{re.sub(r'[^A-Za-z0-9_-]+','_', term_input)}.csv"
        download_button_for_df(hits, f"⬇️ Download rows with '{term_input}' (CSV)", fname)

# ===== Browse Attribution =====
else:
    st.subheader("Browse All Attribution")

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

    st.markdown("### Terms")
    terms_view = (
        TERM_ATTR[["value", "credit", "credit_share", "rating"]]
        .drop_duplicates()
        .sort_values("credit_share", ascending=False)
    )
    st.dataframe(terms_view.head(int(row_limit)), use_container_width=True)

# Footer
st.caption("Use the sidebar to filter by date & sentiment. Switch views via the selector above.")