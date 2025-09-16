# app.py — Attribution Explorer (single view + selector, DuckDB backend)
# ---------------------------------------------------------------------
# What this app does
# - Loads (LOCAL or AZURE):
#     final_model_dataset.(parquet|csv)  -> DuckDB view: v
#     attribution_all_scored.csv         -> DuckDB view: v_attr (and v_item_attr, v_term_attr)
# - Lets you (via a single view selector):
#     • Item Lookup — influence by any item dimension/value
#     • Term Lookup — search keywords/bigrams + see hits
#     • Browse — scan attribution tables
# - Global sidebar filters apply everywhere (date, sentiment, pubs, thresholds)
# - All filtering/search/limits run in DuckDB (fast on 300k+)
#
# Notes
# - Read-only: does NOT mutate files.
# - Works with LOCAL paths by default; switches to AZURE if AZURE_CONTAINER is set.
# - For Azure, you can use secrets.toml or environment variables.
# ---------------------------------------------------------------------

from __future__ import annotations

import os
import re
import pathlib
import datetime as dt
from typing import List, Tuple, Optional

import streamlit as st
import pandas as pd
import duckdb

# Optional Azure imports (only used if AZURE mode)
try:
    from azure.storage.blob import BlobServiceClient  # type: ignore
    _has_azure = True
except Exception:
    _has_azure = False

# ---------- Page config ----------
st.set_page_config(page_title="Attribution Explorer", layout="wide")
st.title("Attribution Explorer")

st.markdown(
    """
    **Instructions**

    1. **Choose a view** from the dropdown at the top:
       * *Item Lookup* – See influence scores by any item dimension and value.
       * *Term Lookup* – Search keywords or bigrams and view matching articles.
       * *Browse Attribution* – Explore all attribution data.

    2. **Apply global filters** in the sidebar (date range, sentiment band,
       publications, and minimum influence thresholds).  
       These filters affect every view.

    3. **Inspect and download** the results. Use the download buttons in each
       section to export the filtered rows as CSV.
    """
)

# ---------- Paths (LOCAL defaults like your original app) ----------
ROOT = pathlib.Path("/Users/annaglass/capstone/capstone")
LOCAL_PARQUET = ROOT / "data" / "final_model_dataset.parquet"
LOCAL_CSV     = ROOT / "data" / "final_model_dataset.csv"
LOCAL_ATTR    = ROOT / "data" / "attribution_all_scored.csv"

TMP_PARQUET = pathlib.Path("/tmp/final_model_dataset.parquet")
TMP_ATTR    = pathlib.Path("/tmp/attribution_all_scored.csv")

# ---------- Secrets helpers (safe if secrets.toml is absent) ----------
def _get_secret_safe(key: str):
    try:
        return st.secrets.get(key)
    except Exception:
        return None

def _sql_str(path_like) -> str:
    """Return a single-quoted SQL string literal with quotes escaped."""
    s = str(path_like)
    return "'" + s.replace("'", "''") + "'"

def _quote_ident(col: str) -> str:
    """Return a DuckDB-safe double-quoted identifier."""
    return '"' + col.replace('"', '""') + '"'

# ---------- Determine mode (LOCAL vs AZURE) ----------
AZURE_CONTAINER = _get_secret_safe("AZURE_CONTAINER") or os.getenv("AZURE_CONTAINER")
USE_AZURE = bool(AZURE_CONTAINER)

# ---------- Azure helpers (only used if USE_AZURE) ----------
def _resolve_blob_service() -> "BlobServiceClient":
    conn_str = _get_secret_safe("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    acct_url = _get_secret_safe("AZURE_ACCOUNT_URL") or os.getenv("AZURE_ACCOUNT_URL")
    sas      = _get_secret_safe("AZURE_SAS") or os.getenv("AZURE_SAS")
    key      = _get_secret_safe("AZURE_STORAGE_KEY") or os.getenv("AZURE_STORAGE_KEY")

    if conn_str and "AccountName=" in conn_str:
        return BlobServiceClient.from_connection_string(conn_str)
    if acct_url and (sas or key):
        cred = (sas or "").lstrip("?") or key
        return BlobServiceClient(account_url=acct_url, credential=cred)
    raise RuntimeError("Azure credentials missing: set connection string OR (account url + SAS/key).")

def _blob_to_tmp(container: str, blob: str, dest: pathlib.Path) -> pathlib.Path:
    svc = _resolve_blob_service()
    bc = svc.get_container_client(container).get_blob_client(blob)
    if not bc.exists():
        raise FileNotFoundError(f"Blob not found: {container}/{blob}")
    data = bc.download_blob().readall()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    return dest

# ---------- Bootstrap data to local files (LOCAL or AZURE) ----------
@st.cache_resource(show_spinner=True)
def materialize_inputs() -> Tuple[pathlib.Path, Optional[pathlib.Path]]:
    """
    Returns (parquet_path, attr_csv_path).
    - If Parquet exists locally, use it; else convert local CSV -> Parquet.
    - In AZURE mode, download blobs to /tmp and prefer Parquet if present in blob.
    """
    if USE_AZURE:
        if not _has_azure:
            raise RuntimeError("azure-storage-blob not installed but AZURE mode requested.")
        container = AZURE_CONTAINER
        # Expected blob names (match your previous structure)
        PARQUET_BLOB = "data/processed/final_model_dataset.parquet"
        CSV_BLOB     = "data/processed/final_model_dataset.csv"
        ATTR_BLOB    = "data/processed/attribution_all_scored.csv"

        # Try Parquet first
        try:
            parquet_path = _blob_to_tmp(container, PARQUET_BLOB, TMP_PARQUET)
        except Exception:
            # Fallback: CSV -> Parquet one-time
            csv_path = _blob_to_tmp(container, CSV_BLOB, TMP_PARQUET.with_suffix(".csv"))
            # Convert CSV to Parquet using DuckDB (streaming)
            con = duckdb.connect()
            con.execute(f"COPY (SELECT * FROM read_csv_auto({_sql_str(csv_path)})) TO {_sql_str(TMP_PARQUET)} (FORMAT PARQUET)")
            parquet_path = TMP_PARQUET

        # Attribution (optional)
        try:
            attr_path = _blob_to_tmp(container, ATTR_BLOB, TMP_ATTR)
        except Exception:
            attr_path = None

        return parquet_path, attr_path

    # LOCAL mode
    if LOCAL_PARQUET.exists():
        parquet_path = LOCAL_PARQUET
    elif LOCAL_CSV.exists():
        # Convert once to /tmp to avoid creating big local files in your repo
        con = duckdb.connect()
        con.execute(f"COPY (SELECT * FROM read_csv_auto({_sql_str(LOCAL_CSV)})) TO {_sql_str(TMP_PARQUET)} (FORMAT PARQUET)")
        parquet_path = TMP_PARQUET
    else:
        raise FileNotFoundError(f"Could not find local data at:\n- {LOCAL_PARQUET}\n- {LOCAL_CSV}")

    attr_path = LOCAL_ATTR if LOCAL_ATTR.exists() else None
    return parquet_path, attr_path

# ---------- DuckDB connection & views ----------
@st.cache_resource(show_spinner=False)
def get_duck_conn(parquet_path: pathlib.Path, attr_path: Optional[pathlib.Path]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    # 1) Create a raw view over the file
    con.execute(f"""
        CREATE OR REPLACE VIEW v_raw AS
        SELECT * FROM read_parquet({_sql_str(parquet_path)}, hive_partitioning=FALSE)
    """)

    # 2) Inspect columns to decide how to build load_ts
    cols_df = con.execute("DESCRIBE v_raw").fetchdf()
    has_load_date = "load_date" in set(cols_df["column_name"].tolist())

    if has_load_date:
        con.execute("""
            CREATE OR REPLACE VIEW v AS
            SELECT
                *,
                TRY_CAST(load_date AS TIMESTAMP) AS load_ts
            FROM v_raw
        """)
    else:
        # No load_date column -> still create v and provide a NULL timestamp so the app runs
        con.execute("""
            CREATE OR REPLACE VIEW v AS
            SELECT
                *,
                CAST(NULL AS TIMESTAMP) AS load_ts
            FROM v_raw
        """)

    # Attribution views (optional)
    if attr_path and attr_path.exists():
        con.execute(f"""
            CREATE OR REPLACE VIEW v_attr AS
            SELECT * FROM read_csv_auto({_sql_str(attr_path)}, IGNORE_ERRORS=TRUE)
        """)
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM v_attr WHERE kind = 'item'")
        con.execute("CREATE OR REPLACE VIEW v_term_attr AS SELECT * FROM v_attr WHERE kind = 'term'")
    else:
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT * FROM (SELECT 1 WHERE 0)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM (SELECT 1 WHERE 0)")
        con.execute("CREATE OR REPLACE VIEW v_term_attr AS SELECT * FROM (SELECT 1 WHERE 0)")

    return con 

# ---------- Sidebar: global filters ----------
st.sidebar.header("Global Filters")

# materialize inputs and connect
try:
    parquet_path, attr_path = materialize_inputs()
    con = get_duck_conn(parquet_path, attr_path)
except Exception as e:
    st.error(f"Data bootstrap failed: {e}")
    st.stop()

# --- Debug info (appears collapsed in the app) ---
with st.expander("Debug (data bootstrap)", expanded=False):
    st.write({
        "mode": "AZURE" if USE_AZURE else "LOCAL",
        "parquet_path": str(parquet_path),
        "attr_path": str(attr_path) if attr_path else None
    })
    st.write("Columns in v:")
    st.dataframe(con.execute("DESCRIBE v").fetchdf(), use_container_width=True)

# read summary stats
n_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
min_max = con.execute("SELECT MIN(load_ts), MAX(load_ts) FROM v").fetchone()
min_d = min_max[0].date() if min_max[0] is not None else None
max_d = min_max[1].date() if min_max[1] is not None else None

# Date range
if min_d and max_d:
    date_range = st.sidebar.date_input("Load date range", value=(min_d, max_d))
else:
    date_range = None

# Sentiment bands
sent_bands = con.execute("SELECT DISTINCT sentiment_band FROM v WHERE sentiment_band IS NOT NULL ORDER BY 1").fetchdf()
sel_bands = st.sidebar.multiselect("Sentiment band", sent_bands["sentiment_band"].tolist(), default=sent_bands["sentiment_band"].tolist())

# Publications
pubs = con.execute("SELECT DISTINCT publication_name FROM v WHERE publication_name IS NOT NULL ORDER BY 1").fetchdf()
sel_pubs = st.sidebar.multiselect("Publication (optional)", pubs["publication_name"].tolist(), default=[])

# Threshold sliders (read maxes safely)
def _max_or_zero(sql: str) -> float:
    val = con.execute(sql).fetchone()[0]
    try:
        return float(val or 0.0)
    except Exception:
        return 0.0

max_pub_credit  = _max_or_zero("SELECT MAX(pub_credit_share)  FROM v")
max_term_credit = _max_or_zero("SELECT MAX(max_term_credit)   FROM v")

min_pub_credit = st.sidebar.slider("Min pub_credit_share", 0.0, max(0.0, float(max_pub_credit)), 0.0, 0.01)
min_term_credit = st.sidebar.slider("Min max_term_credit", 0.0, max(0.0, float(max_term_credit)), 0.0, 0.01)

# Row limit
row_limit = st.sidebar.number_input("Rows to display (for speed)", min_value=50, max_value=50000, value=2000, step=50)

# ---------- Build WHERE clause + args from global filters ----------
def build_where_and_args(extra: str = "", extra_args: dict | None = None) -> Tuple[str, dict]:
    clauses = ["1=1"]
    args: dict = {}

    # dates
    if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and date_range[0] and date_range[1]:
        dmin = pd.to_datetime(date_range[0])
        dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # exclusive upper bound
        clauses.append("load_ts >= $dmin AND load_ts < $dmax")
        args["dmin"] = dmin
        args["dmax"] = dmax

    # sentiment
    if sel_bands:
        clauses.append("sentiment_band IN $bands")
        args["bands"] = sel_bands

    # publications
    if sel_pubs:
        clauses.append("publication_name IN $pubs")
        args["pubs"] = sel_pubs

    # numeric thresholds
    clauses.append("COALESCE(pub_credit_share, 0.0) >= $thr_pub")
    args["thr_pub"] = float(min_pub_credit)
    clauses.append("COALESCE(max_term_credit, 0.0) >= $thr_term")
    args["thr_term"] = float(min_term_credit)

    if extra:
        clauses.append(extra)
    if extra_args:
        args.update(extra_args)

    return " AND ".join(clauses), args

# ---------- Helpers ----------
def download_button_for_df(df_in: pd.DataFrame, label: str, fname: str):
    csv_bytes = df_in.to_csv(index=False).encode("utf-8")
    st.download_button(label, csv_bytes, file_name=fname, mime="text/csv")

def get_v_columns(con: duckdb.DuckDBPyConnection) -> List[str]:
    cols = con.execute("DESCRIBE v").fetchdf()["column_name"].tolist()
    return [c for c in cols if c not in ("load_ts",)]  # hide computed helper

# ---------- Available item dimensions present in both (attr + df) ----------
# from attribution (distinct dimensions)
dims_df = con.execute("SELECT DISTINCT dimension FROM v_item_attr WHERE dimension IS NOT NULL").fetchdf()
dim_candidates = set(dims_df["dimension"].tolist())
df_cols = set(get_v_columns(con))
available_dims = sorted([d for d in dim_candidates if d in df_cols])

# ---------- View selector ----------
view = st.selectbox("View", ["🔎 Item Lookup", "🔎 Term Lookup", "📚 Browse Attribution"], index=0)

# ==========================
# VIEW — ITEM LOOKUP
# ==========================
if view == "🔎 Item Lookup":
    st.subheader("Item Lookup (dimensions & values)")

    if not available_dims:
        st.info("No item dimensions available for lookup.")
    else:
        col1, col2 = st.columns([1, 2])

        with col1:
            default_idx = available_dims.index("publication_name") if "publication_name" in available_dims else 0
            dim = st.selectbox("Dimension", available_dims, index=default_idx)

            values_df = con.execute(
                "SELECT DISTINCT value FROM v_item_attr WHERE dimension = $dim AND value IS NOT NULL ORDER BY 1",
                {"dim": dim},
            ).fetchdf()
            dim_values = values_df["value"].tolist()
            value = st.selectbox("Value", dim_values)

        with col2:
            score_df = con.execute(
                """
                SELECT dimension, value, credit, credit_share, rating
                FROM v_item_attr
                WHERE dimension = $dim AND value = $val
                ORDER BY credit_share DESC
                """,
                {"dim": dim, "val": value},
            ).fetchdf()
            st.write("**Attribution Score**")
            st.dataframe(score_df, use_container_width=True)

        # Matching articles in v with global filters AND dim=value
        where, args = build_where_and_args(extra=f"{_quote_ident(dim)} = $val", extra_args={"val": value})
        sql = f"SELECT * FROM v WHERE {where} LIMIT $lim"
        args["lim"] = int(row_limit)
        filtered = con.execute(sql, args).fetchdf()

        st.write(f"**Matching Articles** ({len(filtered):,} rows shown; cap = {row_limit:,})")
        st.dataframe(filtered, use_container_width=True)

        safe_val = re.sub(r"[^A-Za-z0-9_-]+", "_", str(value))
        fname = f"{dim}__{safe_val}.csv"
        download_button_for_df(filtered, "⬇️ Download filtered rows (CSV)", fname)

# ==========================
# VIEW — TERM LOOKUP
# ==========================
elif view == "🔎 Term Lookup":
    st.subheader("Term Lookup (keywords & bigrams)")

    # Controls
    term_input = st.text_input("Type a term to search (exact or substring)", "")
    whole_word = st.checkbox("Whole word match (uses regex)", value=True)

    # Top terms table
    topN = st.number_input("Show top N terms by credit_share", min_value=10, max_value=2000, value=100, step=10)
    top_terms = con.execute(
        """
        SELECT value, credit, credit_share, rating
        FROM v_term_attr
        QUALIFY ROW_NUMBER() OVER (PARTITION BY value ORDER BY credit_share DESC) = 1
        ORDER BY credit_share DESC
        LIMIT $n
        """,
        {"n": int(topN)},
    ).fetchdf()
    st.write("**Top Terms by Credit Share**")
    st.dataframe(top_terms, use_container_width=True)

    # Selected term details + article hits
    if term_input:
        tscore = con.execute(
            "SELECT value, credit, credit_share, rating FROM v_term_attr WHERE value = $val ORDER BY credit_share DESC",
            {"val": term_input},
        ).fetchdf()
        if tscore.empty:
            st.info("Term not found in attribution table; showing substring matches in articles only.")
        else:
            st.write("**Term Attribution**")
            st.dataframe(tscore, use_container_width=True)

        where_base, args = build_where_and_args()
        # Build a combined text to search; prefer processed fields if present
        text_expr = "COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,'')"

        if whole_word:
            # Regex whole-word (case-insensitive) — duckdb REGEXP_MATCHES supports PCRE
            # (?i) for case-insensitive; use \b boundaries
            pattern = r"(?i)\b" + re.escape(term_input) + r"\b"
            where = f"{where_base} AND REGEXP_MATCHES({text_expr}, $rx)"
            args["rx"] = pattern
        else:
            where = f"{where_base} AND LOWER({text_expr}) LIKE $pat"
            args["pat"] = f"%{term_input.lower()}%"

        hits_sql = f"SELECT * FROM v WHERE {where} LIMIT $lim"
        args["lim"] = int(row_limit)
        hits = con.execute(hits_sql, args).fetchdf()

        st.write(f"**Articles containing “{term_input}”** ({len(hits):,}; showing up to {row_limit:,})")
        st.dataframe(hits, use_container_width=True)

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
        items_view = con.execute(
            """
            SELECT dimension, value, credit, credit_share, rating
            FROM v_item_attr
            WHERE dimension = $dim
            ORDER BY credit_share DESC, value ASC
            LIMIT $lim
            """,
            {"dim": dim_browse, "lim": int(row_limit)},
        ).fetchdf()
        st.dataframe(items_view, use_container_width=True)
    else:
        st.info("No item dimensions available for browsing.")

    # Terms
    st.markdown("### Terms")
    terms_view = con.execute(
        """
        SELECT value, credit, credit_share, rating
        FROM v_term_attr
        QUALIFY ROW_NUMBER() OVER (PARTITION BY value ORDER BY credit_share DESC) = 1
        ORDER BY credit_share DESC
        LIMIT $lim
        """,
        {"lim": int(row_limit)},
    ).fetchdf()
    st.dataframe(terms_view, use_container_width=True)

# ---------- Footer ----------
st.caption(
    "Tip: Use the sidebar to filter by date & sentiment. "
    "Switch the view selector to move between Item Lookup, Term Lookup, and Browse. "
    f"Rows in main table: {n_rows:,}."
)