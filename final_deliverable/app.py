# final_deliverable/app.py
# ------------------------------------------------------------
# Capstone Explorer (Streamlit + DuckDB, Azure Blob, Paginated)
#
# What this app does
# - Ensures a local Parquet cache of your main dataset
# - Queries with DuckDB (server-side filtering/sorting/paging)
# - Displays only a paged slice (fast & reliable on 350k+ rows)
# - Optionally loads an attribution CSV if present
#
# Required secrets (Streamlit Cloud: App -> Settings -> Secrets):
#   AZURE_CONTAINER = "capstone"
#   EITHER:
#     AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
#   OR:
#     AZURE_ACCOUNT_URL = "https://<account>.blob.core.windows.net"
#     AZURE_SAS = "sv=...&ss=...&srt=...&se=...&sp=...&sig=..."    # no leading '?'
#   OR:
#     AZURE_ACCOUNT_URL = "https://<account>.blob.core.windows.net"
#     AZURE_STORAGE_KEY = "<account key==>"
#
# Optional (repo root): .streamlit/config.toml
#   [server]
#   fileWatcherType = "poll"
#   runOnSave = false
#   folderWatchBlacklist = ["data/.*", "\\.venv/.*", ".*\\.(csv|parquet|gpickle)$"]
# ------------------------------------------------------------

import os
# Force polling watcher (avoid inotify limit on Streamlit Cloud)
os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "poll"

import io
import pathlib
import traceback
import datetime as dt

import streamlit as st
import pandas as pd
import duckdb
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as papq
from azure.storage.blob import BlobServiceClient


# ----------------------- CONFIG -----------------------------

# Blob paths (adjust if your paths differ)
PARQUET_BLOB_MAIN = "data/processed/final_model_dataset.parquet"       # preferred main dataset
CSV_BLOB_MAIN     = "data/processed/final_model_dataset.csv"           # fallback if parquet not present

# Optional 2nd dataset (attribution). If not present, app still runs.
ATTR_BLOB_CSV     = "data/processed/attribution_all_scored.csv"

# Local cache (ephemeral on Streamlit Cloud; recreated per boot)
LOCAL_CACHE_MAIN  = pathlib.Path("/tmp/final_model_dataset.parquet")


# --------------------- STREAMLIT SETUP ----------------------

st.set_page_config(page_title="Capstone Explorer", layout="wide")
st.write("boot_ok ✅")  # ensures /healthz succeeds even if data later fails


# --------------------- AZURE HELPERS ------------------------

def _resolve_blob_service() -> BlobServiceClient:
    """Create a BlobServiceClient from secrets/environment."""
    conn_str = st.secrets.get("AZURE_STORAGE_CONNECTION_STRING") or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
    acct_url = st.secrets.get("AZURE_ACCOUNT_URL") or os.getenv("AZURE_ACCOUNT_URL")
    sas      = st.secrets.get("AZURE_SAS") or os.getenv("AZURE_SAS")
    key      = st.secrets.get("AZURE_STORAGE_KEY") or os.getenv("AZURE_STORAGE_KEY")

    if conn_str and "AccountName=" in conn_str and ("AccountKey=" in conn_str or "SharedAccessSignature=" in conn_str):
        return BlobServiceClient.from_connection_string(conn_str)
    if acct_url and (sas or key):
        cred = (sas or "").lstrip("?") or key
        return BlobServiceClient(account_url=acct_url, credential=cred)

    raise RuntimeError(
        "Azure credentials missing. Provide AZURE_STORAGE_CONNECTION_STRING "
        "or AZURE_ACCOUNT_URL + (AZURE_SAS or AZURE_STORAGE_KEY)."
    )


def _blob_client(container: str, blob: str):
    svc = _resolve_blob_service()
    return svc.get_container_client(container).get_blob_client(blob)


def _blob_exists(container: str, blob: str) -> bool:
    try:
        return _blob_client(container, blob).exists()
    except Exception:
        return False


def _download_blob_bytes(container: str, blob: str) -> bytes:
    return _blob_client(container, blob).download_blob().readall()


# --------------- PARQUET CACHE (ONE-TIME PER BOOT) ----------

@st.cache_resource(show_spinner=True)
def ensure_main_parquet_cached(container: str) -> pathlib.Path:
    """
    Ensure we have a local Parquet copy of the main dataset at LOCAL_CACHE_MAIN.
    Prefer downloading an existing Parquet from blob; otherwise convert CSV -> Parquet once.
    """
    LOCAL_CACHE_MAIN.parent.mkdir(parents=True, exist_ok=True)
    if LOCAL_CACHE_MAIN.exists() and LOCAL_CACHE_MAIN.stat().st_size > 0:
        return LOCAL_CACHE_MAIN

    if _blob_exists(container, PARQUET_BLOB_MAIN):
        # Download Parquet directly (fast path)
        data = _download_blob_bytes(container, PARQUET_BLOB_MAIN)
        LOCAL_CACHE_MAIN.write_bytes(data)
        return LOCAL_CACHE_MAIN

    # Fallback: CSV -> Parquet one-time conversion
    if not _blob_exists(container, CSV_BLOB_MAIN):
        raise FileNotFoundError(
            f"Neither Parquet nor CSV found for main dataset. "
            f"Looked for:\n - {PARQUET_BLOB_MAIN}\n - {CSV_BLOB_MAIN}"
        )

    data = _download_blob_bytes(container, CSV_BLOB_MAIN)
    # Use Arrow CSV reader (streaming & lower memory vs pandas.read_csv)
    table = pacsv.read_csv(pa.py_buffer(data))

    # Optional: example type tweaks (uncomment/adapt as needed)
    # if "load_date" in table.column_names:
    #     table = table.set_column(
    #         table.column_names.index("load_date"),
    #         "load_date",
    #         pa.compute.strptime(table["load_date"], format="%Y-%m-%d", unit="us"),
    #     )

    papq.write_table(table, LOCAL_CACHE_MAIN)
    return LOCAL_CACHE_MAIN


# ------------------- DUCKDB CONNECTION ----------------------

@st.cache_resource(show_spinner=False)
def get_duck_conn(local_parquet_path: pathlib.Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    # Register main dataset view; DuckDB lazily scans Parquet & pushes down filters/sorts.
    con.execute("""
        CREATE OR REPLACE VIEW v AS
        SELECT * FROM read_parquet(?, hive_partitioning=FALSE)
    """, [str(local_parquet_path)])
    return con


def register_optional_attr(con: duckdb.DuckDBPyConnection, container: str) -> bool:
    """If the attribution CSV exists, register it as a DuckDB view (v_attr). Returns True/False."""
    try:
        if not _blob_exists(container, ATTR_BLOB_CSV):
            return False
        data = _download_blob_bytes(container, ATTR_BLOB_CSV)
        # Read via Arrow then register as DuckDB relation
        attr_tbl = pacsv.read_csv(pa.py_buffer(data))
        con.register("attr_arrow", attr_tbl)  # temp table from Arrow
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT * FROM attr_arrow")
        return True
    except Exception:
        return False


# ------------------- SERVER-SIDE QUERY ----------------------

def query_rows(
    con: duckdb.DuckDBPyConnection,
    *,
    pub: list[str] | None = None,
    band: list[str] | None = None,
    dmin: dt.date | None = None,
    dmax: dt.date | None = None,
    search: str | None = None,
    sort_col: str | None = None,
    sort_dir: str = "asc",
    page: int = 0,
    page_size: int = 100,
) -> tuple[pa.Table, int]:
    """
    Filter/sort/paginate in DuckDB and return an Arrow table slice + total count.
    """
    clauses = ["SELECT * FROM v WHERE 1=1"]
    args = {}

    if pub:
        clauses.append("AND publication_name IN $pub")
        args["pub"] = pub
    if band:
        clauses.append("AND sentiment_band IN $band")
        args["band"] = band
    if dmin:
        clauses.append("AND load_date >= $dmin")
        args["dmin"] = pd.to_datetime(dmin)
    if dmax:
        # exclusive upper bound for easy day-range filtering
        clauses.append("AND load_date <  $dmax")
        args["dmax"] = pd.to_datetime(dmax) + pd.Timedelta(days=1)
    if search:
        clauses.append("AND lower(coalesce(processed_headline, '')) LIKE $q")
        args["q"] = f"%{search.lower()}%"

    # Sorting (identifier-escaped)
    if sort_col:
        ident = duckdb.escape_identifier(sort_col)
        direction = "DESC" if str(sort_dir).lower() == "desc" else "ASC"
        clauses.append(f"ORDER BY {ident} {direction}")

    # Pagination
    clauses.append("LIMIT $lim OFFSET $off")
    args["lim"] = int(page_size)
    args["off"] = int(page) * int(page_size)

    sql = " ".join(clauses)

    cur = con.execute(sql, args)
    page_tbl = cur.fetch_arrow_table()

    # Count total with same filters (no sort/limit/offset)
    base = " ".join(s for s in clauses if not s.startswith("ORDER BY") and not s.startswith("LIMIT"))
    total_sql = f"SELECT COUNT(*) FROM ({base}) t"
    total = con.execute(total_sql, args).fetchone()[0]

    return page_tbl, int(total)


# ------------------------ UI LOGIC --------------------------

st.header("Attribution Lookup — DuckDB (Azure Blob)")

container = st.secrets.get("AZURE_CONTAINER") or os.getenv("AZURE_CONTAINER")
if not container:
    st.error("Missing AZURE_CONTAINER secret. Set it in Streamlit Secrets.")
    st.stop()

# Bootstrap data (safe: shows error in UI instead of crashing process)
with st.spinner("Bootstrapping data (Parquet cache + DuckDB)…"):
    try:
        pq_path = ensure_main_parquet_cached(container)
        con = get_duck_conn(pq_path)
        has_attr = register_optional_attr(con, container)
    except Exception as e:
        st.error(f"Data bootstrap failed: {e}")
        st.expander("Traceback").code(traceback.format_exc())
        st.stop()

# ---- Top summary / metadata
meta_cols = st.columns([1, 1, 1, 1])
with meta_cols[0]:
    n_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
    st.metric("Rows (main)", f"{n_rows:,}")
with meta_cols[1]:
    n_pubs = con.execute("SELECT COUNT(DISTINCT publication_name) FROM v").fetchone()[0]
    st.metric("Distinct publications", f"{n_pubs:,}")
with meta_cols[2]:
    date_min, date_max = con.execute("SELECT min(load_date), max(load_date) FROM v").fetchone()
    min_d = date_min.date() if pd.notna(date_min) else None
    max_d = date_max.date() if pd.notna(date_max) else None
    st.metric("Date range", f"{min_d} → {max_d}" if (min_d and max_d) else "n/a")
with meta_cols[3]:
    st.metric("Attribution table", "Loaded" if has_attr else "Not found")

st.divider()

# ---- Filters
f1, f2, f3, f4 = st.columns([1, 1, 1, 1])

with f1:
    pubs = con.execute("SELECT DISTINCT publication_name FROM v ORDER BY 1").fetchdf()["publication_name"].dropna().tolist()
    sel_pub = st.multiselect("Publication", pubs, max_selections=10, placeholder="Select up to 10…")

with f2:
    bands = con.execute("SELECT DISTINCT sentiment_band FROM v ORDER BY 1").fetchdf()["sentiment_band"].dropna().tolist()
    sel_band = st.multiselect("Sentiment band", bands)

with f3:
    default_range = (min_d, max_d) if (min_d and max_d) else None
    dr = st.date_input("Date range", value=default_range)

with f4:
    q = st.text_input("Search headline (contains)")

s1, s2, s3 = st.columns([1, 1, 1])
with s1:
    sort_col = st.selectbox(
        "Sort by",
        ["load_date", "publication_name", "sentiment_score", "vipr_score", "hit_strength"],
        index=0,
    )
with s2:
    sort_dir = st.radio("Direction", ["asc", "desc"], horizontal=True, index=0)
with s3:
    page_size = st.selectbox("Page size", [50, 100, 200, 500], index=1)

page = st.number_input("Page (0-based)", min_value=0, step=1, value=0)

# ---- Query + results
with st.spinner("Querying…"):
    try:
        tbl, total = query_rows(
            con,
            pub=sel_pub or None,
            band=sel_band or None,
            dmin=dr[0] if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dr[0] if isinstance(dr, list) and dr else None),
            dmax=dr[1] if isinstance(dr, (list, tuple)) and len(dr) == 2 else (dr[1] if isinstance(dr, list) and len(dr) > 1 else None),
            search=q or None,
            sort_col=sort_col,
            sort_dir=sort_dir,
            page=page,
            page_size=page_size,
        )
    except Exception as e:
        st.error(f"Query failed: {e}")
        st.expander("Traceback").code(traceback.format_exc())
        st.stop()

shown = min((page + 1) * page_size, total)
st.write(f"Showing {shown:,} of {total:,} (page {page})")

# Display current slice
slice_df = tbl.to_pandas()
st.dataframe(slice_df, use_container_width=True, hide_index=True)

# Download just the current page
st.download_button(
    "Download current page (CSV)",
    data=slice_df.to_csv(index=False).encode("utf-8"),
    file_name=f"results_page{page}.csv",
    mime="text/csv",
)

st.caption("Tip: Increase Page size or change Sort to navigate quickly. For full exports, add a server-side export button that writes a filtered Parquet/CSV to Blob.")


# -------------------- OPTIONAL: ATTR VIEW -------------------
# If your attribution CSV was found, offer a very simple peek.
if has_attr:
    with st.expander("Attribution table (sample)"):
        try:
            attr_head = con.execute("SELECT * FROM v_attr LIMIT 200").fetchdf()
            st.dataframe(attr_head, use_container_width=True, hide_index=True)
            st.caption("This is a small sample for inspection; integrate with filters as needed.")
        except Exception as e:
            st.warning(f"Unable to preview attribution: {e}")