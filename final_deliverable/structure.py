# structure.py — shared bootstrap, helpers, header, filters, and enriched view (with local fallback)

from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import streamlit as st
import duckdb
import pandas as pd
import altair as alt
from azure.storage.blob import BlobServiceClient

# -----------------------------------------------------------------------------
# Brand & CSS
# -----------------------------------------------------------------------------
BRAND = {"primary": "#12715D", "accent": "#4AB48E", "text": "#133C35", "bg2": "#F4F6F5"}

def inject_base_css() -> None:
    st.markdown(
        f"""
<style>
  html, body, [class*="css"] {{
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: {BRAND['text']};
  }}
  .block-container {{ padding-top: 0.4rem; padding-bottom: 2rem; }}
  section[data-testid="stSidebar"] > div {{ background-color: {BRAND['bg2']}; }}
  div[data-testid="stMetric"] {{
    background: {BRAND['bg2']}; border-radius: 12px; padding: 10px; border: 1px solid #e6e6e6;
  }}
  h1,h2,h3,h4,h5 {{ color: {BRAND['primary']}; }}
  .header-wrap {{ display: flex; align-items: center; gap: 1rem; margin-bottom: .5rem; }}
  .penta-logo {{ height: 56px; width: auto; }}
  @media (max-width: 1200px) {{ .penta-logo {{ height: 48px; }} }}
</style>
""",
        unsafe_allow_html=True,
    )
    alt.data_transformers.disable_max_rows()

# -----------------------------------------------------------------------------
# Azure + DuckDB bootstrap (defensive)
# -----------------------------------------------------------------------------
APP_TMP = Path(os.getenv("STREAMLIT_TMPDIR", "/tmp"))

@st.cache_resource
def _azure_client():
    """Initialize Azure client with simplified error handling."""
    try:
        # Check if secrets are available
        if "data" not in st.secrets:
            raise KeyError("Azure configuration missing. Add [data] section to secrets.toml.")
        
        cfg = st.secrets["data"]
        
        # Validate required configuration
        if "AZURE_STORAGE_CONNECTION_STRING" not in cfg:
            raise KeyError("AZURE_STORAGE_CONNECTION_STRING not found in secrets.")
        if "AZURE_CONTAINER" not in cfg:
            raise KeyError("AZURE_CONTAINER not found in secrets.")
        
        bsc = BlobServiceClient.from_connection_string(cfg["AZURE_STORAGE_CONNECTION_STRING"])
        cont = bsc.get_container_client(cfg["AZURE_CONTAINER"])
        return bsc, cont, cfg
    except KeyError as e:
        raise RuntimeError(f"Azure configuration missing: {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to connect to Azure: {e}")

def _norm(blob: str) -> str:
    return str(blob).strip().lstrip("/")

def _exists(cont, blob: str) -> bool:
    """Check if blob exists with simplified error handling."""
    return bool(blob) and cont.get_blob_client(_norm(blob)).exists()

def _download_blob(cont, blob: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    bc = cont.get_blob_client(_norm(blob))
    with open(dest, "wb") as f:
        f.write(bc.download_blob().readall())
    return dest

def _local_candidates() -> Dict[str, Path]:
    """Check for local fallback files if Azure blobs are missing."""
    repo_root = Path(__file__).resolve().parent.parent
    candidates = {
        "parquet": [
            repo_root / "data/processed/final_model_dataset.parquet",
        ],
        "csv": [
            repo_root / "data/final_model_dataset.csv",  # main data directory
            repo_root / "data/final_dataset_sampled.csv",  # main data directory
            repo_root / "data/processed/final_model_dataset.csv",
            repo_root / "data/processed/final_dataset_sampled.csv",
            repo_root / "data/processed/data/processed_data.csv",
            repo_root / "data/processed/data/text_processed_data.csv",
        ],
        "attr": [
            repo_root / "data/attribution_all_scored.csv",  # main data directory
            repo_root / "data/processed/attribution_all_scored.csv",
        ],
        "logo": [
            Path(__file__).parent / "penta_logo.png",  # actual location
            Path(__file__).parent / "assets" / "penta_logo.png",
            Path(__file__).parent / "assets" / "logo.png",
        ],
    }
    found = {}
    for key, paths in candidates.items():
        for p in paths:
            if p.exists():
                found[key] = p
                break
    return found

@st.cache_data(show_spinner=False)
def load_data_from_azure() -> Dict[str, Optional[Path]]:
    """Load data from Azure with simplified logic and better error handling."""
    # First try to get local data as fallback
    local = _local_candidates()
    
    try:
        # Try Azure connection
        bsc, cont, cfg = _azure_client()
        
        # Define blob mappings
        blob_configs = {
            "data_path": ("parquet_blob", "final_model_dataset.parquet"),
            "csv_path": ("csv_blob", "final_model_dataset.csv"),
            "attr_path": ("attr_blob", "attribution_all_scored.csv"),
            "logo_path": ("logo_blob", "penta_logo.png")
        }
        
        paths = {}
        for key, (blob_key, filename) in blob_configs.items():
            blob_name = str(cfg.get(blob_key, "")).strip()
            if blob_name and _exists(cont, blob_name):
                paths[key] = _download_blob(cont, blob_name, APP_TMP / filename)
            else:
                paths[key] = None
        
        # Use local fallback if no data files found from Azure
        if not paths["data_path"] and not paths["csv_path"]:
            if "parquet" in local:
                paths["data_path"] = local["parquet"]
                st.info("Using local Parquet file (Azure data not available).")
            elif "csv" in local:
                paths["csv_path"] = local["csv"]
                st.info("Using local CSV file (Azure data not available).")
            
            # Add local attribution and logo if available
            if "attr" in local:
                paths["attr_path"] = local["attr"]
            if "logo" in local:
                paths["logo_path"] = local["logo"]
        
        if not paths["data_path"] and not paths["csv_path"]:
            raise FileNotFoundError("No dataset found in Azure or local paths.")
        
        return paths
        
    except Exception as e:
        # Azure failed, try local fallback
        st.warning(f"Azure connection failed: {e}")
        st.info("Falling back to local data files...")
        
        if not local:
            raise FileNotFoundError("No data files found locally and Azure connection failed.")
        
        paths = {"data_path": None, "csv_path": None, "attr_path": None, "logo_path": None}
        
        if "parquet" in local:
            paths["data_path"] = local["parquet"]
            st.info("Using local Parquet file.")
        elif "csv" in local:
            paths["csv_path"] = local["csv"]
            st.info("Using local CSV file.")
        
        if "attr" in local:
            paths["attr_path"] = local["attr"]
        if "logo" in local:
            paths["logo_path"] = local["logo"]
        
        if not paths["data_path"] and not paths["csv_path"]:
            raise FileNotFoundError("No valid data files found locally.")
        
        return paths

@st.cache_resource
def connect_duckdb_with_azure():
    paths = load_data_from_azure()
    con = duckdb.connect(database=":memory:")
    if paths["data_path"] and str(paths["data_path"]).endswith(".parquet"):
        con.execute("CREATE VIEW v AS SELECT * FROM read_parquet(?)", [str(paths["data_path"])])
    elif paths["csv_path"]:
        con.execute("CREATE VIEW v AS SELECT * FROM read_csv_auto(?, header=True)", [str(paths["csv_path"])])
    else:
        raise FileNotFoundError("No valid dataset for DuckDB.")
    return con, paths

def maybe_set_logo(paths: Dict[str, Optional[Path]]) -> Dict[str, Optional[Path]]:
    """Ensure logo_path exists in paths dict."""
    return paths.setdefault("logo_path", None) or paths

def render_header(paths: Dict[str, Optional[Path]], title: str) -> None:
    logo_html = ""
    if paths.get("logo_path"):
        logo_html = f'<img src="file://{Path(paths["logo_path"]).as_posix()}" class="penta-logo"/>'
    st.write(f'<div class="header-wrap">{logo_html}<h1 style="margin:0;">{title}</h1></div>', unsafe_allow_html=True)

# ---------------- v_enriched & helpers ----------------
def _clean_expr(col_sql: str) -> str:
    """Simplified column cleaning with fewer regex operations."""
    return f"""
    NULLIF(TRIM(
      REGEXP_REPLACE(
        REGEXP_REPLACE(
          REGEXP_REPLACE(COALESCE({col_sql}, ''), '&#160;|&nbsp;|&NBSP;|\\s+', ' '),
          '^[,;:\\-]+\\s*', ''
        ), '^\\((.*)\\)$', '\\\\1'
      )
    ), '')
    """

def quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _first_existing(cols: List[str], existing: Set[str]) -> Optional[str]:
    """Find first column that exists in the dataset."""
    return next((c for c in cols if c in existing), None)

def _cleaned_select(base_cols: List[str], alias: str, existing: Set[str]) -> str:
    base = _first_existing(base_cols, existing)
    return (f"{_clean_expr('v.'+quote_ident(base))} AS {quote_ident(alias)}") if base else f"NULL AS {quote_ident(alias)}"

def build_v_enriched(con) -> Set[str]:
    """Build enriched view with simplified column mapping."""
    cols_df = con.execute("PRAGMA table_info('v')").fetchdf()
    columns = set(cols_df["name"].astype(str).tolist())

    # Simplified column mapping
    column_mappings = [
        (["author", "author_name"], "author_clean"),
        (["publication_name", "publication"], "publication_clean"),
        (["channel", "channel_name", "source_type"], "channel_clean"),
        (["topic", "topics"], "topic_clean"),
    ]
    
    select_parts = ["v.*"]
    for source_cols, target_col in column_mappings:
        select_parts.append(_cleaned_select(source_cols, target_col, columns))
    
    con.execute("CREATE OR REPLACE VIEW v_enriched AS SELECT " + ", ".join(select_parts) + " FROM v AS v")
    return columns

@st.cache_data
def date_bounds(con, col: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        mn, mx = con.execute(
            f"SELECT MIN(CAST({quote_ident(col)} AS TIMESTAMP)), MAX(CAST({quote_ident(col)} AS TIMESTAMP)) FROM v"
        ).fetchone()
        if mn and mx:
            return pd.to_datetime(mn), pd.to_datetime(mx)
    except Exception:
        pass
    return None

@st.cache_data
def distinct_clean(con, expr_sql: str) -> List[str]:
    """Get distinct values with simplified query."""
    try:
        df = con.execute(
            f"SELECT DISTINCT {expr_sql} AS val FROM v_enriched WHERE {expr_sql} IS NOT NULL ORDER BY 1"
        ).fetchdf()
        return df["val"].astype(str).tolist()
    except Exception:
        return []

def shorten(label: str, max_len: int = 28) -> str:
    s = str(label)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

def build_sidebar_filters(con, COLUMNS: Set[str]) -> Dict:
    """Build sidebar filters with simplified logic."""
    st.sidebar.header("Filters")
    
    # Date filter
    ld = date_bounds(con, "load_ts")
    date_range = None
    if ld:
        date_range = st.sidebar.date_input("Load date range", (ld[0].date(), ld[1].date()))
    
    # Publication filter
    sel_pubs = []
    if "publication_clean" in COLUMNS:
        sel_pubs = st.sidebar.multiselect("Publication", distinct_clean(con, "publication_clean"))
    
    # Channel filter
    sel_channels = []
    if "channel" in COLUMNS:
        sel_channels = st.sidebar.multiselect("Channel", distinct_clean(con, "channel_clean"))
    
    # Row limit
    row_limit = st.sidebar.number_input("Rows in tables", 100, 10000, 1000, 50)

    # Build SQL conditions
    clauses, args = ["1=1"], {}
    
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        dmin = pd.to_datetime(date_range[0])
        dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        clauses.append("CAST(load_ts AS TIMESTAMP) >= $dmin AND CAST(load_ts AS TIMESTAMP) < $dmax")
        args.update(dmin=dmin, dmax=dmax)
    
    if sel_pubs:
        clauses.append("publication_clean IN $pubs")
        args["pubs"] = sel_pubs
    
    if sel_channels:
        clauses.append("channel_clean IN $chs")
        args["chs"] = sel_channels

    return {"where_sql": " AND ".join(clauses), "args": args, "row_limit": int(row_limit)}

def explain_attribution(row: pd.Series, peers: Optional[pd.DataFrame] = None) -> str:
    """Simplified attribution explanation with better error handling."""
    dim = str(row.get("dimension", "dimension"))
    val = str(row.get("value", "value"))
    cred = float(row.get("credit", 0) or 0)
    share = float(row.get("credit_share", 0) or 0)
    
    out = [f"**{dim} = {val}** contributes **{share:.2%}** of total credit (raw {cred:,.4f})."]
    
    if peers is not None and not peers.empty and "credit" in peers.columns:
        try:
            same = peers[peers["dimension"] == dim]
            if not same.empty and val in same["value"].values:
                same = same.assign(_r=same["credit"].rank(ascending=False, method="min"))
                r = int(same.loc[same["value"] == val, "_r"].iloc[0])
                n = int(same.shape[0])
                percentile = 100 * (1 - (r - 1) / max(n, 1))
                out.append(f"Rank **#{r} of {n}** (~{percentile:.0f}th percentile).")
        except Exception:
            pass  # Skip ranking if there's an error
    
    return " ".join(out)