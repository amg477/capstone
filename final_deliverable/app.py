# app.py —  PolicyPath (Tabbed: Attribution • Dashboard • Network)

from __future__ import annotations
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import duckdb
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import base64
# Using Plotly and Altair instead of matplotlib for better Streamlit Cloud compatibility

# -------------------- Page config --------------------
st.set_page_config(page_title="PolicyPath", layout="wide")

# -------------------- Session State Initialization --------------------
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'recent_searches' not in st.session_state:
    st.session_state.recent_searches = []
if 'favorites' not in st.session_state:
    st.session_state.favorites = []
if 'saved_views' not in st.session_state:
    st.session_state.saved_views = {}

# -------------------- Global Penta Brand Styling --------------------
st.markdown("""
<style>
    /* Penta Brand Colors */
    :root {
        --penta-primary: #12715D;
        --penta-accent: #4AB48E;
        --penta-light: #E5F4F1;
        --penta-lighter: #C8EADF;
        --penta-dark: #0A473B;
        --penta-white: #FFFFFF;
    }
    
    /* Main app styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--penta-dark);
        font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
        font-weight: 600;
    }
    
    h1 {
        font-size: 2.5rem;
        letter-spacing: -0.02em;
    }
    
    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    
    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: var(--penta-light);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: var(--penta-dark);
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--penta-primary);
        color: var(--penta-white);
    }
    
    /* Metrics */
    .metric-container {
        background-color: var(--penta-light);
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid var(--penta-primary);
    }
    
    /* Data tables */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background-color: var(--penta-light);
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--penta-primary);
        color: var(--penta-white);
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(18, 113, 93, 0.2);
    }
    
    .stButton > button:hover {
        background-color: var(--penta-accent);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(18, 113, 93, 0.3);
    }
    
    /* Loading animations */
    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--penta-light);
        border-radius: 50%;
        border-top-color: var(--penta-primary);
        animation: spin 1s ease-in-out infinite;
    }
    
    @keyframes spin {
        to { transform: rotate(360deg); }
    }
    
    .pulse-animation {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Enhanced tooltips */
    .tooltip {
        position: relative;
        display: inline-block;
        cursor: help;
    }
    
    .tooltip .tooltiptext {
        visibility: hidden;
        width: 200px;
        background-color: var(--penta-dark);
        color: var(--penta-white);
        text-align: center;
        border-radius: 6px;
        padding: 8px;
        position: absolute;
        z-index: 1;
        bottom: 125%;
        left: 50%;
        margin-left: -100px;
        opacity: 0;
        transition: opacity 0.3s;
        font-size: 12px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .tooltip:hover .tooltiptext {
        visibility: visible;
        opacity: 1;
    }
    
    /* Enhanced metrics */
    .metric-card {
        background: linear-gradient(135deg, var(--penta-light) 0%, var(--penta-lighter) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--penta-primary);
        box-shadow: 0 2px 8px rgba(18, 113, 93, 0.1);
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(18, 113, 93, 0.2);
    }
    
    /* Enhanced data tables */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    .stDataFrame table {
        border-collapse: separate;
        border-spacing: 0;
    }
    
    .stDataFrame th {
        background: linear-gradient(135deg, var(--penta-primary) 0%, var(--penta-accent) 100%);
        color: var(--penta-white);
        font-weight: 600;
        padding: 12px;
    }
    
    .stDataFrame td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--penta-light);
    }
    
    .stDataFrame tr:hover {
        background-color: var(--penta-light);
    }
    
    /* Enhanced charts */
    .chart-container {
        background: var(--penta-white);
        border-radius: 12px;
        padding: 1rem;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Search enhancements */
    .search-container {
        position: relative;
        margin: 1rem 0;
    }
    
    .search-suggestions {
        position: absolute;
        top: 100%;
        left: 0;
        right: 0;
        background: var(--penta-white);
        border: 1px solid var(--penta-light);
        border-radius: 6px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        z-index: 1000;
        max-height: 200px;
        overflow-y: auto;
    }
    
    .search-suggestion {
        padding: 8px 12px;
        cursor: pointer;
        border-bottom: 1px solid var(--penta-light);
    }
    
    .search-suggestion:hover {
        background-color: var(--penta-light);
    }
    
    /* Enhanced suggestion buttons */
    .suggestion-button {
        background: linear-gradient(135deg, var(--penta-light) 0%, var(--penta-lighter) 100%);
        border: 1px solid var(--penta-primary);
        border-radius: 6px;
        padding: 6px 12px;
        margin: 2px;
        font-size: 0.85rem;
        color: var(--penta-dark);
        transition: all 0.2s ease;
        cursor: pointer;
        width: 100%;
        text-align: left;
    }
    
    .suggestion-button:hover {
        background: linear-gradient(135deg, var(--penta-primary) 0%, var(--penta-accent) 100%);
        color: var(--penta-white);
        transform: translateY(-1px);
        box-shadow: 0 2px 4px rgba(18, 113, 93, 0.2);
    }
    
    /* Suggestion container */
    .suggestions-container {
        background: var(--penta-lighter);
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid var(--penta-primary);
    }
    
    .suggestions-title {
        color: var(--penta-dark);
        font-weight: 600;
        margin-bottom: 0.5rem;
        font-size: 0.9rem;
    }
    
    /* Dark mode toggle */
    .dark-mode-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: var(--penta-primary);
        color: var(--penta-white);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }
    
    .dark-mode-toggle:hover {
        background: var(--penta-accent);
        transform: scale(1.1);
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        .header-title h1 {
            font-size: 2rem;
        }
        
        .header-subtitle {
            font-size: 1rem;
        }
        
        .penta-logo {
            height: 40px;
        }
        
        .metric-card {
            padding: 1rem;
        }
    }
    
    /* Success/Error animations */
    .success-message {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease;
    }
    
    .error-message {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        animation: slideIn 0.5s ease;
    }
    
    @keyframes slideIn {
        from { transform: translateX(-100%); opacity: 0; }
        to { transform: translateX(0); opacity: 1; }
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Brand styling --------------------
def apply_penta_style():
    # Penta brand colors for Plotly/Altair charts
    PRIMARY_GREEN = "#12715D"      # Main brand green
    ACCENT_GREEN  = "#4AB48E"      # Accent green
    LIGHT_GREEN   = "#E5F4F1"      # Light background
    LIGHTER_GREEN = "#C8EADF"      # Even lighter
    DARK_GREEN    = "#0A473B"      # Dark green for text
    WHITE         = "#FFFFFF"      # Clean white
    
    # Set Altair theme for consistent styling
    alt.themes.enable('default')
    
    # Configure Altair with Penta colors
    alt.data_transformers.disable_max_rows()
    
    # Return color palette for use in charts
    return {
        'primary': PRIMARY_GREEN,
        'accent': ACCENT_GREEN,
        'light': LIGHT_GREEN,
        'lighter': LIGHTER_GREEN,
        'dark': DARK_GREEN,
        'white': WHITE
    }

def s(path: str, default=None):
    try:
        cur = st.secrets
        for part in path.split("."):
            cur = cur[part]
        return cur
    except Exception:
        return default

# -------------------- Enhanced Utility Functions --------------------
def show_loading_spinner(text="Loading..."):
    """Show a custom loading spinner with text."""
    with st.spinner(text):
        st.markdown(f"""
        <div class="loading-spinner"></div>
        <span style="margin-left: 10px;">{text}</span>
        """, unsafe_allow_html=True)

def show_success_message(message):
    """Show a success message with animation."""
    st.markdown(f"""
    <div class="success-message">
        ✅ {message}
    </div>
    """, unsafe_allow_html=True)

def show_error_message(message):
    """Show an error message with animation."""
    st.markdown(f"""
    <div class="error-message">
        ❌ {message}
    </div>
    """, unsafe_allow_html=True)

def add_to_recent_searches(search_term):
    """Add search term to recent searches."""
    if search_term and search_term not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, search_term)
        st.session_state.recent_searches = st.session_state.recent_searches[:10]  # Keep only 10 recent

def get_search_suggestions(query, data_list):
    """Get search suggestions based on query."""
    if not query or len(query) < 2:
        return []
    
    suggestions = []
    query_lower = query.lower()
    
    for item in data_list:
        if query_lower in item.lower():
            suggestions.append(item)
    
    return suggestions[:5]  # Return top 5 suggestions

def create_metric_card(title, value, change=None, icon="📊"):
    """Create an enhanced metric card."""
    change_html = ""
    if change is not None:
        change_color = "green" if change > 0 else "red" if change < 0 else "gray"
        change_symbol = "↗" if change > 0 else "↘" if change < 0 else "→"
        change_html = f'<div style="color: {change_color}; font-size: 0.9rem; margin-top: 0.5rem;">{change_symbol} {abs(change):.1f}%</div>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div style="display: flex; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 1.5rem; margin-right: 0.5rem;">{icon}</span>
            <h4 style="margin: 0; color: var(--penta-dark);">{title}</h4>
        </div>
        <div style="font-size: 2rem; font-weight: bold; color: var(--penta-primary);">{value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)

def create_tooltip(text, tooltip_text):
    """Create an element with tooltip."""
    return f"""
    <div class="tooltip">
        {text}
        <span class="tooltiptext">{tooltip_text}</span>
    </div>
    """

def export_data_button(data, filename, format_type="csv"):
    """Create an enhanced export button."""
    if format_type == "csv":
        csv = data.to_csv(index=False)
        st.download_button(
            label=f"📥 Download {filename}.csv",
            data=csv,
            file_name=f"{filename}.csv",
            mime="text/csv",
            help="Download data as CSV file"
        )
    elif format_type == "json":
        json_data = data.to_json(orient='records', indent=2)
        st.download_button(
            label=f"📥 Download {filename}.json",
            data=json_data,
            file_name=f"{filename}.json",
            mime="application/json",
            help="Download data as JSON file"
        )

def _q(p: Path) -> str:
    return "'" + str(p).replace("'", "''") + "'"

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

def _clean_expr(col_sql: str) -> str:
    return f"""
    NULLIF(TRIM(
      REGEXP_REPLACE(
        REGEXP_REPLACE(
          REGEXP_REPLACE(
            REGEXP_REPLACE(COALESCE({col_sql}, ''), '&#160;|&nbsp;|&NBSP;', ' '),
            '\\s+', ' '
          ),
          '^[,;:\\-]+\\s*', ''
        ), '^\\((.*)\\)$', '\\\\1'
      )
    ), '')
    """

def _first_existing(cols: List[str], existing: Set[str]) -> Optional[str]:
    for c in cols:
        if c in existing: 
            return c
    return None

def _cleaned_select(cols: List[str], alias: str, existing: Set[str]) -> str:
    base = _first_existing(cols, existing)
    return (f"{_clean_expr('v.'+base)} AS {alias}") if base else f"NULL AS {alias}"

def shorten(label: str, max_len: int = 28) -> str:
    s = str(label)
    return s if len(s) <= max_len else s[:max_len-1] + "…"

def build_where(extra: str = "", params: Optional[Dict] = None, 
                date_range=None, sel_bands=None, sel_pubs=None) -> Tuple[str, Dict]:
    clauses: List[str] = ["1=1"]
    args: Dict = {} if params is None else dict(params)

    if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and date_range[0] and date_range[1]:
        dmin = pd.to_datetime(date_range[0])
        dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        clauses.append("load_date >= $dmin AND load_date < $dmax")
        args["dmin"], args["dmax"] = dmin, dmax

    if sel_bands:
        clauses.append("sentiment_band IN $bands")
        args["bands"] = sel_bands

    if sel_pubs:
        clauses.append("publication_name IN $pubs")
        args["pubs"] = sel_pubs

    if extra:
        clauses.append(extra)

    return " AND ".join(clauses), args

def where_from_filters(date_range=None, sel_pubs=None, sel_channels=None, 
                      sel_bands=None, sel_authors=None, sel_topics=None) -> Tuple[str, Dict]:
    clauses, args = ["1=1"], {}
    
    if date_range and len(date_range) == 2:
        dmin = pd.to_datetime(date_range[0])
        dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        clauses.append("CAST(load_date AS TIMESTAMP) >= $dmin AND CAST(load_date AS TIMESTAMP) < $dmax")
        args.update(dmin=dmin, dmax=dmax)
    
    if sel_pubs:
        clauses.append("publication_clean IN $pubs")
        args["pubs"] = sel_pubs
    if sel_channels:
        clauses.append("COALESCE(channel_clean, channel_name_clean) IN $chs")
        args["chs"] = sel_channels
    if sel_bands:
        clauses.append("sentiment_band IN $bands")
        args["bands"] = sel_bands
    if sel_authors:
        clauses.append("COALESCE(author_clean, author_name_clean) IN $auths")
        args["auths"] = sel_authors
    if sel_topics:
        clauses.append("COALESCE(topic_clean, topics_clean) IN $topics")
        args["topics"] = sel_topics
    
    return " AND ".join(clauses), args

@st.cache_data
def date_bounds(con: duckdb.DuckDBPyConnection, col: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        mn, mx = con.execute(f"SELECT MIN(CAST({col} AS TIMESTAMP)), MAX(CAST({col} AS TIMESTAMP)) FROM v").fetchone()
        if mn and mx: 
            return (pd.to_datetime(mn), pd.to_datetime(mx))
    except Exception: 
        pass
    return None

@st.cache_data
def distinct_clean(con: duckdb.DuckDBPyConnection, expr_sql: str) -> List[str]:
    df = con.execute(f"SELECT DISTINCT {expr_sql} AS val FROM v_enriched WHERE {expr_sql} IS NOT NULL ORDER BY 1").fetchdf()
    return df["val"].tolist()

def explain_attribution(row: pd.Series, universe: Optional[pd.DataFrame] = None) -> str:
    dim = str(row.get("dimension", "dimension"))
    val = str(row.get("value", "value"))
    cred = float(row.get("credit", 0.0))
    share = float(row.get("credit_share", 0.0))
    rating = row.get("rating", None)

    parts = [f"**{dim} = {val}**"]
    parts.append(
        "within the current selection"
        + (f" (rating = **{int(rating)}**)" if pd.notna(rating) else "")
        + "."
    )
    parts.append(f"It contributes **{share:.2%}** of total attribution credit (raw credit **{cred:,.4f}**).")

    if universe is not None and not universe.empty and "dimension" in universe.columns:
        peers = universe[universe["dimension"] == dim]
        if not peers.empty and val in peers.get("value", pd.Series(dtype=str)).values:
            peers = peers.assign(_rank=peers["credit"].rank(ascending=False, method="min"))
            n = len(peers)
            my_rank = int(peers.loc[peers["value"] == val, "_rank"].iloc[0])
            pct = 100 * (1 - (my_rank - 1) / max(n, 1))
            parts.append(f"Rank **#{my_rank} of {n}** (~**{pct:.0f}th percentile**).")
            med = peers["credit"].median()
            if med and med > 0:
                parts.append(f"About **{(cred/med-1):+.0%}** vs median {dim}.")

    if share < 0.001:
        parts.append("Very small share — likely low standalone influence.")

    return " ".join(parts)

# -------------------- Data loading --------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
MODE = (s("data.mode", "local") or "local").lower()

DATA_DIR = (ROOT / s("data.data_dir", "data")).resolve()
PARQUET_NAME = s("data.parquet", "final_model_dataset.parquet")
CSV_NAME = s("data.csv", "final_model_dataset.csv")
ATTR_NAME = s("data.attr_csv", "attribution_all_scored.csv")

CANDIDATE_DIRS = [DATA_DIR, ROOT / "data", ROOT / "data" / "processed"]
SEARCH_DIRS = [ROOT/"data/processed", ROOT/"data", APP_DIR]
CANDIDATE_FILES = ["final_model_dataset.parquet", "final_model_dataset.csv", "data.parquet", "data.csv"]

LOGO_PATH = (ROOT / s("data.logo", "final_deliverable/penta_logo.png"))

def _find_first_existing(*names: str) -> Optional[Path]:
    for d in CANDIDATE_DIRS:
        for nm in names:
            p = d / nm
            if p.exists():
                return p
    return None

def _first_data() -> Optional[Path]:
    for d in SEARCH_DIRS:
        for nm in CANDIDATE_FILES:
            p = d/nm
            if p.exists(): 
                return p
    return None

def _setup_azure_data():
    if MODE != "azure":
        return None, None, None, None
        
    try:
        conn_str = s("data.AZURE_STORAGE_CONNECTION_STRING")
        container = s("data.container")
        pq_blob = s("data.parquet_blob")
        csv_blob = s("data.csv_blob")
        attr_blob = s("data.attr_blob")
        logo_blob = s("data.logo_blob", None)

        if not conn_str or not container:
            st.error("Azure mode enabled but connection string or container is missing in secrets.")
            st.stop()

        from azure.storage.blob import BlobServiceClient
        svc = BlobServiceClient.from_connection_string(conn_str)
        cont = svc.get_container_client(container)

        TMP = Path("/tmp/influence_dl")
        TMP.mkdir(parents=True, exist_ok=True)

        def _dl(blob_name: str, filename: str) -> Path:
            dest = TMP / filename
            bc = cont.get_blob_client(blob_name)
            data = bc.download_blob().readall()
            dest.write_bytes(data)
            return dest

        data_parquet = _dl(pq_blob, "final_model_dataset.parquet") if pq_blob else None
        data_csv = _dl(csv_blob, "final_model_dataset.csv") if csv_blob else None
        attr_csv = _dl(attr_blob, "attribution_all_scored.csv") if attr_blob else None
        logo_path = _dl(logo_blob, "penta_logo.png") if logo_blob else LOGO_PATH

        return data_parquet, data_csv, attr_csv, logo_path

    except Exception as e:
        st.error(f"Azure init failed: {e}")
        st.stop()

@st.cache_resource(show_spinner=True)
def connect_duckdb_with_azure() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    if MODE == "azure":
        data_parquet, data_csv, attr_csv, logo_path = _setup_azure_data()
        global LOGO_PATH
        LOGO_PATH = logo_path
    else:
        data_parquet = _find_first_existing(PARQUET_NAME)
        data_csv = _find_first_existing(CSV_NAME)
        attr_csv = _find_first_existing(ATTR_NAME)

    try:
        if data_parquet and Path(data_parquet).exists():
            con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet({_q(data_parquet)})")
        elif data_csv and Path(data_csv).exists():
            con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto({_q(data_csv)}, IGNORE_ERRORS=TRUE)")
        else:
            found = _first_data()
            if found:
                if found.suffix.lower() == ".parquet":
                    con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet('{found.as_posix()}')")
                else:
                    con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto('{found.as_posix()}', IGNORE_ERRORS=TRUE)")
            else:
                st.error("Could not find dataset. Searched common folders.")
                st.warning("No dataset found. Upload a CSV or Parquet to continue.")
                con.execute("CREATE OR REPLACE VIEW v AS SELECT 1 as id WHERE 0")

        if attr_csv and Path(attr_csv).exists():
            con.execute(f"CREATE OR REPLACE VIEW v_attr AS SELECT * FROM read_csv_auto({_q(attr_csv)}, IGNORE_ERRORS=TRUE)")
        else:
            con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT 1 WHERE 0")

        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM v_attr WHERE kind='item'")
        con.execute("CREATE OR REPLACE VIEW v_term_attr AS SELECT * FROM v_attr WHERE kind='term'")

    except Exception as e:
        st.error(f"Error loading data: {e}")
        con.execute("CREATE OR REPLACE VIEW v AS SELECT 1 as id WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT 1 WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT 1 WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_term_attr AS SELECT 1 WHERE 0")

    return con

con = connect_duckdb_with_azure()

try:
    COLUMNS = set(con.execute("DESCRIBE v").fetchdf()["column_name"].tolist())
except Exception as e:
    st.error(f"Error describing table: {e}")
    COLUMNS = set()

select_parts = [
    "v.*",
    _cleaned_select(["author","author_name"], "author_clean", COLUMNS),
    _cleaned_select(["author_name"], "author_name_clean", COLUMNS),
    _cleaned_select(["publication_name","publication","source"], "publication_clean", COLUMNS),
    _cleaned_select(["channel","channel_name","source_type"], "channel_clean", COLUMNS),
    _cleaned_select(["channel_name","source_type"], "channel_name_clean", COLUMNS),
    _cleaned_select(["topic","topics"], "topic_clean", COLUMNS),
    _cleaned_select(["topics","topic"], "topics_clean", COLUMNS),
    _cleaned_select(["source_name","publication_name"], "source_name_clean", COLUMNS),
]
con.execute("CREATE OR REPLACE VIEW v_enriched AS SELECT " + ", ".join(select_parts) + " FROM v AS v")

# -------------------- Logo display --------------------
if LOGO_PATH and Path(LOGO_PATH).exists():
    with open(LOGO_PATH, "rb") as f:
        logo_b64 = base64.b64encode(f.read()).decode("utf-8")
    st.markdown(
        f"""
        <style>
        .header-bar {{
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1rem 0;
            border-bottom: 2px solid #E5F4F1;
        }}
        .penta-logo {{
            height: 60px;
            width: auto;
            margin-right: 20px;
        }}
        .header-title h1 {{
            margin: 0;
            color: #0A473B;
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
            font-weight: 600;
            font-size: 2.5rem;
            letter-spacing: -0.02em;
        }}
        .header-subtitle {{
            color: #12715D;
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: 0.25rem;
        }}
        </style>
        <div class="header-bar">
            <img src="data:image/png;base64,{logo_b64}" class="penta-logo"/>
            <div class="header-title">
                <h1>PolicyPath</h1>
                <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
            </div>
            <div style="margin-left: auto; display: flex; align-items: center; gap: 1rem;">
                <button class="dark-mode-toggle" onclick="toggleDarkMode()" title="Toggle Dark Mode">
                    🌙
                </button>
            </div>
        </div>
        
        <script>
        function toggleDarkMode() {{
            document.body.classList.toggle('dark-mode');
            localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
        }}
        
        // Load saved dark mode preference
        if (localStorage.getItem('darkMode') === 'true') {{
            document.body.classList.add('dark-mode');
        }}
        </script>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <style>
        .header-bar {
            display: flex;
            align-items: center;
            margin-bottom: 2rem;
            padding: 1rem 0;
            border-bottom: 2px solid #E5F4F1;
        }
        .header-title h1 {
            margin: 0;
            color: #0A473B;
            font-family: 'Inter', 'Helvetica Neue', Arial, sans-serif;
            font-weight: 600;
            font-size: 2.5rem;
            letter-spacing: -0.02em;
        }
        .header-subtitle {
            color: #12715D;
            font-size: 1.1rem;
            font-weight: 400;
            margin-top: 0.25rem;
        }
        </style>
        <div class="header-bar">
            <div class="header-title">
                <h1>PolicyPath</h1>
                <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# -------------------- Enhanced Sidebar --------------------
with st.sidebar:
    st.markdown("### 🚀 Quick Actions")
    
    # Quick search with suggestions
    quick_search = st.text_input("🔍 Quick Search", placeholder="Search publications, authors, terms...", key="sidebar_search")
    
    # Show quick search suggestions
    if quick_search and len(quick_search) >= 2:
        try:
            # Get suggestions from various columns
            columns_result = con.execute("DESCRIBE v").fetchdf()
            available_cols = columns_result['column_name'].tolist()
            
            # Search across multiple relevant columns
            searchable_cols = [col for col in available_cols if any(keyword in col.lower() for keyword in ['publication', 'author', 'headline', 'body', 'channel'])]
            
            if searchable_cols:
                suggestions_query = f"""
                SELECT DISTINCT 
                    CASE 
                        {' '.join([f"WHEN LOWER({col}) LIKE LOWER($search) THEN {col}" for col in searchable_cols[:3]])}
                        ELSE NULL
                    END as suggestion
                FROM v 
                WHERE {' OR '.join([f"LOWER({col}) LIKE LOWER($search)" for col in searchable_cols[:3]])}
                AND suggestion IS NOT NULL
                ORDER BY suggestion
                LIMIT 5
                """
                
                suggestions = con.execute(suggestions_query, {"search": f"%{quick_search}%"}).fetchdf()
                
                if not suggestions.empty:
                    st.markdown("**💡 Quick Suggestions:**")
                    for i, suggestion in enumerate(suggestions['suggestion'].tolist()[:5]):
                        if suggestion and len(str(suggestion).strip()) > 0:
                            suggestion_text = str(suggestion)[:25] + "..." if len(str(suggestion)) > 25 else str(suggestion)
                            if st.button(f"🔍 {suggestion_text}", 
                                       key=f"quick_suggestion_{i}", 
                                       help=f"Click to search for: {suggestion}"):
                                st.session_state.current_search = suggestion
                                st.rerun()
        except Exception as e:
            st.warning(f"Error getting quick suggestions: {str(e)}")
    
    if quick_search:
        add_to_recent_searches(quick_search)
        st.session_state.current_search = quick_search
    
    st.markdown("---")
    
    # Saved views
    if st.session_state.saved_views:
        st.markdown("### 💾 Saved Views")
        for view_name, view_data in st.session_state.saved_views.items():
            if st.button(f"📁 {view_name}", key=f"load_{view_name}"):
                st.session_state.current_view = view_name
                st.rerun()
    else:
        st.markdown("### 💾 Saved Views")
        st.info("No saved views yet. Create one from the main tabs!")
    
    st.markdown("---")
    
    # Favorites
    if st.session_state.favorites:
        st.markdown("### ⭐ Favorites")
        for fav in st.session_state.favorites[:5]:
            if st.button(f"⭐ {fav}", key=f"fav_{fav}"):
                st.session_state.current_search = fav
                st.rerun()
    else:
        st.markdown("### ⭐ Favorites")
        st.info("No favorites yet. Start searching to build your favorites!")
    
    st.markdown("---")
    
    # App info
    st.markdown("### ℹ️ About PolicyPath")
    st.markdown("""
    **Built by**: Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammmad Waqas, Mark Saba, Posy Olivetti
    """)

# -------------------- Main tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 PolicyPath", "🎯 Paths", "📊 Pulse", "🕸️ People"])

with tab1:
    st.markdown("""
    ## Welcome to PolicyPath
    
    **Your indispensable guide to healthcare policy influence**
    
    PolicyPath leverages Penta's data-driven approach to map how narratives travel through publications, authors, and channels. 
    Discover the key voices shaping U.S. healthcare policy and pinpoint the people and outlets driving influence.
    
    ### What makes PolicyPath different?
    
    **Data-First Intelligence**: Unlike competitors who only talk about data, Penta delivers actionable insights through comprehensive stakeholder analysis.
    
    **Comprehensive Coverage**: Track influence across publications, authors, channels, and policy terms to understand the complete narrative landscape.
    
    **Real-Time Analysis**: Monitor how policy narratives evolve and identify emerging voices before they become mainstream.
    
    ### Key Capabilities
    
    🎯 **Paths**: Analyze influence attribution by publication, author, channel, and policy terms  
    📊 **Pulse**: Monitor key performance indicators and narrative trends  
    🕸️ **People**: Visualize the network of relationships driving policy influence  
    
    *Built by Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammmad Waqas, Mark Saba, Posy Olivetti*
    *
    """)

with tab2:
    st.subheader("🎯 Paths - Attribution Analysis")
    st.markdown("""
    Discover the influence pathways in healthcare policy. Search for specific publications, authors, channels, or terms to understand their impact scores and attribution patterns.
    """)

    # Enhanced lookup section with smart search
    col1, col2 = st.columns([2, 1])
    
    with col1:
        lookup_type = st.radio("Search Type", ["Item Attribution", "Term Attribution"], horizontal=True, help="Choose between searching for specific items (publications, authors) or terms")
    
    with col2:
        if st.button("💾 Save Current View", help="Save your current search parameters"):
            view_name = st.text_input("View Name", key="save_view")
            if view_name:
                st.session_state.saved_views[view_name] = {
                    'lookup_type': lookup_type,
                    'timestamp': pd.Timestamp.now()
                }
                show_success_message(f"View '{view_name}' saved successfully!")

    # Show recent searches
    if st.session_state.recent_searches:
        with st.expander("🔍 Recent Searches"):
            for search in st.session_state.recent_searches[:5]:
                if st.button(f"🔍 {search}", key=f"recent_{search}"):
                    st.session_state.current_search = search

    if lookup_type == "Item Attribution":
        # Get available columns for item search
        try:
            columns_result = con.execute("DESCRIBE v").fetchdf()
            available_cols = columns_result['column_name'].tolist()
            
            # Filter for relevant columns
            searchable_cols = [col for col in available_cols if any(keyword in col.lower() for keyword in ['publication', 'author', 'channel', 'publisher'])]
            
            if not searchable_cols:
                st.warning("No searchable columns found in the dataset.")
            else:
                col1, col2 = st.columns(2)
                
                with col1:
                    sel_col = st.selectbox("Search by", searchable_cols, help="Choose what type of item to search for")
                
                with col2:
                    search_term = st.text_input("Search term", placeholder=f"Enter {sel_col} to search...", help="Type to search for specific items", key=f"search_{sel_col}")
                
                # Real-time search suggestions
                if search_term and len(search_term) >= 2:
                    try:
                        # Get suggestions as user types
                        suggestions_query = f"SELECT DISTINCT {sel_col} FROM v WHERE LOWER({sel_col}) LIKE LOWER($search) AND {sel_col} IS NOT NULL ORDER BY {sel_col} LIMIT 10"
                        suggestions = con.execute(suggestions_query, {"search": f"%{search_term}%"}).fetchdf()
                        
                        if not suggestions.empty:
                            st.markdown("**💡 Suggestions:**")
                            suggestion_cols = st.columns(min(3, len(suggestions)))
                            
                            for i, suggestion in enumerate(suggestions[sel_col].tolist()[:9]):  # Show max 9 suggestions
                                col_idx = i % 3
                                with suggestion_cols[col_idx]:
                                    if st.button(f"🔍 {suggestion[:30]}{'...' if len(suggestion) > 30 else ''}", 
                                               key=f"suggestion_{i}_{sel_col}", 
                                               help=f"Click to search for: {suggestion}"):
                                        st.session_state[f"selected_{sel_col}"] = suggestion
                                        st.rerun()
                            
                            st.markdown("---")
                    except Exception as e:
                        st.warning(f"Error getting suggestions: {str(e)}")
                
                # Use selected suggestion or search term
                if f"selected_{sel_col}" in st.session_state:
                    search_term = st.session_state[f"selected_{sel_col}"]
                    st.success(f"Selected: {search_term}")
                    if st.button("Clear Selection", key=f"clear_{sel_col}"):
                        del st.session_state[f"selected_{sel_col}"]
                        st.rerun()
                
                if search_term:
                    # Add to recent searches
                    add_to_recent_searches(f"{sel_col}: {search_term}")
                    
                    # Search for matching values
                    try:
                        search_query = f"SELECT DISTINCT {sel_col} FROM v WHERE LOWER({sel_col}) LIKE LOWER($search) AND {sel_col} IS NOT NULL ORDER BY {sel_col} LIMIT 20"
                        matches = con.execute(search_query, {"search": f"%{search_term}%"}).fetchdf()
                        
                        if not matches.empty:
                            st.success(f"Found {len(matches)} matches for '{search_term}' in {sel_col}")
                            
                            # Let user select from matches
                            selected_item = st.selectbox("Select item", matches[sel_col].tolist(), key=f"select_{sel_col}")
                            
                            if selected_item:
                                # Show data for selected item
                                item_data = con.execute(f"SELECT * FROM v WHERE {sel_col} = $item LIMIT 100", {"item": selected_item}).fetchdf()
                                
                                if not item_data.empty:
                                    st.markdown(f"### 📊 Data for: {selected_item}")
                                    
                                    # Show key metrics
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        create_metric_card("Records", len(item_data), icon="📄")
                                    with col2:
                                        if 'circulation_size' in item_data.columns:
                                            avg_circ = item_data['circulation_size'].mean()
                                            create_metric_card("Avg Circulation", f"{avg_circ:,.0f}", icon="📈")
                                    with col3:
                                        if 'body_token_count' in item_data.columns:
                                            avg_tokens = item_data['body_token_count'].mean()
                                            create_metric_card("Avg Tokens", f"{avg_tokens:,.0f}", icon="📝")
                                    
                                    # Show the data
                                    st.dataframe(item_data, use_container_width=True, height=400)
                                    
                                    # Export options
                                    col1, col2 = st.columns(2)
                                    with col1:
                                        export_data_button(item_data, f"{sel_col}_{selected_item}", "csv")
                                    with col2: 
                                        export_data_button(item_data, f"{sel_col}_{selected_item}", "xlsx")
                                else:
                                    st.warning("No data found for the selected item.")
                        else:
                            st.warning(f"No matches found for '{search_term}' in {sel_col}")
                            
                    except Exception as e:
                        st.error(f"Error searching: {str(e)}")
                        st.info("Please try a different search term or column.")
        except Exception as e:
            st.error(f"Error accessing data: {str(e)}")
            st.info("Please check your data connection.")
    else:
        st.markdown("### 🔍 Term Search")
        term = st.text_input("Type a term to search", placeholder="Enter a policy term or keyword...", help="Search for specific terms in headlines and content", key="term_search")
        
        # Real-time term suggestions
        if term and len(term) >= 2:
            try:
                # Get text columns for suggestions
                columns_result = con.execute("DESCRIBE v").fetchdf()
                available_cols = columns_result['column_name'].tolist()
                text_columns = [col for col in available_cols if any(keyword in col.lower() for keyword in ['headline', 'body', 'content', 'text'])]
                
                if text_columns:
                    # Get unique terms that contain the search term
                    suggestions_query = f"""
                    SELECT DISTINCT 
                        CASE 
                            WHEN LOWER({text_columns[0]}) LIKE LOWER($search) THEN {text_columns[0]}
                            WHEN LOWER({text_columns[1] if len(text_columns) > 1 else text_columns[0]}) LIKE LOWER($search) THEN {text_columns[1] if len(text_columns) > 1 else text_columns[0]}
                        END as suggestion
                    FROM v 
                    WHERE {' OR '.join([f"LOWER({col}) LIKE LOWER($search)" for col in text_columns])}
                    AND suggestion IS NOT NULL
                    ORDER BY suggestion
                    LIMIT 10
                    """
                    
                    suggestions = con.execute(suggestions_query, {"search": f"%{term}%"}).fetchdf()
                    
                    if not suggestions.empty:
                        st.markdown("**💡 Term Suggestions:**")
                        suggestion_cols = st.columns(min(3, len(suggestions)))
                        
                        for i, suggestion in enumerate(suggestions['suggestion'].tolist()[:9]):
                            if suggestion and len(str(suggestion).strip()) > 0:
                                col_idx = i % 3
                                with suggestion_cols[col_idx]:
                                    suggestion_text = str(suggestion)[:40] + "..." if len(str(suggestion)) > 40 else str(suggestion)
                                    if st.button(f"🔍 {suggestion_text}", 
                                               key=f"term_suggestion_{i}", 
                                               help=f"Click to search for: {suggestion}"):
                                        st.session_state["selected_term"] = suggestion
                                        st.rerun()
                        
                        st.markdown("---")
            except Exception as e:
                st.warning(f"Error getting term suggestions: {str(e)}")
        
        # Use selected suggestion or search term
        if "selected_term" in st.session_state:
            term = st.session_state["selected_term"]
            st.success(f"Selected term: {term}")
            if st.button("Clear Term Selection", key="clear_term"):
                del st.session_state["selected_term"]
                st.rerun()
        
        if term:
            # Add to recent searches
            add_to_recent_searches(f"Term: {term}")
            
            try:
                # Search for term in text content
                text_columns = []
                try:
                    columns_result = con.execute("DESCRIBE v").fetchdf()
                    available_cols = columns_result['column_name'].tolist()
                    text_columns = [col for col in available_cols if any(keyword in col.lower() for keyword in ['headline', 'body', 'content', 'text'])]
                except:
                    text_columns = ['headline', 'body']
                
                if text_columns:
                    # Build search query across text columns
                    search_conditions = []
                    for col in text_columns:
                        search_conditions.append(f"LOWER({col}) LIKE LOWER($search)")
                    
                    search_query = f"SELECT * FROM v WHERE {' OR '.join(search_conditions)} LIMIT 100"
                    hits = con.execute(search_query, {"search": f"%{term}%"}).fetchdf()
                    
                    if not hits.empty:
                        st.success(f"Found {len(hits)} articles containing '{term}'")
                        
                        # Show term frequency analysis
                        st.markdown("### 📊 Term Analysis")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            create_metric_card("Total Matches", len(hits), icon="🔍")
                        
                        with col2:
                            if 'circulation_size' in hits.columns:
                                total_reach = hits['circulation_size'].sum()
                                create_metric_card("Total Reach", f"{total_reach:,.0f}", icon="📈")
                        
                        with col3:
                            if 'load_date' in hits.columns:
                                date_range = hits['load_date'].nunique()
                                create_metric_card("Date Range", f"{date_range} days", icon="📅")
                        
                        # Show sample of results
                        st.markdown("### 📄 Sample Results")
                        st.dataframe(hits, use_container_width=True, height=400)
                        
                        # Export options
                        col1, col2 = st.columns(2)
                        with col1:
                            export_data_button(hits, f"term_search_{term}", "csv")
                        with col2:
                            export_data_button(hits, f"term_search_{term}", "json")
                    else:
                        st.warning(f"No articles found containing '{term}'")
                        st.info("Try a different search term or check your spelling.")
                else:
                    st.warning("No searchable text columns found in the dataset.")
                    
            except Exception as e:
                st.error(f"Error searching for term: {str(e)}")
                st.info("Please try a different search term.")

with tab3:
    st.subheader("📊 Pulse - Real-time Analytics")
    st.markdown("""
    Monitor the pulse of healthcare policy influence with real-time analytics, interactive visualizations, and comprehensive KPI tracking.
    """)

    # Enhanced dashboard with real-time feel
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🎛️ Smart Filters")
    
    with col2:
        if st.button("🔄 Refresh Data", help="Refresh all data and visualizations"):
            st.rerun()
    
    with col3:
        auto_refresh = st.checkbox("🔄 Auto-refresh", help="Automatically refresh data every 30 seconds")
        if auto_refresh:
            st.markdown('<div class="pulse-animation">🔄 Auto-refresh enabled</div>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    
    with col1:
        date_bounds_result = con.execute("SELECT MIN(CAST(load_date AS TIMESTAMP)), MAX(CAST(load_date AS TIMESTAMP)) FROM v").fetchone()
        if date_bounds_result and date_bounds_result[0] and date_bounds_result[1]:
            date_range = st.date_input(
                "Date range",
                value=(pd.to_datetime(date_bounds_result[0]).date(), pd.to_datetime(date_bounds_result[1]).date()),
                min_value=pd.to_datetime(date_bounds_result[0]).date(),
                max_value=pd.to_datetime(date_bounds_result[1]).date()
            )
        else:
            date_range = None
        
        # Get available columns for filtering
        try:
            columns_result = con.execute("DESCRIBE v").fetchdf()
            available_cols = columns_result['column_name'].tolist()
            
            # Publications filter
            pub_cols = [col for col in available_cols if 'publication' in col.lower()]
            if pub_cols:
                pub_col = pub_cols[0]
                pubs = con.execute(f"SELECT DISTINCT {pub_col} FROM v WHERE {pub_col} IS NOT NULL ORDER BY 1 LIMIT 50").fetchdf()[pub_col].tolist()
                sel_pubs = st.multiselect("Publications", pubs, default=[])
            else:
                sel_pubs = []
            
            # Channels filter
            channel_cols = [col for col in available_cols if 'channel' in col.lower()]
            if channel_cols:
                channel_col = channel_cols[0]
                channels = con.execute(f"SELECT DISTINCT {channel_col} FROM v WHERE {channel_col} IS NOT NULL ORDER BY 1 LIMIT 50").fetchdf()[channel_col].tolist()
                sel_channels = st.multiselect("Channels", channels, default=[])
            else:
                sel_channels = []
        except Exception as e:
            st.warning(f"Error loading filter options: {str(e)}")
            sel_pubs = []
            sel_channels = []
    
    with col2:
        try:
            # Authors filter
            author_cols = [col for col in available_cols if 'author' in col.lower()]
            if author_cols:
                author_col = author_cols[0]
                authors = con.execute(f"SELECT DISTINCT {author_col} FROM v WHERE {author_col} IS NOT NULL ORDER BY 1 LIMIT 50").fetchdf()[author_col].tolist()
                sel_authors = st.multiselect("Authors", authors, default=[])
            else:
                sel_authors = []
            
            # Source type filter
            source_cols = [col for col in available_cols if 'source' in col.lower()]
            if source_cols:
                source_col = source_cols[0]
                sources = con.execute(f"SELECT DISTINCT {source_col} FROM v WHERE {source_col} IS NOT NULL ORDER BY 1 LIMIT 20").fetchdf()[source_col].tolist()
                sel_sources = st.multiselect("Source Types", sources, default=[])
            else:
                sel_sources = []
        except Exception as e:
            st.warning(f"Error loading author/source filters: {str(e)}")
            sel_authors = []
            sel_sources = []
    
    # Build where clause
    w, args = where_from_filters(date_range, sel_pubs, sel_channels, [], sel_authors, [])
    
    # KPI Metrics
    try:
        # Get available columns for metrics
        columns_result = con.execute("DESCRIBE v").fetchdf()
        available_cols = columns_result['column_name'].tolist()
        
        # Total publications
        pub_cols = [col for col in available_cols if 'publication' in col.lower()]
        if pub_cols:
            pub_col = pub_cols[0]
            total_pubs = con.execute(f"SELECT COUNT(DISTINCT {pub_col}) FROM v WHERE {w}", args).fetchone()[0]
        else:
            total_pubs = 0
        
        # Unique sources
        source_cols = [col for col in available_cols if 'source' in col.lower()]
        if source_cols:
            source_col = source_cols[0]
            uniq_sources = con.execute(f"SELECT COUNT(DISTINCT {source_col}) FROM v WHERE {w}", args).fetchone()[0]
        else:
            uniq_sources = 0
        
        # Unique authors
        author_cols = [col for col in available_cols if 'author' in col.lower()]
        if author_cols:
            author_col = author_cols[0]
            uniq_authors = con.execute(f"SELECT COUNT(DISTINCT {author_col}) FROM v WHERE {w}", args).fetchone()[0]
        else:
            uniq_authors = 0
            
    except Exception as e:
        st.warning(f"Error calculating metrics: {str(e)}")
        total_pubs = 0
        uniq_sources = 0
        uniq_authors = 0
    
    infl_col = "pub_credit_share" if "pub_credit_share" in COLUMNS else ("credit_share" if "credit_share" in COLUMNS else None)
    avg_infl = con.execute(f"SELECT AVG({infl_col}) FROM v WHERE {w}", args).fetchone()[0] if infl_col else None

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total publications", f"{total_pubs:,}")
    m2.metric("Avg influence score", f"{avg_infl:.3f}" if avg_infl is not None else "n/a")
    m3.metric("Unique sources", f"{uniq_sources:,}")
    m4.metric("Unique authors", f"{uniq_authors:,}")
    
    st.divider()

    # Charts
    cat_cols = [c for c in ["publication_name","source_name","channel_name","author_name","topic","sentiment_band"] if c in COLUMNS]
    
    if not cat_cols:
        st.info("No categorical columns to group by.")
    else:
        dim = st.selectbox("Group charts by", cat_cols, index=0)

        circ_col = next((c for c in ["circulation","circulation_size","reach","impressions","audience"] if c in COLUMNS), None)
        circ_sql = f"COALESCE(SUM({circ_col}),0)" if circ_col else "COUNT(*)"
        
        agg = con.execute(f"""
            SELECT {dim} AS dim,
                   AVG({infl_col}) AS avg_influence,
                   COUNT(*) AS n,
                   {circ_sql} AS total_metric
            FROM v WHERE {w}
            GROUP BY 1 HAVING dim IS NOT NULL
        """, args).fetchdf()

        top_n = st.slider("Top N", 5, 50, 20, 1)

        cA, cB = st.columns(2)
        if not agg.empty:
            b1 = alt.Chart(agg.sort_values("avg_influence", ascending=False).head(top_n)).mark_bar(color="#12715D").encode(
                y=alt.Y("dim:N", sort="-x", title=None),
                x=alt.X("avg_influence:Q", title="Avg influence"),
                tooltip=["dim", alt.Tooltip("avg_influence:Q", format=".3f"), "n"],
            )
            b2 = alt.Chart(agg.sort_values("n", ascending=False).head(top_n)).mark_bar(color="#4AB48E").encode(
                y=alt.Y("dim:N", sort="-x", title=None),
                x=alt.X("n:Q", title="Count"),
                tooltip=["dim", "n", alt.Tooltip("avg_influence:Q", format=".3f")],
            )
            cA.altair_chart(b1.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
            cB.altair_chart(b2.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
        else:
            st.info("No data for current filters.")

        # Pie Chart
        if infl_col and not agg.empty:
            pie_df = agg.sort_values("avg_influence", ascending=False).head(20)
            fig_pie = px.pie(
                pie_df, names="dim", values="avg_influence",
                color_discrete_sequence=["#12715D", "#4AB48E", "#CFECE4", "#E7F6F1"]
            )
            fig_pie.update_traces(textinfo="percent+label", pull=[0.02]*len(pie_df))
            fig_pie.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()

    # Sankey Diagram
    if not cat_cols:
        pass
    else:
        left, right = st.columns(2)
        src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
        tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols) - 1), key="sank_tgt")

        c1, c2, c3, c4 = st.columns(4)
        top_sources = c1.slider("Top Sources", 3, 50, 15, 1)
        top_targets = c2.slider("Top Targets", 2, 20, 6, 1)
        max_links = c3.slider("Max Links", 10, 500, 120, 10)
        bucket_other = c4.checkbox("Bucket 'Other'", value=True)

        if src != tgt:
            # Rank to keep only the most common nodes
            src_rank = con.execute(
                f"SELECT {src} AS s, COUNT(*) AS n FROM v WHERE {w} AND {src} IS NOT NULL "
                f"GROUP BY 1 ORDER BY n DESC LIMIT {int(top_sources)}", args
            ).fetchdf()
            tgt_rank = con.execute(
                f"SELECT {tgt} AS t, COUNT(*) AS n FROM v WHERE {w} AND {tgt} IS NOT NULL "
                f"GROUP BY 1 ORDER BY n DESC LIMIT {int(top_targets)}", args
            ).fetchdf()
            keep_s = set(src_rank["s"].dropna().astype(str))
            keep_t = set(tgt_rank["t"].dropna().astype(str))

            sdata = con.execute(
                f"""
                SELECT
                CASE WHEN {src} IN $ks THEN {src} ELSE 'Other' END AS s,
                CASE WHEN {tgt} IN $kt THEN {tgt} ELSE 'Other' END AS t,
                COUNT(*) AS v
                FROM v
                WHERE {w} AND {src} IS NOT NULL AND {tgt} IS NOT NULL
                GROUP BY 1,2
                ORDER BY v DESC
                LIMIT {int(max_links)}
                """,
                {**args, "ks": list(keep_s), "kt": list(keep_t)}
            ).fetchdf()

            if not bucket_other:
                sdata = sdata[(sdata["s"].isin(keep_s)) & (sdata["t"].isin(keep_t))]

            if not sdata.empty:
                # Build nodes/links with truncated display labels
                nodes_all = pd.Series(pd.concat([sdata["s"], sdata["t"]])).astype(str).unique().tolist()
                labels_short = [shorten(n) for n in nodes_all]
                index = {n: i for i, n in enumerate(nodes_all)}

                hc = st.checkbox("High-contrast labels", value=True, help="Use light nodes with dark borders + larger text")

                node_fill = "#CFECE4" if hc else "#12715D"
                node_border = "#12715D" if hc else "white"
                link_rgba = "rgba(18,113,93,0.22)" if hc else "rgba(18,113,93,0.35)"
                font_color = "#133C35"
                font_size = 16 if hc else 15

                fig = go.Figure(go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=labels_short,
                        pad=26,
                        thickness=22,
                        color=[node_fill] * len(nodes_all),
                        line=dict(color="rgba(0,0,0,0)", width=0)
                    ),
                    link=dict(
                        source=[index[s] for s in sdata["s"]],
                        target=[index[t] for t in sdata["t"]],
                        value=sdata["v"],
                        color=link_rgba,
                        hovertemplate=(
                            "Count: %{value:,}"
                            "<br>source: %{source.label}"
                            "<br>target: %{target.label}<extra></extra>"
                        )
                    )
                ))

                fig.update_layout(
                    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=font_size, color=font_color),
                    hoverlabel=dict(font_size=13, font_family="Inter, Helvetica, Arial, sans-serif"),
                    margin=dict(l=8, r=8, t=6, b=6),
                    height=640
                )

                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for Sankey with these fields.")
        else:
            st.info("Choose different fields for source and target.")
    
    st.divider()

    # Time Series
    time_col = "conversion_ts" if "conversion_ts" in COLUMNS else ("load_date" if "load_date" in COLUMNS else ("source_time" if "source_time" in COLUMNS else None))
    conv_col = "conversions" if "conversions" in COLUMNS else None
    
    if time_col:
        conv = con.execute(
            f"""
            SELECT DATE_TRUNC('day', CAST({time_col} AS TIMESTAMP)) AS d,
                {('SUM(' + conv_col + ')') if conv_col else 'COUNT(*)'} AS y
            FROM v WHERE {w}
            GROUP BY 1 ORDER BY 1
            """,
            args
        ).fetchdf()
        
        if not conv.empty:
            conv_chart = (
                alt.Chart(conv)
                .mark_bar(color="#12715D")
                .encode(x=alt.X("d:T", title="Date"), y=alt.Y("y:Q", title="Conversions"))
                .properties(height=300)
                .interactive()
            )
            st.altair_chart(conv_chart.configure_view(strokeWidth=0), use_container_width=True)
    
    st.divider()

    # Enhanced Sample Data with Export Options
    st.markdown("### 📊 Filtered Data Sample")
    
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        sample_size = st.slider("Sample Size", 100, 5000, 1000, help="Number of rows to display")
    
    with col2:
        if st.button("📥 Export CSV", help="Download filtered data as CSV"):
            sample = con.execute(f"SELECT * FROM v WHERE {w} LIMIT {sample_size}", args).fetchdf()
            export_data_button(sample, "filtered_data", "csv")
    
    with col3:
        if st.button("📊 Export JSON", help="Download filtered data as JSON"):
            sample = con.execute(f"SELECT * FROM v WHERE {w} LIMIT {sample_size}", args).fetchdf()
            export_data_button(sample, "filtered_data", "json")
    
    # Display data with enhanced styling
    sample = con.execute(f"SELECT * FROM v WHERE {w} LIMIT {sample_size}", args).fetchdf()
    
    if not sample.empty:
        st.markdown(f"**Showing {len(sample)} rows**")
        st.dataframe(sample, use_container_width=True, height=400)
        
        # Data insights
        with st.expander("🔍 Data Insights"):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                create_metric_card("Total Rows", f"{len(sample):,}", icon="📊")
            
            with col2:
                numeric_cols = sample.select_dtypes(include=['number']).columns
                create_metric_card("Numeric Columns", len(numeric_cols), icon="🔢")
            
            with col3:
                text_cols = sample.select_dtypes(include=['object']).columns
                create_metric_card("Text Columns", len(text_cols), icon="📝")
    else:
        st.warning("No data found for the selected filters.")

with tab4:
    st.subheader("🕸️ People - Network Intelligence")
    st.markdown("""
    Explore the intricate web of relationships that drive healthcare policy influence. Visualize connections between publications, authors, channels, and terms to understand the network dynamics.
    """)

    # TODO: Add network analysis features here when you have network data
    # Place network_edges.csv in data/ directory with columns: source, target, weight

    # Enhanced network controls
    col1, col2, col3 = st.columns([2, 1, 1])
    
    with col1:
        st.markdown("### 🌐 Network Controls")
    
    with col2:
        network_layout = st.selectbox("Layout", ["Force-directed", "Hierarchical", "Circular"], help="Choose network visualization layout")
    
    with col3:
        show_labels = st.checkbox("Show Labels", value=True, help="Display node labels")
    
    # Network analysis options
    col1, col2 = st.columns(2)
    
    with col1:
        min_connections = st.slider("Minimum Connections", 1, 50, 5, help="Filter nodes by minimum number of connections")
    
    with col2:
        max_nodes = st.slider("Max Nodes to Display", 10, 200, 50, help="Limit number of nodes for performance")
    
    # Find network edges file
    edges_path = None
    for d in [ROOT / "data", ROOT / "data" / "processed", ROOT / "processed", APP_DIR]:
        p = d / "network_edges.csv"
        if p.exists():
            edges_path = p
            break
    
    if edges_path and edges_path.exists():
        edges = pd.read_csv(edges_path)
        required = {"source", "target", "weight"}
        
        if required.issubset(set(edges.columns)):
            st.write("Edges sample:")
            st.dataframe(edges.head(200), use_container_width=True)

            st.markdown("**Basic node strength**")
            deg = pd.concat(
                [
                    edges.groupby("source")["weight"].sum().rename("out_weight"),
                    edges.groupby("target")["weight"].sum().rename("in_weight"),
                ],
                axis=1,
            ).fillna(0)
            deg["strength"] = deg["in_weight"] + deg["out_weight"]
            st.dataframe(deg.sort_values("strength", ascending=False).head(30))

            # TODO: Add centrality metrics and community detection here

            min_w = float(edges["weight"].quantile(0.75)) if not edges.empty else 0.0
            min_w = st.slider(
                "Min edge weight to show",
                float(edges["weight"].min()) if not edges.empty else 0.0,
                float(edges["weight"].max()) if not edges.empty else 1.0,
                min_w,
            )
            sub = edges[edges["weight"] >= min_w].copy()
            st.write(f"Filtered edges: {len(sub):,} (of {len(edges):,})")
            st.dataframe(sub.head(200), use_container_width=True)

            st.caption("Tip: For interactive graphs, consider pyvis or Plotly (left out for minimal deps).")

            # TODO: Add interactive network visualization here
        else:
            st.warning("network_edges.csv found but must contain columns: source, target, weight")
    else:
        st.info("No network_edges.csv found. Place one under data/ or data/processed with columns: source,target,weight.")

    # TODO: Add temporal network analysis here when you have time-series network data
