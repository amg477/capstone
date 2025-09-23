# app.py — Influence Explorer (Tabbed: Attribution • Dashboard • Network)
# -----------------------------------------------------------------------
# - Python 3.9 compatible
# - Local & Streamlit Cloud friendly (optional Azure via secrets)
# - DuckDB backend, fast filters, simple UI
#
# Secrets (put these in Deploy → Settings → Secrets as TOML):
#
# [data]
# mode = "azure"               # or "local" (default if omitted)
#
# # If mode="azure"
# AZURE_STORAGE_CONNECTION_STRING = "DefaultEndpointsProtocol=...;AccountName=...;AccountKey=...;EndpointSuffix=core.windows.net"
# container     = "capstone"
# parquet_blob  = "data/processed/final_model_dataset.parquet"   # preferred
# csv_blob      = "data/processed/final_model_dataset.csv"       # fallback
# attr_blob     = "data/processed/attribution_all_scored.csv"    # optional
# # logo_blob   = "final_deliverable/penta_logo.png"             # optional
#
# # If mode="local" (files committed in repo)
# data_dir = "data/processed"
# parquet  = "final_model_dataset.parquet"
# csv      = "final_model_dataset.csv"
# attr_csv = "attribution_all_scored.csv"
# logo     = "final_deliverable/penta_logo.png"

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import duckdb
import pandas as pd
import streamlit as st

# Optional brand styling for charts
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# -------------------- Page config --------------------
st.set_page_config(page_title="Influence Explorer", layout="wide")

# -------------------- Brand styling --------------------
def apply_penta_style():
    """Apply Penta Group brand styling to matplotlib/seaborn plots."""
    PRIMARY_GREEN = "#12715D"   # Deep green
    ACCENT_GREEN  = "#4AB48E"   # Mint accent

    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Georgia', 'Times New Roman', 'serif']
    rcParams['font.size'] = 12
    rcParams['axes.titlesize'] = 14
    rcParams['axes.labelsize'] = 12

    sns.set_style("whitegrid")
    sns.set_palette([PRIMARY_GREEN, ACCENT_GREEN, "#E5F4F1", "#C8EADF"])

    rcParams['axes.edgecolor']  = PRIMARY_GREEN
    rcParams['axes.labelcolor'] = PRIMARY_GREEN
    rcParams['xtick.color']     = PRIMARY_GREEN
    rcParams['ytick.color']     = PRIMARY_GREEN
    rcParams['text.color']      = PRIMARY_GREEN
    rcParams['grid.color']      = "#E0E0E0"
    rcParams['axes.titleweight']= 'bold'

    rcParams['figure.facecolor'] = "white"
    rcParams['axes.facecolor']   = "white"

# -------------------- Secrets helpers --------------------
def s(path: str, default=None):
    """
    Safe nested get from st.secrets using 'section.key' (e.g., 'data.data_dir').
    Returns default if not found.
    """
    try:
        cur = st.secrets
        for part in path.split("."):
            cur = cur[part]
        return cur
    except Exception:
        return default

# -------------------- Azure-aware path resolution --------------------
APP_DIR = Path(__file__).resolve().parent
ROOT    = APP_DIR.parent  # one level up from final_deliverable/

MODE = (s("data.mode", "local") or "local").lower()

def _q(p: Path) -> str:
    return "'" + str(p).replace("'", "''") + "'"

# Default local mode config (also used as fallback if Azure missing)
DATA_DIR     = (ROOT / s("data.data_dir", "data")).resolve()
PARQUET_NAME = s("data.parquet", "final_model_dataset.parquet")
CSV_NAME     = s("data.csv",     "final_model_dataset.csv")
ATTR_NAME    = s("data.attr_csv","attribution_all_scored.csv")

# Candidate dirs to search in local mode
CANDIDATE_DIRS: List[Path] = [DATA_DIR, ROOT / "data", ROOT / "data" / "processed"]

def _find_first_existing(*names: str) -> Optional[Path]:
    for d in CANDIDATE_DIRS:
        for nm in names:
            p = d / nm
            if p.exists():
                return p
    return None

DATA_PARQUET = _find_first_existing(PARQUET_NAME)
DATA_CSV     = _find_first_existing(CSV_NAME)
ATTR_CSV     = _find_first_existing(ATTR_NAME)

# Logo (optional, local default)
LOGO_PATH = (ROOT / s("data.logo", "final_deliverable/penta_logo.png"))
if not LOGO_PATH.exists() and (APP_DIR / "penta_logo.png").exists():
    LOGO_PATH = APP_DIR / "penta_logo.png"

# If Azure mode is enabled, override by downloading blobs to /tmp
if MODE == "azure":
    try:
        from azure.storage.blob import BlobServiceClient
        conn_str  = s("data.AZURE_STORAGE_CONNECTION_STRING")
        container = s("data.container")
        pq_blob   = s("data.parquet_blob")  # preferred
        csv_blob  = s("data.csv_blob")
        attr_blob = s("data.attr_blob")
        logo_blob = s("data.logo_blob", None)

        if not conn_str or not container:
            st.error("Azure mode enabled but connection string or container is missing in secrets.")
            st.stop()

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

        DATA_PARQUET = _dl(pq_blob,  "final_model_dataset.parquet") if pq_blob  else None
        DATA_CSV     = _dl(csv_blob, "final_model_dataset.csv")     if csv_blob else None
        ATTR_CSV     = _dl(attr_blob, "attribution_all_scored.csv") if attr_blob else None
        LOGO_PATH    = _dl(logo_blob, "penta_logo.png")             if logo_blob else LOGO_PATH

    except Exception as e:
        st.error(f"Azure init failed: {e}")
        st.stop()

# -------------------- Utilities --------------------
def _quote_ident(name: str) -> str:
    """DuckDB-safe identifier quoting."""
    return '"' + name.replace('"', '""') + '"'

# -------------------- Connect & create views --------------------
@st.cache_resource(show_spinner=True)
def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    # main table v
    if DATA_PARQUET and Path(DATA_PARQUET).exists():
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet({_q(DATA_PARQUET)})")
    elif DATA_CSV and Path(DATA_CSV).exists():
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto({_q(DATA_CSV)}, IGNORE_ERRORS=TRUE)")
    else:
        st.error(
            "Could not find dataset. Searched:\n"
            + "\n".join(f"- {d}" for d in CANDIDATE_DIRS)
            + "\nLooking for: "
            + f"{PARQUET_NAME} or {CSV_NAME}"
        )
        raise FileNotFoundError("Dataset not found in expected folders.")

    # optional attribution
    if ATTR_CSV and Path(ATTR_CSV).exists():
        con.execute(f"CREATE OR REPLACE VIEW v_attr AS SELECT * FROM read_csv_auto({_q(ATTR_CSV)}, IGNORE_ERRORS=TRUE)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM v_attr WHERE kind='item'")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT * FROM v_attr WHERE kind='term'")
    else:
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT * FROM (SELECT 1 WHERE 0)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM (SELECT 1 WHERE 0)")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT * FROM (SELECT 1 WHERE 0)")

    return con

# -------------------- Attribution explainer --------------------
def explain_attribution(row: pd.Series, universe: Optional[pd.DataFrame] = None) -> str:
    dim   = str(row.get("dimension", "dimension"))
    val   = str(row.get("value", "value"))
    cred  = float(row.get("credit", 0.0))
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

# -------------------- Bootstrap, style, header --------------------
con = connect_duckdb()
apply_penta_style()

if LOGO_PATH and Path(LOGO_PATH).exists():
    st.image(str(LOGO_PATH), width=200)

st.title("Influence Explorer")

# -------------------- Sidebar filters --------------------
st.sidebar.header("Filters")

# Date range (if load_ts exists)
try:
    min_ts, max_ts = con.execute("SELECT MIN(load_ts), MAX(load_ts) FROM v").fetchone()
except Exception:
    min_ts = max_ts = None

if min_ts and max_ts:
    dmin = pd.to_datetime(min_ts).date()
    dmax = pd.to_datetime(max_ts).date()
    date_range = st.sidebar.date_input("Load date range", value=(dmin, dmax))
else:
    date_range = None

# Sentiment band options
try:
    bands = con.execute("SELECT DISTINCT sentiment_band FROM v WHERE sentiment_band IS NOT NULL ORDER BY 1").fetchdf()["sentiment_band"].tolist()
except Exception:
    bands = []
sel_bands = st.sidebar.multiselect("Sentiment band", bands, default=bands)

# Publications
try:
    pubs = con.execute("SELECT DISTINCT publication_name FROM v WHERE publication_name IS NOT NULL ORDER BY 1").fetchdf()["publication_name"].tolist()
except Exception:
    pubs = []
sel_pubs = st.sidebar.multiselect("Publication", pubs, default=[])

row_limit = st.sidebar.number_input("Rows to display", 50, 50000, 2000, 50)

def build_where(extra: str = "", params: Optional[Dict] = None) -> Tuple[str, Dict]:
    clauses: List[str] = ["1=1"]
    args: Dict = {} if params is None else dict(params)

    if date_range and isinstance(date_range, (tuple, list)) and len(date_range) == 2 and date_range[0] and date_range[1]:
        dmin = pd.to_datetime(date_range[0])
        dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        clauses.append("load_ts >= $dmin AND load_ts < $dmax")
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

# Debug block (optional—remove later)
with st.sidebar.expander("Debug: data paths"):
    st.write("MODE:", MODE)
    st.write("DATA_PARQUET:", DATA_PARQUET)
    st.write("DATA_CSV:", DATA_CSV)
    st.write("ATTR_CSV:", ATTR_CSV)
    st.write("LOGO_PATH:", LOGO_PATH)
    st.write("Searched dirs:", [str(d) for d in CANDIDATE_DIRS])

# -------------------- Tabs --------------------
attr_tab, dash_tab, net_tab = st.tabs(["🧮 Attribution", "📊 Dashboard", "🕸️ Network"])

# ==========================
# TAB 1 — ATTRIBUTION
# ==========================
with attr_tab:
    st.subheader("Attribution Lookups")

    lookup_type = st.radio("Lookup type", ["Item", "Term"], horizontal=True)

    if lookup_type == "Item":
        dims = con.execute("SELECT DISTINCT dimension FROM v_item_attr WHERE dimension IS NOT NULL ORDER BY 1").fetchdf()
        all_dims = dims["dimension"].tolist()
        if not all_dims:
            st.info("No item attribution available.")
        else:
            default_idx = all_dims.index("publication_name") if "publication_name" in all_dims else 0
            sel_dim = st.selectbox("Dimension", all_dims, index=default_idx)
            values = con.execute(
                "SELECT DISTINCT value FROM v_item_attr WHERE dimension=$d AND value IS NOT NULL ORDER BY 1",
                {"d": sel_dim},
            ).fetchdf()["value"].tolist()
            if not values:
                st.info("No values for the chosen dimension.")
            else:
                sel_val = st.selectbox("Value", values)
                score_df = con.execute(
                    """
                    SELECT dimension, value, credit, credit_share, rating
                    FROM v_item_attr
                    WHERE dimension=$d AND value=$v
                    ORDER BY credit_share DESC
                    LIMIT 1
                    """,
                    {"d": sel_dim, "v": sel_val},
                ).fetchdf()

                st.markdown("**Attribution Score**")
                st.dataframe(score_df, use_container_width=True)

                peers_df = con.execute(
                    "SELECT dimension, value, credit, credit_share, rating FROM v_item_attr WHERE dimension=$d",
                    {"d": sel_dim},
                ).fetchdf()
                if not score_df.empty:
                    st.markdown("#### What this means")
                    st.info(explain_attribution(score_df.iloc[0], peers_df))

                quoted_dim = _quote_ident(sel_dim)
                where_sql, args = build_where(extra=f"{quoted_dim} = $val", params={"val": sel_val})
                rows = con.execute(f"SELECT * FROM v WHERE {where_sql} LIMIT $lim", {**args, "lim": int(row_limit)}).fetchdf()

                st.markdown(f"**Matching Articles** ({len(rows):,} shown; cap = {int(row_limit):,})")
                st.dataframe(rows, use_container_width=True)

                safe_val = re.sub(r"[^A-Za-z0-9_-]+", "_", str(sel_val))
                st.download_button(
                    "⬇️ Download filtered rows (CSV)",
                    rows.to_csv(index=False).encode("utf-8"),
                    file_name=f"{sel_dim}__{safe_val}.csv",
                    mime="text/csv",
                )

    else:  # Term
        term = st.text_input("Type a term to search", "")
        whole_word = st.checkbox("Whole word match (regex)", value=True)

        # show top terms for context
        topN = st.number_input("Top N terms by credit_share", 10, 2000, 50, 10)
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
        if not top_terms.empty:
            st.dataframe(top_terms, use_container_width=True)

        if term:
            tscore = con.execute(
                "SELECT value, credit, credit_share, rating FROM v_term_attr WHERE value = $val ORDER BY credit_share DESC",
                {"val": term},
            ).fetchdf()
            if not tscore.empty:
                st.markdown("**Term Attribution**")
                st.dataframe(tscore, use_container_width=True)

                # Peers universe for terms (synthetic dimension='term' for the explainer)
                term_peers = con.execute(
                    "SELECT 'term' AS dimension, value, credit, credit_share, rating FROM v_term_attr"
                ).fetchdf()
                st.markdown("#### What this means")
                row = tscore.iloc[0].copy()
                row["dimension"] = "term"
                row["value"] = row["value"]
                st.info(explain_attribution(row, term_peers))

            base_where, args = build_where()
            text_expr = "COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,'')"
            if whole_word:
                pattern = r"(?i)\b" + re.escape(term) + r"\b"
                where_sql = f"{base_where} AND REGEXP_MATCHES({text_expr}, $rx)"
                args["rx"] = pattern
            else:
                where_sql = f"{base_where} AND LOWER({text_expr}) LIKE $pat"
                args["pat"] = f"%{term.lower()}%"

            hits = con.execute(f"SELECT * FROM v WHERE {where_sql} LIMIT $lim", {**args, "lim": int(row_limit)}).fetchdf()
            st.markdown(f"**Articles containing “{term}”** ({len(hits):,}; showing up to {row_limit:,})")
            st.dataframe(hits, use_container_width=True)

# ==========================
# TAB 2 — DASHBOARD
# ==========================
with dash_tab:
    st.subheader("Dashboard (quick stats)")

    c1, c2, c3 = st.columns(3)
    try:
        total_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
    except Exception:
        total_rows = 0
    try:
        num_pubs = con.execute("SELECT COUNT(DISTINCT publication_name) FROM v").fetchone()[0]
    except Exception:
        num_pubs = 0
    try:
        date_min, date_max = con.execute("SELECT MIN(load_ts), MAX(load_ts) FROM v").fetchone()
    except Exception:
        date_min = date_max = None

    c1.metric("Rows", f"{total_rows:,}")
    c2.metric("Publications", f"{num_pubs:,}")
    if date_min and date_max:
        c3.metric("Date Range", f"{pd.to_datetime(date_min).date()} → {pd.to_datetime(date_max).date()}")
    else:
        c3.metric("Date Range", "n/a")

    # Top publications by average pub_credit_share (if column exists)
    st.markdown("### Top Publications by avg(pub_credit_share)")
    try:
        cols = con.execute("DESCRIBE v").fetchdf()["column_name"].tolist()
    except Exception:
        cols = []
    if "pub_credit_share" in cols and "publication_name" in cols:
        top_pub = con.execute(
            """
            SELECT publication_name, AVG(pub_credit_share) AS avg_share, COUNT(*) AS n
            FROM v
            WHERE publication_name IS NOT NULL
            GROUP BY 1
            ORDER BY avg_share DESC
            LIMIT 20
            """
        ).fetchdf()
        if not top_pub.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(data=top_pub, x="publication_name", y="avg_share", ax=ax)
            ax.set_title("Top Publications by Average pub_credit_share")
            ax.set_xlabel("")
            ax.set_ylabel("avg(pub_credit_share)")
            ax.tick_params(axis='x', rotation=45)
            st.pyplot(fig)
            st.dataframe(top_pub, use_container_width=True)
        else:
            st.info("No rows for pub_credit_share.")
    else:
        st.info("Column 'pub_credit_share' not found in v; skipping chart.")

    # Top terms snapshot (if attribution present)
    st.markdown("### Top Terms (credit_share)")
    terms = con.execute(
        """
        SELECT value, credit_share, credit
        FROM v_term_attr
        QUALIFY ROW_NUMBER() OVER (PARTITION BY value ORDER BY credit_share DESC) = 1
        ORDER BY credit_share DESC
        LIMIT 30
        """
    ).fetchdf()
    if not terms.empty:
        st.dataframe(terms, use_container_width=True)

# ==========================
# TAB 3 — NETWORK (lightweight)
# ==========================
with net_tab:
    st.subheader("Network Preview")
    # Optional: commit data/network_edges.csv with columns: source,target,weight
    # Search common dirs:
    def _find_edges():
        for d in [ROOT / "data", ROOT / "data" / "processed", ROOT / "processed", APP_DIR]:
            p = d / "network_edges.csv"
            if p.exists():
                return p
        return None

    EDGES_PATH = _find_edges()
    if EDGES_PATH and EDGES_PATH.exists():
        edges = pd.read_csv(EDGES_PATH)
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
        else:
            st.warning("network_edges.csv found but must contain columns: source, target, weight")
    else:
        st.info("No network_edges.csv found. Place one under data/ or data/processed with columns: source,target,weight.")

# -------------------- Footer --------------------
try:
    n_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
except Exception:
    n_rows = 0
st.caption(f"Rows in main dataset: {n_rows:,}.")