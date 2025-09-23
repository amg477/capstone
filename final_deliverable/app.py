# app.py — Influence Explorer (Tabbed Dashboard: Attribution • Dashboard • Network)
# ------------------------------------------------------------------------------
# Compact, Python 3.9–friendly, local-only version with three tabs:
#   1) Attribution — Item & Term lookups + "What this means"
#   2) Dashboard   — KPIs + simple charts (fast, DuckDB-backed)
#   3) Network     — Lightweight network preview from an optional edges file
#
# Expects local files under ../data/ relative to this file (adjust ROOT as needed):
#   - final_model_dataset.parquet (preferred) or final_model_dataset.csv  -> view v
#   - attribution_all_scored.csv (optional) -> views v_attr, v_item_attr, v_term_attr
#   - network_edges.csv (optional) with columns: source, target, weight

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import duckdb
import pandas as pd
import streamlit as st

# Set Brand Colors
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

def apply_penta_style():
    """
    Apply Penta Group brand styling to matplotlib/seaborn plots.
    """
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

# -------------------- Config --------------------
st.set_page_config(page_title="Influence Explorer", layout="wide")

ROOT = Path("/Users/annaglass/capstone/capstone")  # ← adjust if needed
DATA_PARQUET = ROOT / "data" / "final_model_dataset.parquet"
DATA_CSV     = ROOT / "data" / "final_model_dataset.csv"
ATTR_CSV     = ROOT / "data" / "attribution_all_scored.csv"
EDGES_CSV    = ROOT / "data" / "network_edges.csv"  # optional (source,target,weight)

APP_DIR = Path(__file__).parent
LOGO_PATH = APP_DIR / "penta_logo.png"
if LOGO_PATH.exists():
    st.image(str(LOGO_PATH), width=200)

st.title("Healthcare Policy: Influence and Networks")

# -------------------- Helpers --------------------

def explain_attribution(row: pd.Series, universe: Optional[pd.DataFrame] = None) -> str:
    dim   = str(row.get("dimension", "dimension"))
    val   = str(row.get("value", "value"))
    cred  = float(row.get("credit", 0.0))
    share = float(row.get("credit_share", 0.0))
    rating = row.get("rating", None)

    parts = [f"**{dim} = {val}**"]
    parts.append(
        f"within the current selection" + (f" (rating = **{int(rating)}**)" if pd.notna(rating) else "") + "."
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


def _q(s: str) -> str:
    return "'" + s.replace("'", "''") + "'"

def _quote_ident(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'

# -------------------- Data bootstrap (local only) --------------------

@st.cache_resource(show_spinner=True)
def connect_duckdb() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    # Build view v from parquet or csv
    if DATA_PARQUET.exists():
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet({_q(str(DATA_PARQUET))})")
    elif DATA_CSV.exists():
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto({_q(str(DATA_CSV))}, IGNORE_ERRORS=TRUE)")
    else:
        raise FileNotFoundError("Missing final_model_dataset.parquet/csv under data/")

    # Optional attribution
    if ATTR_CSV.exists():
        con.execute(f"CREATE OR REPLACE VIEW v_attr AS SELECT * FROM read_csv_auto({_q(str(ATTR_CSV))}, IGNORE_ERRORS=TRUE)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM v_attr WHERE kind='item'")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT * FROM v_attr WHERE kind='term'")
    else:
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT * FROM (SELECT 1 WHERE 0)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM (SELECT 1 WHERE 0)")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT * FROM (SELECT 1 WHERE 0)")

    return con

con = connect_duckdb()

# call style
apply_penta_style()
# -------------------- Sidebar filters --------------------

st.sidebar.header("Filters")
# Date range (based on optional load_ts)
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
    bands = con.execute("SELECT DISTINCT sentiment_band FROM v WHERE sentiment_band IS NOT NULL ORDER BY 1").fetchdf()[
        "sentiment_band"
    ].tolist()
except Exception:
    bands = []
sel_bands = st.sidebar.multiselect("Sentiment band", bands, default=bands)

# Publications
try:
    pubs = con.execute("SELECT DISTINCT publication_name FROM v WHERE publication_name IS NOT NULL ORDER BY 1").fetchdf()[
        "publication_name"
    ].tolist()
except Exception:
    pubs = []
sel_pubs = st.sidebar.multiselect("Publication", pubs, default=[])

row_limit = st.sidebar.number_input("Rows to display", 50, 50000, 2000, 50)

# WHERE builder shared by tabs

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

# -------------------- Tabs --------------------

attr_tab, dash_tab, net_tab = st.tabs(["Attribution", "Dashboard", "Network"])

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
                # Build peers universe for terms (all rows for that value's dimension ~ terms only)
                term_peers = con.execute(
                    "SELECT 'term' AS dimension, value, credit, credit_share, rating FROM v_term_attr",
                ).fetchdf()
                st.markdown("#### What this means")
                st.info(explain_attribution(tscore.iloc[0], term_peers))

            base_where, args = build_where()
            text_expr = "COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,'')"
            if whole_word:
                pattern = r"(?i)\\b" + re.escape(term) + r"\\b"
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
    st.subheader("Dashboard")

    # KPIs
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

    # Top publications by share (if available)
    st.markdown("### Top Publications by pub_credit_share")
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
        st.bar_chart(top_pub.set_index("publication_name")["avg_share"])  # simple built-in chart
        st.dataframe(top_pub, use_container_width=True)
    else:
        st.info("No pub_credit_share available to chart.")

    # Top terms (from attribution) snapshot
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
# TAB 3 — NETWORK
# ==========================
with net_tab:
    st.subheader("Network Preview")

    if EDGES_CSV.exists():
        edges = pd.read_csv(EDGES_CSV)
        required = {"source", "target", "weight"}
        if required.issubset(set(edges.columns)):
            st.write("Edges sample:")
            st.dataframe(edges.head(200), use_container_width=True)

            # Quick, dependency-light adjacency stats
            st.markdown("**Basic stats**")
            deg = pd.concat([
                edges.groupby("source")["weight"].sum().rename("out_weight"),
                edges.groupby("target")["weight"].sum().rename("in_weight"),
            ], axis=1).fillna(0)
            deg["strength"] = deg["in_weight"] + deg["out_weight"]
            st.dataframe(deg.sort_values("strength", ascending=False).head(30))

            # Simple edge filter
            min_w = st.slider("Min edge weight to show", float(edges["weight"].min()), float(edges["weight"].max()), float(edges["weight"].quantile(0.75)))
            sub = edges[edges["weight"] >= min_w].copy()
            st.write(f"Filtered edges: {len(sub):,} (of {len(edges):,})")
            st.dataframe(sub.head(200), use_container_width=True)

            st.caption("Tip: For interactive graphs, consider exporting to pyvis or Plotly. This tab keeps dependencies light.")
        else:
            st.warning("network_edges.csv found but must contain columns: source, target, weight")
    else:
        st.info("No network_edges.csv found. Place one under data/ with columns: source,target,weight.")

# -------------------- Footer --------------------
try:
    n_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
except Exception:
    n_rows = 0
st.caption(f"Rows in main dataset: {n_rows:,}.")