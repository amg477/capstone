# app.py — Influence Explorer (Tabbed: Attribution • Dashboard • Network)

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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rcParams

# -------------------- Page config --------------------
st.set_page_config(page_title="Influence Explorer", layout="wide")

# -------------------- Brand styling --------------------
def apply_penta_style():
    PRIMARY_GREEN = "#12715D"
    ACCENT_GREEN  = "#4AB48E"
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

def s(path: str, default=None):
    try:
        cur = st.secrets
        for part in path.split("."):
            cur = cur[part]
        return cur
    except Exception:
        return default

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
if not LOGO_PATH.exists() and (APP_DIR / "penta_logo.png").exists():
    LOGO_PATH = APP_DIR / "penta_logo.png"

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
        <div class="header-bar">
            <img src="data:image/png;base64,{logo_b64}" class="penta-logo"/>
            <div class="header-title"><h1>Influence Explorer</h1></div>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        '<div class="header-bar"><div class="header-title"><h1>Influence Explorer</h1></div></div>',
        unsafe_allow_html=True
    )

# -------------------- Main tabs --------------------
tab1, tab2, tab3, tab4 = st.tabs(["📋 Instructions", "🎯 Attribution", "📊 Dashboard", "🕸️ Network"])

with tab1:
    st.markdown("""
    ## Welcome to Influence Explorer
    
    This application helps you analyze influence patterns in your data through multiple lenses:
    
    ### 🎯 Attribution Tab
    - **Item Attribution**: Analyze influence by publication, author, channel, etc.
    - **Term Attribution**: Search for specific terms and see their influence scores
    
    ### 📊 Dashboard Tab
    - **KPI Metrics**: Key performance indicators at a glance
    - **Visualizations**: Bar charts, pie charts, and time series
    - **Sankey Diagram**: Flow analysis between different dimensions
    - **Sample Data**: View filtered data in tabular format
    
    ### 🕸️ Network Tab
    - **Network Analysis**: Visualize relationships between entities
    - **Edge Analysis**: Understand connection strengths
    - **Node Strength**: Identify influential nodes in the network
    
    ### 🔧 How to Use
    1. **Explore Tabs**: Navigate between different analysis views
    2. **Interact**: Click on charts, adjust parameters, and explore the data
    3. **Export**: Use Streamlit's built-in export features to save results
    
    ### 📈 Tips for Analysis
    - Use the Sankey diagram to understand flow patterns
    - Check attribution scores to identify key influencers
    - Look at time series to spot trends and patterns
    
    **Data Status**: ✅ Loaded and ready for analysis
    """)

with tab2:
    st.subheader("Attribution Lookups")

    lookup_type = st.radio("Lookup type", ["Item", "Term"], horizontal=True)

    if lookup_type == "Item":
        try:
            dims = con.execute("SELECT DISTINCT dimension FROM v_item_attr WHERE dimension IS NOT NULL ORDER BY 1").fetchdf()["dimension"].tolist()
        except Exception:
            dims = []
        
        if not dims:
            st.info("No item attribution available.")
        else:
            sel_dim = st.selectbox("Dimension", dims, index=(dims.index("publication_name") if "publication_name" in dims else 0))
            values = con.execute("SELECT DISTINCT value FROM v_item_attr WHERE dimension=$d AND value IS NOT NULL ORDER BY 1", {"d": sel_dim}).fetchdf()["value"].tolist()
            
            if values:
                sel_val = st.selectbox("Value", values)
                score_df = con.execute("""
                   SELECT dimension, value, credit, credit_share, rating
                   FROM v_item_attr
                   WHERE dimension=$d AND value=$v
                   ORDER BY credit_share DESC
                   LIMIT 1
                """, {"d": sel_dim, "v": sel_val}).fetchdf()
                
                st.dataframe(score_df, use_container_width=True)
                
                peers_df = con.execute("SELECT dimension, value, credit FROM v_item_attr WHERE dimension=$d", {"d": sel_dim}).fetchdf()
                
                if not score_df.empty:
                    st.markdown("#### What this means")
                    st.info(explain_attribution(score_df.iloc[0], peers_df))

                quoted_dim = _quote_ident(sel_dim)
                where_sql, args = build_where(extra=f"{quoted_dim} = $val", params={"val": sel_val})
                rows = con.execute(f"SELECT * FROM v WHERE {where_sql} LIMIT 1000", args).fetchdf()
                st.dataframe(rows, use_container_width=True)
            else:
                st.info("No values for this dimension.")
    else:
        term = st.text_input("Type a term to search")
        
        if term:
            tscore = con.execute("SELECT value, credit, credit_share, rating FROM v_term_attr WHERE value=$v ORDER BY credit_share DESC", {"v": term}).fetchdf()
            st.dataframe(tscore, use_container_width=True)
            
            peers = con.execute("SELECT value, credit, credit_share, rating FROM v_term_attr").fetchdf()
            
            if not tscore.empty:
                st.markdown("**Term Attribution**")
                st.dataframe(tscore, use_container_width=True)
                
                term_peers = con.execute(
                    "SELECT 'term' AS dimension, value, credit, credit_share, rating FROM v_term_attr"
                ).fetchdf()
                row = tscore.iloc[0].copy()
                row["dimension"] = "term"
                row["value"] = row["value"]
                st.info(explain_attribution(row, term_peers))

            text = "LOWER(COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,''))"
            hits = con.execute(
                f"SELECT * FROM v WHERE {text} LIKE $pat LIMIT 1000",
                {"pat": f"%{term.lower()}%"}
            ).fetchdf()
            st.dataframe(hits, use_container_width=True)

with tab3:
    st.subheader("Dashboard")
    
    # Dashboard filters
    st.markdown("### Filters")
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
        
        sel_pubs = st.multiselect("Publications", 
                                 con.execute("SELECT DISTINCT publication_clean FROM v_enriched WHERE publication_clean IS NOT NULL ORDER BY 1").fetchdf()["publication_clean"].tolist(),
                                 default=[])
        
        sel_channels = st.multiselect("Channels",
                                    con.execute("SELECT DISTINCT COALESCE(channel_clean, channel_name_clean) AS ch FROM v_enriched WHERE ch IS NOT NULL ORDER BY 1").fetchdf()["ch"].tolist(),
                                    default=[])
    
    with col2:
        sel_bands = st.multiselect("Sentiment bands",
                                  con.execute("SELECT DISTINCT sentiment_band FROM v_enriched WHERE sentiment_band IS NOT NULL ORDER BY 1").fetchdf()["sentiment_band"].tolist(),
                                  default=[])
        
        sel_authors = st.multiselect("Authors",
                                    con.execute("SELECT DISTINCT COALESCE(author_clean, author_name_clean) AS auth FROM v_enriched WHERE auth IS NOT NULL ORDER BY 1").fetchdf()["auth"].tolist(),
                                    default=[])
        
        sel_topics = st.multiselect("Topics",
                                   con.execute("SELECT DISTINCT COALESCE(topic_clean, topics_clean) AS topic FROM v_enriched WHERE topic IS NOT NULL ORDER BY 1").fetchdf()["topic"].tolist(),
                                   default=[])
    
    # Build where clause
    w, args = where_from_filters(date_range, sel_pubs, sel_channels, sel_bands, sel_authors, sel_topics)
    
    # KPI Metrics
    total_pubs = con.execute(f"SELECT COUNT(DISTINCT publication_clean) FROM v_enriched WHERE {w}", args).fetchone()[0]
    uniq_sources = con.execute(f"SELECT COUNT(DISTINCT COALESCE(source_name_clean, publication_clean)) FROM v_enriched WHERE {w}", args).fetchone()[0]
    uniq_authors = con.execute(f"SELECT COUNT(DISTINCT COALESCE(author_clean, author_name_clean)) FROM v_enriched WHERE {w}", args).fetchone()[0]
    
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

    # Sample Data
    st.markdown("### Filtered sample")
    sample = con.execute(f"SELECT * FROM v WHERE {w} LIMIT 1000", args).fetchdf()
    st.dataframe(sample, use_container_width=True, height=360)

with tab4:
    st.subheader("Network Preview")
    
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
