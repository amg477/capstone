# app.py — Influence Explorer (logo+title, instructions tab, readable Sankey)

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import duckdb
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import base64
import base64
from pathlib import Path

# ---------- Page + brand ----------
st.set_page_config(page_title="Influence Explorer", layout="wide")
BRAND = {"primary": "#12715D", "accent": "#4AB48E", "text": "#133C35", "bg2": "#F4F6F5"}

st.markdown("""
<style>
/* Kill the gray outline / shadow on Sankey node labels */
div[data-testid="stPlotlyChart"] .sankey .node text,
div[data-testid="stPlotlyChart"] .sankey-layer .node text {
  stroke: none !important;
  paint-order: fill !important;
  text-shadow: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown(f"""
<style>
  html, body, [class*="css"] {{
    font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial;
    color: {BRAND['text']};
  }}
  .block-container {{ padding-top: 2.5rem; padding-bottom: 1.5rem; }}

  /* Header bar container */
  .header-bar {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1rem;
  }}
  /* Much smaller logo */
  .penta-logo {{
    height: 36px;     /* adjust to make smaller/larger */
    width: auto;
  }}
  .header-title h1 {{
    font-size: 1.9rem;
    line-height: 1.2;
    margin: 0;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    color: {BRAND['primary']};
  }}
</style>
""", unsafe_allow_html=True)

# ---------- Header (logo left + title right) ----------
APP_DIR = Path(__file__).resolve().parent
ROOT    = APP_DIR.parent
logo_path = next(
    (p for p in [
        ROOT / "final_deliverable/penta_logo.png",
        ROOT / "data/penta_logo.png",
        APP_DIR / "penta_logo.png"
    ] if p.exists()),
    None
)

if logo_path:
    with open(logo_path, "rb") as f:
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

# ---------- Data location helpers (no widgets here) ----------
SEARCH_DIRS = [ROOT / "data" / "processed", ROOT / "data", APP_DIR]
CANDIDATE = ["final_model_dataset.parquet", "final_model_dataset.csv", "data.parquet", "data.csv"]

def _first_local_data() -> Optional[Path]:
    for d in SEARCH_DIRS:
        for nm in CANDIDATE:
            p = d / nm
            if p.exists():
                return p
    return None

def pick_data_path_with_uploader() -> Optional[Path]:
    """UI helper (NOT cached). Tries local files, else asks user to upload."""
    found = _first_local_data()
    if found:
        return found
    st.warning(
        "No dataset found in common folders.\n\n"
        "Searched:\n- " + "\n- ".join(p.as_posix() for p in SEARCH_DIRS) +
        "\n\nUpload a CSV or Parquet to continue."
    )
    upl = st.file_uploader("Upload CSV or Parquet", type=["csv", "parquet"])
    if upl is None:
        return None
    tmp = (ROOT / f"tmp_upload{Path(upl.name).suffix}")
    tmp.write_bytes(upl.getbuffer())
    return tmp

@st.cache_resource(show_spinner=True)
def connect_duckdb(data_path: Optional[Path]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()

    # main table v
    if data_path and data_path.exists():
        if data_path.suffix.lower() == ".parquet":
            con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet('{data_path.as_posix()}')")
        else:
            con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto('{data_path.as_posix()}', IGNORE_ERRORS=TRUE)")
    else:
        # empty view if nothing chosen yet
        con.execute("CREATE OR REPLACE VIEW v AS SELECT 1 WHERE 0")

    # optional attribution views
    attr_paths = [
        ROOT / "data" / "processed" / "attribution_all_scored.csv",
        ROOT / "data" / "attribution_all_scored.csv",
        APP_DIR / "attribution_all_scored.csv",
    ]
    attr = next((p for p in attr_paths if p.exists()), None)
    if attr:
        con.execute(f"CREATE OR REPLACE VIEW v_attr AS SELECT * FROM read_csv_auto('{attr.as_posix()}', IGNORE_ERRORS=TRUE)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM v_attr WHERE kind='item'")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT * FROM v_attr WHERE kind='term'")
    else:
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT 1 WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT 1 WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT 1 WHERE 0")
    return con

# ---- Pick path (UI) then connect (cached) ----
DATA_PATH = pick_data_path_with_uploader()
con = connect_duckdb(DATA_PATH)

# ---------- Cleaned view ----------
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

def _first_existing(cols: List[str], existing: set) -> Optional[str]:
    for c in cols:
        if c in existing: return c
    return None

def _cleaned_select(cols: List[str], alias: str, existing: set) -> str:
    base = _first_existing(cols, existing)
    return (f"{_clean_expr('v.'+base)} AS {alias}") if base else f"NULL AS {alias}"

COLUMNS = set(con.execute("DESCRIBE v").fetchdf()["column_name"].tolist())

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

# ---------- Helpers ----------
def explain_attribution(row: pd.Series, peers: Optional[pd.DataFrame] = None) -> str:
    dim = str(row.get("dimension","dimension"))
    val = str(row.get("value","value"))
    cred = float(row.get("credit",0) or 0)
    share = float(row.get("credit_share",0) or 0)
    out = [f"**{dim} = {val}** contributes **{share:.2%}** of total credit (raw {cred:,.4f})."]
    if peers is not None and not peers.empty and "credit" in peers.columns:
        same = peers[peers.get("dimension","")==dim]
        if not same.empty and val in same.get("value",[]).values:
            same["_r"] = same["credit"].rank(ascending=False, method="min")
            r = int(same.loc[same["value"]==val, "_r"].iloc[0]); n = int(same.shape[0])
            out.append(f"Rank **#{r} of {n}** (~{100*(1-(r-1)/max(n,1)):.0f}th percentile).")
    if share < 0.001: out.append("Very small share — likely low standalone influence.")
    return " ".join(out)

@st.cache_data
def date_bounds(col: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        mn, mx = con.execute(f"SELECT MIN(CAST({col} AS TIMESTAMP)), MAX(CAST({col} AS TIMESTAMP)) FROM v").fetchone()
        if mn and mx: return (pd.to_datetime(mn), pd.to_datetime(mx))
    except Exception: pass
    return None

@st.cache_data
def distinct_clean(expr_sql: str) -> List[str]:
    df = con.execute(f"SELECT DISTINCT {expr_sql} AS val FROM v_enriched WHERE {expr_sql} IS NOT NULL ORDER BY 1").fetchdf()
    return df["val"].tolist()

def shorten(label: str, max_len: int = 28) -> str:
    s = str(label)
    return s if len(s) <= max_len else s[:max_len-1] + "…"

# ---------- Sidebar filters ----------
st.sidebar.header("Filters")
ld = date_bounds("load_ts")
date_range = st.sidebar.date_input("Load date range", (ld[0].date(), ld[1].date())) if ld else None

sel_pubs     = st.sidebar.multiselect("Publication", distinct_clean("publication_clean")) if "publication_name" in COLUMNS else []
sel_channels = st.sidebar.multiselect("Channel", distinct_clean("COALESCE(channel_clean, channel_name_clean)")) if {"channel","channel_name","source_type"} & COLUMNS else []
sel_bands    = st.sidebar.multiselect("Sentiment band", distinct_clean("sentiment_band")) if "sentiment_band" in COLUMNS else []
sel_authors  = st.sidebar.multiselect("Author", distinct_clean("COALESCE(author_clean, author_name_clean)")) if {"author","author_name"} & COLUMNS else []
sel_topics   = st.sidebar.multiselect("Topic", distinct_clean("COALESCE(topic_clean, topics_clean)")) if {"topic","topics"} & COLUMNS else []
row_limit    = st.sidebar.number_input("Rows in tables", 100, 10000, 1000, 50)

def where_from_filters() -> Tuple[str, Dict]:
    clauses, args = ["1=1"], {}
    if date_range and len(date_range)==2:
        dmin = pd.to_datetime(date_range[0]); dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        clauses.append("CAST(load_ts AS TIMESTAMP) >= $dmin AND CAST(load_ts AS TIMESTAMP) < $dmax")
        args.update(dmin=dmin, dmax=dmax)
    if sel_pubs:
        clauses.append("publication_clean IN $pubs"); args["pubs"]=sel_pubs
    if sel_channels:
        clauses.append("COALESCE(channel_clean, channel_name_clean) IN $chs"); args["chs"]=sel_channels
    if sel_bands:
        clauses.append("sentiment_band IN $bands"); args["bands"]=sel_bands
    if sel_authors:
        clauses.append("COALESCE(author_clean, author_name_clean) IN $auths"); args["auths"]=sel_authors
    if sel_topics:
        clauses.append("COALESCE(topic_clean, topics_clean) IN $topics"); args["topics"]=sel_topics
    return " AND ".join(clauses), args

# ---------- Tabs ----------
tab_help, tab_attr, tab_dash, tab_net = st.tabs(["ℹ️ Instructions", "🧮 Attribution", "📊 Dashboard", "🕸️ Network"])

# ===== Instructions =====
with tab_help:
    st.subheader("How to use this app")
    st.markdown("""
**Filters (left sidebar)**
- Narrow the dataset by date, publication, channel, sentiment, author, and topic.  
- All charts/tables update instantly.

**Attribution tab**
- *Item* lets you pick a **dimension** (e.g., publication_name) and a **value** (one publisher).
- Shows its **credit / credit_share / rating** and a short **interpretation**, plus matching rows.
- *Term* looks up a specific keyword from the attribution table and lists matching articles.

**Dashboard tab**
- KPI cards summarize totals.
- Two **side-by-side bar charts**: top groups by **avg influence** and by **count**.
- **Pie chart**: top 20 groups by avg influence share.
- **Sankey**: flow between any two categorical fields (choose source/target).  
  Use *Top Sources*, *Top Targets*, *Max Links*, and *Bucket “Other”* to keep it readable.
- **Conversions over time** bar chart (falls back to counts if your dataset lacks conversions).
- **Filtered sample** table reflects your sidebar selections.

**Network tab**
- Shows a sample of `network_edges.csv` if present (columns: source, target, weight), plus top nodes by strength.
""")

# ===== Attribution =====
with tab_attr:
    st.subheader("Attribution Lookups")
    lookup_type = st.radio("Lookup type", ["Item","Term"], horizontal=True)

    if lookup_type=="Item":
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
                st.dataframe(score_df, width="stretch")
                peers_df = con.execute("SELECT dimension, value, credit FROM v_item_attr WHERE dimension=$d", {"d": sel_dim}).fetchdf()
                if not score_df.empty:
                    st.markdown("#### What this means")
                    st.info(explain_attribution(score_df.iloc[0], peers_df))
                w,args = where_from_filters()
                rows = con.execute(f'SELECT * FROM v WHERE {w} AND "{sel_dim}"=$v LIMIT $lim', {**args, "v": sel_val, "lim": int(row_limit)}).fetchdf()
                st.dataframe(rows, width="stretch")
            else:
                st.info("No values for this dimension.")
    else:
        term = st.text_input("Type a term to search")
        if term:
            tscore = con.execute("SELECT value, credit, credit_share, rating FROM v_term_attr WHERE value=$v ORDER BY credit_share DESC", {"v": term}).fetchdf()
            st.dataframe(tscore, width="stretch")
            peers = con.execute("SELECT 'term' AS dimension, value, credit FROM v_term_attr").fetchdf()
            if not tscore.empty:
                row = tscore.iloc[0].copy(); row["dimension"]="term"
                st.markdown("#### What this means")
                st.info(explain_attribution(row, peers))
            w,args = where_from_filters()
            text = "LOWER(COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,''))"
            hits = con.execute(
                f"SELECT * FROM v WHERE {w} AND {text} LIKE $pat LIMIT $lim",
                {**args, "pat": f"%{term.lower()}%", "lim": int(row_limit)}
            ).fetchdf()
            st.dataframe(hits, width="stretch")

# ===== Dashboard =====
with tab_dash:
    st.subheader("Dashboard")
    w, args = where_from_filters()

    total_pubs   = con.execute(f"SELECT COUNT(DISTINCT publication_clean) FROM v_enriched WHERE {w}", args).fetchone()[0]
    uniq_sources = con.execute(f"SELECT COUNT(DISTINCT COALESCE(source_name_clean, publication_clean)) FROM v_enriched WHERE {w}", args).fetchone()[0]
    uniq_authors = con.execute(f"SELECT COUNT(DISTINCT COALESCE(author_clean, author_name_clean)) FROM v_enriched WHERE {w}", args).fetchone()[0]
    infl_col     = "pub_credit_share" if "pub_credit_share" in COLUMNS else ("credit_share" if "credit_share" in COLUMNS else None)
    avg_infl     = con.execute(f"SELECT AVG({infl_col}) FROM v WHERE {w}", args).fetchone()[0] if infl_col else None

    m1,m2,m3,m4 = st.columns(4)
    m1.metric("Total publications", f"{total_pubs:,}")
    m2.metric("Avg influence score", f"{avg_infl:.3f}" if avg_infl is not None else "n/a")
    m3.metric("Unique sources", f"{uniq_sources:,}")
    m4.metric("Unique authors", f"{uniq_authors:,}")

    st.divider()

    cat_cols = [c for c in ["publication_name","source_name","channel_name","author_name","topic","sentiment_band"] if c in COLUMNS]
    if not cat_cols:
        st.info("No categorical columns to group by.")
    else:
        dim = st.selectbox("Group charts by", cat_cols, index=0)

        circ_col = next((c for c in ["circulation","circulation_size","reach","impressions","audience"] if c in COLUMNS), None)
        circ_sql = f"COALESCE(SUM({circ_col}),0)" if circ_col else "COUNT(*)"
        circ_label = "Total circulation" if circ_col else "Count"

        agg = con.execute(f"""
            SELECT {dim} AS dim,
                   AVG({infl_col}) AS avg_influence,
                   COUNT(*) AS n,
                   {circ_sql} AS total_metric
            FROM v WHERE {w}
            GROUP BY 1 HAVING dim IS NOT NULL
        """, args).fetchdf()

        top_n = st.slider("Top N", 5, 50, 20, 1)

        cA,cB = st.columns(2)
        if not agg.empty:
            b1 = alt.Chart(agg.sort_values("avg_influence", ascending=False).head(top_n)).mark_bar(color=BRAND["primary"]).encode(
                y=alt.Y("dim:N", sort="-x", title=None),
                x=alt.X("avg_influence:Q", title="Avg influence"),
                tooltip=["dim", alt.Tooltip("avg_influence:Q", format=".3f"), "n"],
            )
            b2 = alt.Chart(agg.sort_values("n", ascending=False).head(top_n)).mark_bar(color=BRAND["accent"]).encode(
                y=alt.Y("dim:N", sort="-x", title=None),
                x=alt.X("n:Q", title="Count"),
                tooltip=["dim", "n", alt.Tooltip("avg_influence:Q", format=".3f")],
            )
            cA.altair_chart(b1.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
            cB.altair_chart(b2.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
        else:
            st.info("No data for current filters.")

        st.divider()

        if infl_col and not agg.empty:
            pie_df = agg.sort_values("avg_influence", ascending=False).head(20)
            fig_pie = px.pie(
                pie_df, names="dim", values="avg_influence",
                color_discrete_sequence=[BRAND["primary"], BRAND["accent"], "#CFECE4", "#E7F6F1"]
            )
            fig_pie.update_traces(textinfo="percent+label", pull=[0.02]*len(pie_df))
            fig_pie.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
            st.plotly_chart(fig_pie, width="stretch")

        st.divider()

        # --------- Sankey (readable + controls) ----------
        left, right = st.columns(2)
        src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
        tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols) - 1), key="sank_tgt")

        c1, c2, c3, c4 = st.columns(4)
        top_sources = c1.slider("Top Sources", 3, 50, 15, 1)
        top_targets = c2.slider("Top Targets", 2, 20, 6, 1)
        max_links   = c3.slider("Max Links", 10, 500, 120, 10)
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
                # Build nodes/links with truncated display labels (full in hover)
                nodes_all = pd.Series(pd.concat([sdata["s"], sdata["t"]])).astype(str).unique().tolist()
                labels_short = [shorten(n) for n in nodes_all]
                index = {n: i for i, n in enumerate(nodes_all)}

                # UI: quick high-contrast toggle
                hc = st.checkbox("High-contrast labels", value=True, help="Use light nodes with dark borders + larger text")

                # High-contrast palette
                node_fill   = "#CFECE4" if hc else BRAND["primary"]     # light mint (better for text)
                node_border = BRAND["primary"] if hc else "white"       # dark border on light nodes
                link_rgba   = "rgba(18,113,93,0.22)" if hc else "rgba(18,113,93,0.35)"
                font_color  = BRAND["text"]                             # dark text
                font_size   = 16 if hc else 15

                fig = go.Figure(go.Sankey(
                    arrangement="snap",
                    node=dict(
                        label=labels_short,
                        pad=26,
                        thickness=22,
                        color=[node_fill] * len(nodes_all),
                        line=dict(color="rgba(0,0,0,0)", width=0)   # ← no outline
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

                # Layout-level font controls Sankey label text on your Plotly version
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

        # --------- Conversions over time ----------
        time_col = "conversion_ts" if "conversion_ts" in COLUMNS else ("load_ts" if "load_ts" in COLUMNS else ("source_time" if "source_time" in COLUMNS else None))
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
                    .mark_bar(color=BRAND["primary"])
                    .encode(x=alt.X("d:T", title="Date"), y=alt.Y("y:Q", title="Conversions"))
                    .properties(height=300)
                    .interactive()
                )
                st.altair_chart(conv_chart.configure_view(strokeWidth=0), use_container_width=True)

        st.divider()
        st.markdown("### Filtered sample")
        sample = con.execute(f"SELECT * FROM v WHERE {w} LIMIT $lim", {**args, "lim": int(row_limit)}).fetchdf()
        st.dataframe(sample, use_container_width=True, height=360)
# ===== Network =====
with tab_net:
    st.subheader("Network Preview")
    net_path = next((p for p in [ROOT/"data/processed/network_edges.csv", ROOT/"data/network_edges.csv", APP_DIR/"network_edges.csv"] if p.exists()), None)
    if net_path:
        edges = pd.read_csv(net_path)
        if {"source","target","weight"}.issubset(edges.columns):
            max_rows = st.slider("Max edges to show", 200, 10000, 2000, 200)
            edges = edges.nlargest(max_rows, "weight")
            st.dataframe(edges.head(500), width="stretch", height=360)
            deg = pd.concat([
                edges.groupby("source")["weight"].sum().rename("out"),
                edges.groupby("target")["weight"].sum().rename("in"),
            ], axis=1).fillna(0)
            deg["strength"] = deg["in"] + deg["out"]
            st.markdown("**Top nodes by strength**")
            st.dataframe(deg.sort_values("strength", ascending=False).head(50), width="stretch")
        else:
            st.warning("network_edges.csv must have columns: source, target, weight")
    else:
        st.info("Place a network_edges.csv in data/processed/ to enable network view.")

# ---------- Footer ----------
try:
    n_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
except Exception:
    n_rows = 0
st.caption(f"Rows in main dataset: {n_rows:,}.")