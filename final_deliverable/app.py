# app.py — Influence Explorer (Azure + logo header + instructions + readable Sankey)

from __future__ import annotations
from pathlib import Path
from typing import Optional, Dict, List, Tuple

import duckdb
import pandas as pd
import streamlit as st
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

# =============================================================================
# Page & Brand
# =============================================================================
st.set_page_config(page_title="Influence Explorer", layout="wide")
BRAND = {"primary": "#12715D", "accent": "#4AB48E", "text": "#133C35", "bg2": "#F4F6F5"}

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

  /* Header layout */
  .header-wrap {{
    display: flex; align-items: center; gap: 1rem; margin-bottom: .5rem;
  }}
  .penta-logo {{ height: 56px; width: auto; }}
  @media (max-width: 1200px) {{
    .penta-logo {{ height: 48px; }}
  }}
</style>
""",
    unsafe_allow_html=True,
)
alt.data_transformers.disable_max_rows()

# =============================================================================
# Azure Loader
# =============================================================================
try:
    from azure.storage.blob import BlobServiceClient
except Exception:
    BlobServiceClient = None

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
APP_TMP = Path("/tmp/influence_data")
APP_TMP.mkdir(parents=True, exist_ok=True)

def _dl_blob_to(tmpdir: Path, cont, blob_name: str, filename: str) -> Path:
    dest = tmpdir / filename
    with open(dest, "wb") as f:
        data = cont.get_blob_client(blob_name).download_blob().readall()
        f.write(data)
    return dest

def load_data_from_azure() -> dict:
    """
    Download dataset/attribution/logo from Azure to /tmp and return local paths.
    Requires secrets.toml:
    [data]
    mode="azure"
    AZURE_STORAGE_CONNECTION_STRING="..."
    container="capstone"
    parquet_blob="data/processed/final_model_dataset.parquet" # or csv_blob
    attr_blob="data/processed/attribution_all_scored.csv"     # optional
    logo_blob="final_deliverable/penta_logo.png"              # optional
    """
    out = {"data_path": None, "attr_path": None, "logo_path": None}
    data = st.secrets.get("data", {})

    if (data.get("mode", "azure").lower() != "azure") or not data.get("AZURE_STORAGE_CONNECTION_STRING"):
        st.error("Azure mode is not configured. Set [data] in secrets.toml with your connection string.")
        st.stop()

    if BlobServiceClient is None:
        st.error("Package 'azure-storage-blob' not installed. Add it to requirements.txt.")
        st.stop()

    try:
        svc = BlobServiceClient.from_connection_string(data["AZURE_STORAGE_CONNECTION_STRING"])
        cont = svc.get_container_client(data["container"])
    except Exception as e:
        st.error(f"Could not connect to Azure container: {e}")
        st.stop()

    pq_blob  = data.get("parquet_blob")
    csv_blob = data.get("csv_blob")
    if pq_blob:
        out["data_path"] = _dl_blob_to(APP_TMP, cont, pq_blob, "main.parquet")
    elif csv_blob:
        out["data_path"] = _dl_blob_to(APP_TMP, cont, csv_blob, "main.csv")
    else:
        st.error("No dataset blobs configured. Provide data.parquet_blob or data.csv_blob in secrets.toml.")
        st.stop()

    if data.get("attr_blob"):
        out["attr_path"] = _dl_blob_to(APP_TMP, cont, data["attr_blob"], "attribution.csv")
    if data.get("logo_blob"):
        out["logo_path"] = _dl_blob_to(APP_TMP, cont, data["logo_blob"], "logo.png")

    return out

@st.cache_resource
def connect_duckdb_with_azure() -> tuple[duckdb.DuckDBPyConnection, dict]:
    paths = load_data_from_azure()
    p = paths["data_path"]
    con = duckdb.connect()

    if p.suffix.lower() == ".parquet":
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_parquet('{p.as_posix()}')")
    else:
        con.execute(f"CREATE OR REPLACE VIEW v AS SELECT * FROM read_csv_auto('{p.as_posix()}', IGNORE_ERRORS=TRUE)")

    if paths["attr_path"]:
        ap = paths["attr_path"]
        con.execute(f"CREATE OR REPLACE VIEW v_attr AS SELECT * FROM read_csv_auto('{ap.as_posix()}', IGNORE_ERRORS=TRUE)")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT * FROM v_attr WHERE kind='item'")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT * FROM v_attr WHERE kind='term'")
    else:
        con.execute("CREATE OR REPLACE VIEW v_attr AS SELECT 1 WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_item_attr AS SELECT 1 WHERE 0")
        con.execute("CREATE OR REPLACE VIEW v_term_attr  AS SELECT 1 WHERE 0")

    return con, paths

con, PATHS = connect_duckdb_with_azure()

# ----- Readiness guard (don’t run any UI if v is missing/empty)
def _quote_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'

try:
    _cols_df = con.execute("DESCRIBE v").fetchdf()
    COLUMNS = set(_cols_df["column_name"].tolist())
    ROWS_IN_V = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
except Exception:
    COLUMNS = set()
    ROWS_IN_V = 0

if ROWS_IN_V == 0:
    st.error("The dataset view `v` is empty. Check your Azure blobs or file permissions.")
    st.stop()

# =============================================================================
# Header (logo + title)
# =============================================================================
with st.container():
    st.write(
        '<div class="header-wrap">'
        + (f'<img src="file://{PATHS["logo_path"].as_posix()}" class="penta-logo"/>' if PATHS.get("logo_path") else "")
        + '<h1 style="margin:0;">Influence Explorer</h1>'
        + "</div>",
        unsafe_allow_html=True,
    )

# =============================================================================
# Cleaned helper view (v_enriched)
# =============================================================================
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
        if c in existing:
            return c
    return None

def _cleaned_select(cols: List[str], alias: str, existing: set) -> str:
    base = _first_existing(cols, existing)
    return (f"{_clean_expr('v.'+base)} AS {alias}") if base else f"NULL AS {alias}"

select_parts = [
    "v.*",
    _cleaned_select(["author", "author_name"], "author_clean", COLUMNS),
    _cleaned_select(["author_name"], "author_name_clean", COLUMNS),
    _cleaned_select(["publication_name", "publication", "source"], "publication_clean", COLUMNS),
    _cleaned_select(["channel", "channel_name", "source_type"], "channel_clean", COLUMNS),
    _cleaned_select(["channel_name", "source_type"], "channel_name_clean", COLUMNS),
    _cleaned_select(["topic", "topics"], "topic_clean", COLUMNS),
    _cleaned_select(["topics", "topic"], "topics_clean", COLUMNS),
    _cleaned_select(["source_name", "publication_name"], "source_name_clean", COLUMNS),
]
con.execute("CREATE OR REPLACE VIEW v_enriched AS SELECT " + ", ".join(select_parts) + " FROM v AS v")

# =============================================================================
# Helpers
# =============================================================================
def explain_attribution(row: pd.Series, peers: Optional[pd.DataFrame] = None) -> str:
    dim = str(row.get("dimension", "dimension"))
    val = str(row.get("value", "value"))
    cred = float(row.get("credit", 0) or 0)
    share = float(row.get("credit_share", 0) or 0)
    out = [f"**{dim} = {val}** contributes **{share:.2%}** of total credit (raw {cred:,.4f})."]
    if peers is not None and not peers.empty and "credit" in peers.columns:
        same = peers[peers.get("dimension", "") == dim]
        if not same.empty and val in same.get("value", []).values:
            same = same.assign(_r=same["credit"].rank(ascending=False, method="min"))
            r = int(same.loc[same["value"] == val, "_r"].iloc[0])
            n = int(same.shape[0])
            out.append(f"Rank **#{r} of {n}** (~{100 * (1 - (r - 1) / max(n, 1)):.0f}th percentile).")
    if share < 0.001:
        out.append("Very small share — likely low standalone influence.")
    return " ".join(out)

@st.cache_data
def date_bounds(col: str) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        mn, mx = con.execute(
            f"SELECT MIN(CAST({col} AS TIMESTAMP)), MAX(CAST({col} AS TIMESTAMP)) FROM v"
        ).fetchone()
        if mn and mx:
            return pd.to_datetime(mn), pd.to_datetime(mx)
    except Exception:
        pass
    return None

@st.cache_data
def distinct_clean(expr_sql: str) -> List[str]:
    df = con.execute(
        f"SELECT DISTINCT {expr_sql} AS val FROM v_enriched "
        f"WHERE {expr_sql} IS NOT NULL ORDER BY 1"
    ).fetchdf()
    return df["val"].tolist()

def shorten(label: str, max_len: int = 28) -> str:
    s = str(label)
    return s if len(s) <= max_len else s[: max_len - 1] + "…"

# =============================================================================
# Sidebar filters
# =============================================================================
st.sidebar.header("Filters")

ld = date_bounds("load_ts")
date_range = st.sidebar.date_input(
    "Load date range", (ld[0].date(), ld[1].date()) if ld else None
) if ld else None

sel_pubs = (
    st.sidebar.multiselect("Publication", distinct_clean("publication_clean"))
    if "publication_name" in COLUMNS
    else []
)
sel_channels = (
    st.sidebar.multiselect("Channel", distinct_clean("COALESCE(channel_clean, channel_name_clean)"))
    if {"channel", "channel_name", "source_type"} & COLUMNS
    else []
)
sel_bands = (
    st.sidebar.multiselect("Sentiment band", distinct_clean("sentiment_band"))
    if "sentiment_band" in COLUMNS
    else []
)
sel_authors = (
    st.sidebar.multiselect("Author", distinct_clean("COALESCE(author_clean, author_name_clean)"))
    if {"author", "author_name"} & COLUMNS
    else []
)
sel_topics = (
    st.sidebar.multiselect("Topic", distinct_clean("COALESCE(topic_clean, topics_clean)"))
    if {"topic", "topics"} & COLUMNS
    else []
)
row_limit = st.sidebar.number_input("Rows in tables", 100, 10000, 1000, 50)

def where_from_filters() -> Tuple[str, Dict]:
    clauses, args = ["1=1"], {}
    if date_range and len(date_range) == 2 and date_range[0] and date_range[1]:
        dmin = pd.to_datetime(date_range[0])
        dmax = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)
        clauses.append("CAST(load_ts AS TIMESTAMP) >= $dmin AND CAST(load_ts AS TIMESTAMP) < $dmax")
        args.update(dmin=dmin, dmax=dmax)
    if sel_pubs:
        clauses.append("publication_clean IN $pubs"); args["pubs"] = sel_pubs
    if sel_channels:
        clauses.append("COALESCE(channel_clean, channel_name_clean) IN $chs"); args["chs"] = sel_channels
    if sel_bands:
        clauses.append("sentiment_band IN $bands"); args["bands"] = sel_bands
    if sel_authors:
        clauses.append("COALESCE(author_clean, author_name_clean) IN $auths"); args["auths"] = sel_authors
    if sel_topics:
        clauses.append("COALESCE(topic_clean, topics_clean) IN $topics"); args["topics"] = sel_topics
    return " AND ".join(clauses), args

# =============================================================================
# Tabs
# =============================================================================
tab_help, tab_attr, tab_dash, tab_net = st.tabs(
    ["ℹ️ Instructions", "🧮 Attribution", "📊 Dashboard", "🕸️ Network"]
)

# ---------- Instructions ----------
with tab_help:
    st.subheader("How to use this app")
    st.markdown(
        """
**Filters (left sidebar)**  
Narrow the dataset by date, publication, channel, sentiment, author, and topic — all charts and tables update instantly.

**Attribution tab**  
*Item* → pick a **dimension** (e.g., publication_name) & a **value**; see its credit/credit_share/rating, interpretation, and matching rows.  
*Term* → lookup a keyword from the attribution table and see matching articles.

**Dashboard tab**  
KPI cards → totals; two side-by-side bars (avg influence & volume); pie (influence share); **Sankey** flow (choose source/target); conversions over time; filtered sample table.

**Network tab**  
Shows a sample of `network_edges.csv` (columns: source, target, weight) & top nodes by strength if present.
"""
    )

# ---------- Attribution ----------
with tab_attr:
    st.subheader("Attribution Lookups")
    lookup_type = st.radio("Lookup type", ["Item", "Term"], horizontal=True)

    if lookup_type == "Item":
        try:
            dims = (
                con.execute(
                    "SELECT DISTINCT dimension FROM v_item_attr WHERE dimension IS NOT NULL ORDER BY 1"
                )
                .fetchdf()["dimension"]
                .tolist()
            )
        except Exception:
            dims = []
        if not dims:
            st.info("No item attribution available.")
        else:
            sel_dim = st.selectbox(
                "Dimension", dims, index=(dims.index("publication_name") if "publication_name" in dims else 0)
            )
            values = (
                con.execute(
                    "SELECT DISTINCT value FROM v_item_attr WHERE dimension=$d AND value IS NOT NULL ORDER BY 1",
                    {"d": sel_dim},
                )
                .fetchdf()["value"]
                .tolist()
            )
            if values:
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
                st.dataframe(score_df, use_container_width=True)

                peers_df = con.execute(
                    "SELECT dimension, value, credit FROM v_item_attr WHERE dimension=$d", {"d": sel_dim}
                ).fetchdf()
                if not score_df.empty:
                    st.markdown("#### What this means")
                    st.info(explain_attribution(score_df.iloc[0], peers_df))

                # Matching rows in v (only if the chosen dimension exists)
                if sel_dim in COLUMNS:
                    id_col = _quote_ident(sel_dim)
                    w, args = where_from_filters()
                    rows = con.execute(
                        f"SELECT * FROM v WHERE {w} AND {id_col}=$v LIMIT $lim",
                        {**args, "v": sel_val, "lim": int(row_limit)},
                    ).fetchdf()
                    st.dataframe(rows, use_container_width=True)
                else:
                    st.info(f"'{sel_dim}' isn’t a column in `v`, so matching rows can’t be shown.")
            else:
                st.info("No values for this dimension.")
    else:
        term = st.text_input("Type a term to search")
        if term:
            tscore = con.execute(
                "SELECT value, credit, credit_share, rating FROM v_term_attr WHERE value=$v ORDER BY credit_share DESC",
                {"v": term},
            ).fetchdf()
            st.dataframe(tscore, use_container_width=True)
            peers = con.execute("SELECT 'term' AS dimension, value, credit FROM v_term_attr").fetchdf()
            if not tscore.empty:
                row = tscore.iloc[0].copy()
                row["dimension"] = "term"
                st.markdown("#### What this means")
                st.info(explain_attribution(row, peers))

            w, args = where_from_filters()
            text = "LOWER(COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,''))"
            hits = con.execute(
                f"SELECT * FROM v WHERE {w} AND {text} LIKE $pat LIMIT $lim",
                {**args, "pat": f"%{term.lower()}%", "lim": int(row_limit)},
            ).fetchdf()
            st.dataframe(hits, use_container_width=True)

# ---------- Dashboard ----------
with tab_dash:
    st.subheader("Dashboard")
    w, args = where_from_filters()

    total_pubs = con.execute(
        f"SELECT COUNT(DISTINCT publication_clean) FROM v_enriched WHERE {w}", args
    ).fetchone()[0]
    uniq_sources = con.execute(
        f"SELECT COUNT(DISTINCT COALESCE(source_name_clean, publication_clean)) FROM v_enriched WHERE {w}",
        args,
    ).fetchone()[0]
    uniq_authors = con.execute(
        f"SELECT COUNT(DISTINCT COALESCE(author_clean, author_name_clean)) FROM v_enriched WHERE {w}",
        args,
    ).fetchone()[0]
    infl_col = (
        "pub_credit_share" if "pub_credit_share" in COLUMNS else ("credit_share" if "credit_share" in COLUMNS else None)
    )
    avg_infl = (
        con.execute(f"SELECT AVG({infl_col}) FROM v WHERE {w}", args).fetchone()[0] if infl_col else None
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total publications", f"{total_pubs:,}")
    m2.metric("Avg influence score", f"{avg_infl:.3f}" if avg_infl is not None else "n/a")
    m3.metric("Unique sources", f"{uniq_sources:,}")
    m4.metric("Unique authors", f"{uniq_authors:,}")

    st.divider()

    cat_cols = [c for c in ["publication_name", "source_name", "channel_name", "author_name", "topic", "sentiment_band"] if c in COLUMNS]
    if not cat_cols:
        st.info("No categorical columns to group by.")
    else:
        dim = st.selectbox("Group charts by", cat_cols, index=0)

        circ_col = next((c for c in ["circulation", "circulation_size", "reach", "impressions", "audience"] if c in COLUMNS), None)
        circ_sql = f"COALESCE(SUM({circ_col}),0)" if circ_col else "COUNT(*)"

        agg = con.execute(
            f"""
            SELECT {dim} AS dim,
                   {('AVG(' + infl_col + ')') if infl_col else 'NULL'} AS avg_influence,
                   COUNT(*) AS n,
                   {circ_sql} AS total_metric
            FROM v WHERE {w}
            GROUP BY 1 HAVING dim IS NOT NULL
            """,
            args,
        ).fetchdf()

        top_n = st.slider("Top N", 5, 50, 20, 1)

        cA, cB = st.columns(2)
        if not agg.empty:
            b1 = (
                alt.Chart(agg.sort_values("avg_influence", ascending=False).head(top_n))
                .mark_bar(color=BRAND["primary"])
                .encode(
                    y=alt.Y("dim:N", sort="-x", title=None),
                    x=alt.X("avg_influence:Q", title="Avg influence"),
                    tooltip=["dim", alt.Tooltip("avg_influence:Q", format=".3f"), "n"],
                )
            )
            b2 = (
                alt.Chart(agg.sort_values("n", ascending=False).head(top_n))
                .mark_bar(color=BRAND["accent"])
                .encode(
                    y=alt.Y("dim:N", sort="-x", title=None),
                    x=alt.X("n:Q", title="Count"),
                    tooltip=["dim", "n", alt.Tooltip("avg_influence:Q", format=".3f")],
                )
            )
            cA.altair_chart(b1.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
            cB.altair_chart(b2.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
        else:
            st.info("No data for current filters.")

        st.divider()

        if infl_col and not agg.empty:
            pie_df = agg.sort_values("avg_influence", ascending=False).head(20)
            fig_pie = px.pie(
                pie_df,
                names="dim",
                values="avg_influence",
                color_discrete_sequence=[BRAND["primary"], BRAND["accent"], "#CFECE4", "#E7F6F1"],
            )
            fig_pie.update_traces(textinfo="percent+label", pull=[0.02] * len(pie_df))
            fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
            st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # --------- Sankey (readable, no label outline) ----------
        left, right = st.columns(2)
        src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
        tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols) - 1), key="sank_tgt")

        c1, c2, c3, c4 = st.columns(4)
        top_sources = c1.slider("Top Sources", 3, 50, 15, 1)
        top_targets = c2.slider("Top Targets", 2, 20, 6, 1)
        max_links = c3.slider("Max Links", 10, 500, 120, 10)
        bucket_other = c4.checkbox("Bucket 'Other'", value=True)

        if src != tgt:
            src_rank = con.execute(
                f"SELECT {src} AS s, COUNT(*) AS n FROM v WHERE {w} AND {src} IS NOT NULL "
                f"GROUP BY 1 ORDER BY n DESC LIMIT {int(top_sources)}",
                args,
            ).fetchdf()
            tgt_rank = con.execute(
                f"SELECT {tgt} AS t, COUNT(*) AS n FROM v WHERE {w} AND {tgt} IS NOT NULL "
                f"GROUP BY 1 ORDER BY n DESC LIMIT {int(top_targets)}",
                args,
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
                {**args, "ks": list(keep_s), "kt": list(keep_t)},
            ).fetchdf()

            if not bucket_other:
                sdata = sdata[(sdata["s"].isin(keep_s)) & (sdata["t"].isin(keep_t))]

            if not sdata.empty:
                nodes_all = pd.Series(pd.concat([sdata["s"], sdata["t"]])).astype(str).unique().tolist()
                labels_short = [shorten(n) for n in nodes_all]
                index = {n: i for i, n in enumerate(nodes_all)}

                fig = go.Figure(
                    go.Sankey(
                        arrangement="snap",
                        node=dict(
                            label=labels_short,
                            pad=24,
                            thickness=22,
                            color=["#CFECE4"] * len(nodes_all),  # light nodes for readability
                            line=dict(color="#CFECE4", width=0),  # **no border/outline**
                        ),
                        link=dict(
                            source=[index[s] for s in sdata["s"]],
                            target=[index[t] for t in sdata["t"]],
                            value=sdata["v"],
                            color="rgba(18,113,93,0.22)",
                            hovertemplate="Count: %{value:,}<br>source: %{source.label}<br>target: %{target.label}<extra></extra>",
                        ),
                    )
                )
                fig.update_layout(
                    font=dict(family="Inter, Helvetica, Arial, sans-serif", size=15, color=BRAND["text"]),
                    hoverlabel=dict(font_size=13, font_family="Inter, Helvetica, Arial, sans-serif"),
                    margin=dict(l=6, r=6, t=6, b=6),
                    height=640,
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Not enough data for Sankey with these fields.")
        else:
            st.info("Choose different fields for source and target.")

        st.divider()

        # --------- Conversions over time ----------
        time_col = (
            "conversion_ts"
            if "conversion_ts" in COLUMNS
            else ("load_ts" if "load_ts" in COLUMNS else ("source_time" if "source_time" in COLUMNS else None))
        )
        conv_col = "conversions" if "conversions" in COLUMNS else None
        if time_col:
            conv = con.execute(
                f"""
                SELECT DATE_TRUNC('day', CAST({time_col} AS TIMESTAMP)) AS d,
                       {('SUM(' + conv_col + ')') if conv_col else 'COUNT(*)'} AS y
                FROM v WHERE {w}
                GROUP BY 1 ORDER BY 1
                """,
                args,
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
        sample = con.execute(
            f"SELECT * FROM v WHERE {w} LIMIT $lim", {**args, "lim": int(row_limit)}
        ).fetchdf()
        st.dataframe(sample, use_container_width=True, height=360)

# ---------- Network ----------
with tab_net:
    st.subheader("Network Preview")
    net_path = next(
        (p for p in [ROOT / "data/processed/network_edges.csv", ROOT / "data/network_edges.csv", APP_DIR / "network_edges.csv"] if p.exists()),
        None,
    )
    if net_path:
        edges = pd.read_csv(net_path)
        if {"source", "target", "weight"}.issubset(edges.columns):
            max_rows = st.slider("Max edges to show", 200, 10000, 2000, 200)
            edges = edges.nlargest(max_rows, "weight")
            st.dataframe(edges.head(500), use_container_width=True, height=360)
            deg = pd.concat(
                [edges.groupby("source")["weight"].sum().rename("out"), edges.groupby("target")["weight"].sum().rename("in")],
                axis=1,
            ).fillna(0)
            deg["strength"] = deg["in"] + deg["out"]
            st.markdown("**Top nodes by strength**")
            st.dataframe(deg.sort_values("strength", ascending=False).head(50), use_container_width=True)
        else:
            st.warning("network_edges.csv must have columns: source, target, weight")
    else:
        st.info("Place a network_edges.csv in data/processed/ to enable network view.")

# =============================================================================
# Footer
# =============================================================================
try:
    n_rows = con.execute("SELECT COUNT(*) FROM v").fetchone()[0]
except Exception:
    n_rows = 0
st.caption(f"Rows in main dataset: {n_rows:,}.")