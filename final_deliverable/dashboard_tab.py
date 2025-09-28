# dashboard_tab.py — KPI cards, bars, pie, Sankey, time series, sample

from __future__ import annotations
import streamlit as st
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go

from structure import shorten

@st.cache_data
def _get_kpi_metrics(con, where_sql, args, COLUMNS):
    """Calculate KPI metrics with caching."""
    try:
        # Total publications
        total_pubs = con.execute(
            f"SELECT COUNT(DISTINCT publication_clean) FROM v_enriched WHERE {where_sql}", args
        ).fetchone()[0]
        
        # Unique sources
        uniq_sources = con.execute(
            f"SELECT COUNT(DISTINCT COALESCE(source_name_clean, publication_clean)) FROM v_enriched WHERE {where_sql}", args
        ).fetchone()[0]
        
        # Unique authors
        uniq_authors = con.execute(
            f"SELECT COUNT(DISTINCT COALESCE(author_clean, author_name_clean)) FROM v_enriched WHERE {where_sql}", args
        ).fetchone()[0]
        
        # Average influence
        infl_col = (
            "pub_credit_share" if "pub_credit_share" in COLUMNS 
            else ("credit_share" if "credit_share" in COLUMNS else None)
        )
        avg_infl = None
        if infl_col:
            avg_infl = con.execute(f"SELECT AVG({infl_col}) FROM v WHERE {where_sql}", args).fetchone()[0]
        
        return {
            "total_pubs": total_pubs,
            "uniq_sources": uniq_sources,
            "uniq_authors": uniq_authors,
            "avg_infl": avg_infl,
            "infl_col": infl_col
        }
    except Exception as e:
        st.error(f"Error calculating KPIs: {e}")
        return {
            "total_pubs": 0,
            "uniq_sources": 0,
            "uniq_authors": 0,
            "avg_infl": None,
            "infl_col": None
        }

@st.cache_data
def _get_aggregated_data(con, where_sql, args, dim, infl_col, COLUMNS):
    """Get aggregated data for charts with caching."""
    try:
        circ_col = next((c for c in ["circulation", "circulation_size", "reach", "impressions", "audience"] if c in COLUMNS), None)
        circ_sql = f"COALESCE(SUM({circ_col}),0)" if circ_col else "COUNT(*)"
        
        agg = con.execute(
            f"""
            SELECT {dim} AS dim,
                   {('AVG(' + infl_col + ')') if infl_col else 'NULL'} AS avg_influence,
                   COUNT(*) AS n,
                   {circ_sql} AS total_metric
            FROM v WHERE {where_sql}
            GROUP BY 1 HAVING dim IS NOT NULL
            """,
            args,
        ).fetchdf()
        return agg
    except Exception as e:
        st.error(f"Error getting aggregated data: {e}")
        return pd.DataFrame()

def _create_bar_chart(data, x_col, y_col, title, color, top_n):
    """Create a bar chart with consistent styling."""
    chart_data = data.sort_values(y_col, ascending=False).head(top_n)
    
    return (
        alt.Chart(chart_data)
        .mark_bar(color=color)
        .encode(
            y=alt.Y(f"{x_col}:N", sort="-x", title=None),
            x=alt.X(f"{y_col}:Q", title=title),
            tooltip=[x_col, alt.Tooltip(f"{y_col}:Q", format=".3f"), "n"],
        )
    )

def _create_pie_chart(data, names_col, values_col, BRAND):
    """Create a pie chart with consistent styling."""
    pie_df = data.sort_values(values_col, ascending=False).head(20)
    
    fig = px.pie(
        pie_df,
        names=names_col,
        values=values_col,
        color_discrete_sequence=[BRAND["primary"], BRAND["accent"], "#CFECE4", "#E7F6F1"],
    )
    fig.update_traces(textinfo="percent+label", pull=[0.02] * len(pie_df))
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
    return fig

def _render_kpi_metrics(metrics, BRAND):
    """Render KPI metrics in columns."""
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total publications", f"{metrics['total_pubs']:,}")
    m2.metric("Avg influence score", f"{metrics['avg_infl']:.3f}" if metrics['avg_infl'] is not None else "n/a")
    m3.metric("Unique sources", f"{metrics['uniq_sources']:,}")
    m4.metric("Unique authors", f"{metrics['uniq_authors']:,}")

def _render_bar_charts(agg, top_n, BRAND):
    """Render bar charts for influence and count."""
    cA, cB = st.columns(2)
    
    b1 = _create_bar_chart(agg, "dim", "avg_influence", "Avg influence", BRAND["primary"], top_n)
    b2 = _create_bar_chart(agg, "dim", "n", "Count", BRAND["accent"], top_n)
    
    cA.altair_chart(b1.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
    cB.altair_chart(b2.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)

@st.cache_data
def _get_sankey_data(con, where_sql, args, src, tgt, top_sources, top_targets, max_links):
    """Get data for Sankey diagram with caching."""
    try:
        # Get top sources and targets
        src_rank = con.execute(
            f"SELECT {src} AS s, COUNT(*) AS n FROM v WHERE {where_sql} AND {src} IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT {int(top_sources)}",
            args,
        ).fetchdf()
        tgt_rank = con.execute(
            f"SELECT {tgt} AS t, COUNT(*) AS n FROM v WHERE {where_sql} AND {tgt} IS NOT NULL GROUP BY 1 ORDER BY n DESC LIMIT {int(top_targets)}",
            args,
        ).fetchdf()
        
        keep_s = set(src_rank["s"].dropna().astype(str))
        keep_t = set(tgt_rank["t"].dropna().astype(str))
        
        # Get Sankey data
        sdata = con.execute(
            f"""
            SELECT
              CASE WHEN {src} IN $ks THEN {src} ELSE 'Other' END AS s,
              CASE WHEN {tgt} IN $kt THEN {tgt} ELSE 'Other' END AS t,
              COUNT(*) AS v
            FROM v
            WHERE {where_sql} AND {src} IS NOT NULL AND {tgt} IS NOT NULL
            GROUP BY 1,2
            ORDER BY v DESC
            LIMIT {int(max_links)}
            """,
            {**args, "ks": list(keep_s), "kt": list(keep_t)},
        ).fetchdf()
        
        return sdata, keep_s, keep_t
    except Exception as e:
        st.error(f"Error getting Sankey data: {e}")
        return pd.DataFrame(), set(), set()

def _create_sankey_diagram(sdata, keep_s, keep_t, bucket_other, BRAND):
    """Create Sankey diagram with consistent styling."""
    if sdata.empty:
        return None
    
    # Filter data if not bucketing others
    if not bucket_other:
        sdata = sdata[(sdata["s"].isin(keep_s)) & (sdata["t"].isin(keep_t))]
    
    if sdata.empty:
        return None
    
    # Create nodes and labels
    nodes_all = pd.Series(pd.concat([sdata["s"], sdata["t"]])).astype(str).unique().tolist()
    labels_short = [shorten(n) for n in nodes_all]
    index = {n: i for i, n in enumerate(nodes_all)}
    
    # Create Sankey figure
    fig = go.Figure(
        go.Sankey(
            arrangement="snap",
            node=dict(
                label=labels_short,
                pad=24,
                thickness=22,
                color=["#CFECE4"] * len(nodes_all),
                line=dict(color="#CFECE4", width=0),
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
    
    return fig

def _render_sankey_section(con, where_sql, args, cat_cols, BRAND):
    """Render Sankey diagram section."""
    left, right = st.columns(2)
    src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
    tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols) - 1), key="sank_tgt")
    
    c1, c2, c3, c4 = st.columns(4)
    top_sources = c1.slider("Top Sources", 3, 50, 15, 1)
    top_targets = c2.slider("Top Targets", 2, 20, 6, 1)
    max_links = c3.slider("Max Links", 10, 500, 120, 10)
    bucket_other = c4.checkbox("Bucket 'Other'", value=True)
    
    if src == tgt:
        st.info("Choose different fields for source and target.")
        return
    
    sdata, keep_s, keep_t = _get_sankey_data(con, where_sql, args, src, tgt, top_sources, top_targets, max_links)
    fig = _create_sankey_diagram(sdata, keep_s, keep_t, bucket_other, BRAND)
    
    if fig:
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Not enough data for Sankey with these fields.")

@st.cache_data
def _get_sample_data(con, where_sql, args, row_limit):
    """Get sample data with caching."""
    try:
        return con.execute(f"SELECT * FROM v WHERE {where_sql} LIMIT $lim", {**args, "lim": int(row_limit)}).fetchdf()
    except Exception as e:
        st.error(f"Error getting sample data: {e}")
        return pd.DataFrame()

def render(con, COLUMNS, filters, BRAND):
    """Main render function with optimized structure."""
    st.subheader("Dashboard")
    
    w, args = filters["where_sql"], filters["args"]
    row_limit = int(filters["row_limit"])
    
    # Get KPI metrics
    metrics = _get_kpi_metrics(con, w, args, COLUMNS)
    _render_kpi_metrics(metrics, BRAND)
    
    st.divider()
    
    # Get categorical columns for grouping
    cat_cols = [c for c in ["publication_name", "source_name", "channel_name", "author_name", "topic", "sentiment_band"] if c in COLUMNS]
    if not cat_cols:
        st.info("No categorical columns to group by.")
        return
    
    # Select dimension for grouping
    dim = st.selectbox("Group charts by", cat_cols, index=0)
    
    # Get aggregated data
    agg = _get_aggregated_data(con, w, args, dim, metrics["infl_col"], COLUMNS)
    
    if agg.empty:
        st.info("No data for current filters.")
        return
    
    # Render bar charts
    top_n = st.slider("Top N", 5, 50, 20, 1)
    _render_bar_charts(agg, top_n, BRAND)
    
    st.divider()
    
    # Render pie chart if influence data is available
    if metrics["infl_col"] and not agg.empty:
        fig_pie = _create_pie_chart(agg, "dim", "avg_influence", BRAND)
        st.plotly_chart(fig_pie, use_container_width=True)
    
    st.divider()
    
    # Render Sankey diagram
    _render_sankey_section(con, w, args, cat_cols, BRAND)
    
    st.divider()
    
    # Render sample data
    st.markdown("### Filtered sample")
    sample = _get_sample_data(con, w, args, row_limit)
    st.dataframe(sample, use_container_width=True, height=360)