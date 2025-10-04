# dashboard_analytics.py — Analytics & Dashboard Functions

from typing import Dict, List, Optional, Tuple
import pandas as pd
import altair as alt
import plotly.graph_objects as go
import streamlit as st

from config import COLOR_SCHEME, CHART_CONFIG, DATA_CONFIG
from ui_formatting import render_metric_card

def render_kpi_metrics(df: pd.DataFrame, df_attr: Optional[pd.DataFrame] = None) -> None:
    """Render key performance indicator metrics."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        # Total Publications
        pub_cols = [c for c in df.columns if "publication" in c.lower()]
        total_pubs = df[pub_cols[0]].nunique() if pub_cols else 0
        render_metric_card("Total Publications", f"{total_pubs:,}")
    
    with col2:
        # Average Influence Score
        influence_cols = ["pub_credit_share", "max_term_credit", "credit_share"]
        avg_influence = None
        
        for col in influence_cols:
            if col in df.columns:
                avg_influence = df[col].mean()
                break
        
        if avg_influence is None and df_attr is not None and "credit_share" in df_attr.columns:
            avg_influence = df_attr["credit_share"].mean()
            st.info("📊 Using attribution dataset for influence metrics")
        
        influence_display = f"{avg_influence:.3f}" if avg_influence is not None else "n/a"
        render_metric_card("Avg Influence Score", influence_display)
    
    with col3:
        # Unique Sources
        source_cols = [c for c in df.columns if "source" in c.lower()]
        unique_sources = df[source_cols[0]].nunique() if source_cols else 0
        render_metric_card("Unique Sources", f"{unique_sources:,}")
    
    with col4:
        # Unique Authors
        author_cols = [c for c in df.columns if "author" in c.lower()]
        unique_authors = df[author_cols[0]].nunique() if author_cols else 0
        render_metric_card("Unique Authors", f"{unique_authors:,}")

def render_chart_with_grouping(
    df: pd.DataFrame, 
    group_column: str, 
    value_column: str = None,
    top_n: int = 20,
    chart_type: str = "count"
) -> None:
    """Render charts grouped by specified column."""
    if group_column not in df.columns:
        st.error(f"Column '{group_column}' not found in data")
        return
    
    # Remove null values
    df_clean = df[df[group_column].notna()].copy()
    
    if df_clean.empty:
        st.warning("No data available for grouping")
        return
    
    # Create aggregation
    if value_column and value_column in df.columns:
        agg_df = df_clean.groupby(group_column)[value_column].sum().reset_index()
        agg_df = agg_df.rename(columns={value_column: 'total_value'})
        tooltip_col = 'total_value'
        title = f"Total {value_column}"
    else:
        agg_df = df_clean.groupby(group_column).size().reset_index()
        agg_df = agg_df.rename(columns={0: 'count'})
        agg_df = agg_df.sort_values('count', ascending=False)
        tooltip_col = 'count'
        title = "Count"
    
    # Sort by values and take top N
    agg_df = agg_df.head(top_n)
    
    # Create chart
    chart = alt.Chart(agg_df).mark_bar(
        color=COLOR_SCHEME["primary"]
    ).encode(
        y=alt.Y(f'{group_column}:N', sort='-x', title=None),
        x=alt.X(f'{tooltip_col}:Q', title=title),
        tooltip=[group_column, alt.Tooltip(f'{tooltip_col}:Q', format='.0f')]
    ).properties(
        height=CHART_CONFIG["default_height"]
    )
    
    st.altair_chart(chart, use_container_width=True)

def render_sankey_chart(df: pd.DataFrame, source_col: str, target_col: str) -> None:
    """Render a Sankey flow chart."""
    if source_col not in df.columns or target_col not in df.columns:
        st.error(f"Source or target column not found")
        return
    
    # Remove null values
    df_clean = df[df[source_col].notna() & df[target_col].notna()].copy()
    
    if df_clean.empty:
        st.warning("No data available for Sankey chart")
        return
    
    # Count relationships
    edge_counts = df_clean.groupby([source_col, target_col]).size().reset_index(name='count')
    
    # Get nodes
    all_nodes = list(set(edge_counts[source_col].unique()) | set(edge_counts[target_col].unique()))
    node_map = {node: i for i, node in enumerate(all_nodes)}
    
    # Create Sankey figure
    fig = go.Figure(go.Sankey(
        arrangement="snap",
        node=dict(
            label=[str(node)[:20] + "..." if len(str(node)) > 20 else str(node) for node in all_nodes],
            pad=CHART_CONFIG["sankey_config"]["node_padding"],
            thickness=CHART_CONFIG["sankey_config"]["link_thickness"],
            color=[COLOR_SCHEME["light"]] * len(all_nodes),
            line=dict(color="rgba(0,0,0,0)", width=0),
        ),
        link=dict(
            source=[node_map[row[source_col]] for _, row in edge_counts.iterrows()],
            target=[node_map[row[target_col]] for _, row in edge_counts.iterrows()],
            value=edge_counts['count'],
            color="rgba(18,113,93,0.22)",
            hovertemplate="Count: %{value:,}<br>source: %{source.label}<br>target: %{target.label}<extra></extra>",
        ),
    ))
    
    fig.update_layout(
        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=16, color="#133C35"),
        hoverlabel=dict(font_size=13, font_family="Inter, Helvetica, Arial, sans-serif"),
        margin=dict(l=8, r=8, t=6, b=6),
        height=640
    )
    
    st.plotly_chart(fig, use_container_width=True)

def render_time_series(df: pd.DataFrame, date_column: str) -> None:
    """Render time series chart."""
    if date_column not in df.columns:
        st.error(f"Date column '{date_column}' not found")
        return
    
    try:
        df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
        df_clean = df[df[date_column].notna()].copy()
        
        if df_clean.empty:
            st.warning("No valid dates found")
            return
        
        # Group by date
        daily_counts = df_clean.groupby(df_clean[date_column].dt.date).size().reset_index(name='count')
        daily_counts = daily_counts.rename(columns={date_column: 'date'})
        
        # Create chart
        chart = alt.Chart(daily_counts).mark_bar(
            color=COLOR_SCHEME["primary"]
        ).encode(
            x=alt.X('date:T', title='Date'),
            y=alt.Y('count:Q', title='Count')
        ).properties(
            height=CHART_CONFIG["default_height"]
        ).interactive()
        
        st.altair_chart(chart, use_container_width=True)
        
    except Exception as e:
        st.error(f"Error creating time series chart: {e}")

def get_chart_column_options(df: pd.DataFrame) -> Dict[str, List[str]]:
    """Get available columns for charting grouped by type."""
    columns = df.columns.tolist()
    
    # Categorical columns
    categorical = [c for c in columns if df[c].dtype == 'object' or df[c].dtype.name == 'category']
    
    # Numeric columns  
    numeric = df.select_dtypes(include=['number']).columns.tolist()
    
    # Date columns
    date_cols = [c for c in columns if 'date' in c.lower() or 'time' in c.lower()]
    
    return {
        'categorical': categorical,
        'numeric': numeric,
        'date': date_cols
    }

def render_filter_controls(df: pd.DataFrame) -> Dict:
    """Render filter controls and return current filter values."""
    col_options = get_chart_column_options(df)
    
    with st.expander("🎛️ Smart Filters", expanded=False):
        col1, col2 = st.columns(2)
        
        filters = {}
        
        with col1:
            # Date range filter
            if col_options['date']:
                date_col = col_options['date'][0]
                try:
                    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
                    min_date = df[date_col].min().date()
                    max_date = df[date_col].max().date()
                    
                    date_range = st.date_input(
                        "Date Range", 
                        value=(min_date, max_date),
                        min_value=min_date,
                        max_value=max_date
                    )
                    filters['date_range'] = date_range
                    filters['date_column'] = date_col
                except Exception as e:
                    st.warning(f"Error processing dates: {e}")
            
            # Publication filter
            pub_cols = [c for c in df.columns if 'publication' in c.lower()]
            if pub_cols:
                pub_col = pub_cols[0]
                publications = df[pub_col].dropna().unique()[:50]
                selected_pubs = st.multiselect("Publications", publications)
                filters['publications'] = selected_pubs
                filters['publication_column'] = pub_col
        
        with col2:
            # Author filter
            author_cols = [c for c in df.columns if 'author' in c.lower()]
            if author_cols:
                author_col = author_cols[0]
                authors = df[author_col].dropna().unique()[:50] 
                selected_authors = st.multiselect("Authors", authors)
                filters['authors'] = selected_authors
                filters['author_column'] = author_col
            
            # Channel filter
            channel_cols = [c for c in df.columns if 'channel' in c.lower()]
            if channel_cols:
                channel_col = channel_cols[0]
                channels = df[channel_col].dropna().unique()[:50]
                selected_channels = st.multiselect("Channels", channels)
                filters['channels'] = selected_channels
                filters['channel_column'] = channel_col
    
    return filters

def apply_filters(df: pd.DataFrame, filters: Dict) -> pd.DataFrame:
    """Apply filters to dataframe."""
    filtered_df = df.copy()
    
    # Apply date range filter
    if 'date_range' in filters and 'date_column' in filters:
        date_range = filters['date_range']
        date_col = filters['date_column']
        
        if len(date_range) == 2:
            start_date, end_date = date_range
            filtered_df = filtered_df[
                (filtered_df[date_col] >= pd.Timestamp(start_date)) &
                (filtered_df[date_col] <= pd.Timestamp(end_date))
            ]
    
    # Apply publication filter
    if 'publications' in filters and 'publication_column' in filters:
        publications = filters['publications']
        pub_col = filters['publication_column']
        if publications:
            filtered_df = filtered_df[filtered_df[pub_col].isin(publications)]
    
    # Apply author filter
    if 'authors' in filters and 'author_column' in filters:
        authors = filters['authors']
        author_col = filters['author_column']
        if authors:
            filtered_df = filtered_df[filtered_df[author_col].isin(authors)]
    
    # Apply channel filter
    if 'channels' in filters and 'channel_column' in filters:
        channels = filters['channels']
        channel_col = filters['channel_column']
        if channels:
            filtered_df = filtered_df[filtered_df[channel_col].isin(channels)]
    
    return filtered_df
