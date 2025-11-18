#!/usr/bin/env python3
"""
Streamlit App for Attribution and PCA Analysis
Compatible with your existing data_loaders.py / charts.py.
Data cleaning is handled offline in clean_people_names.py.
"""

# Import Packages 
from __future__ import annotations
import warnings
import streamlit as st
import os
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add the streamlit_app directory to the path for imports
if __name__ == "__main__" or "streamlit" in sys.modules:
    # When running as Streamlit app, add current directory to path
    app_dir = Path(__file__).parent
    if str(app_dir) not in sys.path:
        sys.path.insert(0, str(app_dir))

# Suppress Plotly deprecation warnings that Streamlit displays
warnings.filterwarnings('ignore', message='.*keyword arguments.*deprecated.*')
warnings.filterwarnings('ignore', message='.*deprecated.*will be removed.*')

try:
    from charts import (
    create_person_emotion_chart,
    create_person_sentiment_chart,
    create_person_mentions_over_time_chart,
    create_topic_sentiment_by_people_chart,
    create_topic_emotion_by_people_chart,
    create_topic_mentions_over_time_chart,
    create_circulation_quartile_chart,
    create_top_people_bar_chart,
    create_sentiment_by_cluster_chart,
    create_circulation_by_cluster_chart,
)
except ImportError:
    # Fallback for different import paths
    from streamlit_app.charts import (
        create_person_emotion_chart,
        create_person_sentiment_chart,
        create_person_mentions_over_time_chart,
        create_topic_sentiment_by_people_chart,
        create_topic_emotion_by_people_chart,
        create_topic_mentions_over_time_chart,
        create_circulation_quartile_chart,
        create_top_people_bar_chart,
        create_sentiment_by_cluster_chart,
        create_circulation_by_cluster_chart,
    )

try:
    from network_analysis import (
        build_person_network_graph,
        build_topic_categorical_network_graph,
        build_person_network_graph_interactive,
        build_topic_categorical_network_graph_interactive,
        AGraph_AVAILABLE,
    )
except ImportError:
    # Fallback for different import paths
    from streamlit_app.network_analysis import (
        build_person_network_graph,
        build_topic_categorical_network_graph,
        build_person_network_graph_interactive,
        build_topic_categorical_network_graph_interactive,
        AGraph_AVAILABLE,
    )
from typing import Optional

# Ensure current folder is importable 
APP_DIR = Path(__file__).resolve().parent

# Load the Data 
from data_loaders import (
    load_influencer_table,  
    load_final_dataset,    
    load_persons_by_row,     
)

@st.cache_data(show_spinner=False, ttl=3600)
def explode_persons(persons_by_row_df: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated persons into long form - simplified"""
    if persons_by_row_df is None or persons_by_row_df.empty:
        return pd.DataFrame(columns=["row_index", "person", "person_norm_lc"])
    df = persons_by_row_df[['row_index', 'persons']].dropna()
    df = df.assign(person=df['persons'].str.split(',')).explode('person')
    df['person'] = df['person'].str.strip()
    df = df[df['person'] != '']
    df['person_norm_lc'] = df['person'].str.lower()
    return df[['row_index', 'person', 'person_norm_lc']]

def _normalize_name_for_dedup(name: str) -> str:
    """Normalize name to handle plural forms and common variations"""
    if not name or not isinstance(name, str):
        return name
    name = name.strip()
    if not name:
        return name
    
    # Split into parts
    parts = name.split()
    if len(parts) < 2:
        return name
    
    # Check if last name ends with 's' and might be a plural form
    last_name = parts[-1]
    # Remove trailing 's' if it looks like a plural (simple heuristic)
    # Only if the name without 's' is a known variation
    if last_name.endswith('s') and len(last_name) > 3:
        # Check if removing 's' would match a more common form
        last_name_no_s = last_name[:-1]
        # This is a simple approach - in production you might want a mapping dict
        # For now, we'll create a normalized version
        normalized_parts = parts[:-1] + [last_name_no_s]
        return " ".join(normalized_parts)
    
    return name

def _get_name_search_variants(name: str) -> set:
    """Get all search variants for a name (including plural forms)"""
    if not name or not isinstance(name, str):
        return {name.lower() if name else ""}
    
    name_lower = name.strip().lower()
    variants = {name_lower}
    
    # Split into parts
    parts = name_lower.split()
    if len(parts) >= 2:
        last_name = parts[-1]
        # Add variant with 's' appended to last name
        if not last_name.endswith('s'):
            variant_with_s = " ".join(parts[:-1] + [last_name + 's'])
            variants.add(variant_with_s)
        # Add variant with 's' removed from last name
        elif last_name.endswith('s') and len(last_name) > 3:
            variant_no_s = " ".join(parts[:-1] + [last_name[:-1]])
            variants.add(variant_no_s)
    
    return variants

@st.cache_data(show_spinner=False, ttl=3600)
def get_all_people_list(pbr_long: pd.DataFrame) -> list:
    """Get cached list of all people with normalized names to deduplicate variations"""
    if pbr_long is None or pbr_long.empty:
        return []
    
    # Get all person names with counts
    person_counts = pbr_long['person'].value_counts().head(5000)
    
    # Create a mapping: normalized_name -> best_canonical_name
    # The "best" name is the one with the highest count
    name_mapping = {}
    normalized_to_names = {}
    
    for name, count in person_counts.items():
        normalized = _normalize_name_for_dedup(name).lower()
        if normalized not in normalized_to_names:
            normalized_to_names[normalized] = []
        normalized_to_names[normalized].append((name, count))
    
    # For each normalized form, pick the most common variant
    for normalized, variants in normalized_to_names.items():
        # Sort by count (descending) and take the first
        variants_sorted = sorted(variants, key=lambda x: x[1], reverse=True)
        canonical_name = variants_sorted[0][0]
        
        # Map all variants to the canonical name
        for variant_name, _ in variants:
            name_mapping[variant_name] = canonical_name
    
    # Return unique canonical names, sorted by frequency
    canonical_names = list(name_mapping.values())
    # Get counts for canonical names
    canonical_counts = {}
    for name, count in person_counts.items():
        canonical = name_mapping.get(name, name)
        canonical_counts[canonical] = canonical_counts.get(canonical, 0) + count
    
    # Sort by count and return
    sorted_canonical = sorted(canonical_counts.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_canonical]


def get_filter_options(df: Optional[pd.DataFrame], column: str, limit: int = 100) -> list[str]:
    """Return sorted string options for a column; safe if data is missing."""
    if df is None or df.empty or column not in df.columns:
        return []
    series = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    series = series[series != ""]
    return sorted(series.unique().tolist())[:limit]


# ---- end inline helpers ----

# -------------------- Branding / Theme --------------------
PENTA_COLORS = ["#12715D", "#4AB48E", "#142536", "#D4A115", "#2A9D8F", "#D94841"]
PENTA_PRIMARY = "#12715D"
PENTA_ACCENT = "#4AB48E"
PENTA_DARK = "#142536"
PENTA_GOLD = "#D4A115"
px.defaults.color_discrete_sequence = PENTA_COLORS

# -------------------- CSS --------------------
def _load_css():
    css_path = os.path.join(os.path.dirname(__file__), "style.css")
    for candidate in [css_path, "style.css"]:
        if os.path.exists(candidate):
            with open(candidate) as f:
                st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
            return
    # Not fatal
    # st.warning("CSS file not found. App will run without custom styling.")

# -------------------- Small UI helpers --------------------
def safe_table(df: pd.DataFrame, max_rows: int = 2000, height: int = 420):
    """Render a dataframe with hard row cap to avoid OOM."""
    if df is None or df.empty:
        st.info("No rows to display.")
        return
    if len(df) > max_rows:
        st.caption(f"Showing first {max_rows:,} of {len(df):,} rows")
        df = df.head(max_rows)
    st.dataframe(df, width="stretch", height=height)

@st.cache_data(show_spinner=False, ttl=3600)
def with_sentiment_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment band column if not present - simplified"""
    if df is None or df.empty:
        return pd.DataFrame()
    
    # Normalize existing sentiment_band to title case if it exists
    if 'sentiment_band' in df.columns:
        df = df.copy()
        df['sentiment_band'] = df['sentiment_band'].astype(str).str.title()
        # Handle any NaN or 'nan' strings
        df['sentiment_band'] = df['sentiment_band'].replace(['Nan', 'None', 'nan', 'none'], pd.NA)
    
    # Create sentiment_band from sentiment_score if needed
    if 'sentiment_score' in df.columns and 'sentiment_band' not in df.columns:
        df = df.copy()
        # Use appropriate thresholds for sentiment_score scale (-100 to 100)
        # Negative: < -10, Neutral: -10 to 10, Positive: > 10
        df['sentiment_band'] = pd.cut(
            df['sentiment_score'],
            bins=[-float('inf'), -10, 10, float('inf')],
            labels=['Negative', 'Neutral', 'Positive']
        )
    elif 'sentiment_score' in df.columns and 'sentiment_band' in df.columns:
        # If both exist, ensure sentiment_band is properly set for any missing values
        df = df.copy()
        missing_mask = df['sentiment_band'].isna()
        if missing_mask.any():
            # Use appropriate thresholds for sentiment_score scale (-100 to 100)
            # Negative: < -10, Neutral: -10 to 10, Positive: > 10
            df.loc[missing_mask, 'sentiment_band'] = pd.cut(
                df.loc[missing_mask, 'sentiment_score'],
                bins=[-float('inf'), -10, 10, float('inf')],
                labels=['Negative', 'Neutral', 'Positive']
            )
    
    return df

# -------------------- Main app --------------------
def main():
    st.markdown('<div class="main-header" style="font-size:2rem;font-weight:700;color:#12715D;margin-bottom:0.5rem;">PolicyPath 🏛️</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:1rem;color:#475569;margin-bottom:1.5rem;">Your indispensable guide to healthcare policy influence</p>', unsafe_allow_html=True)

    _load_css()

    # Load all data upfront - caching makes this fast on subsequent loads
    
    # Load all datasets (cached, so fast after first load)
    influencer_df = load_influencer_table()
    final_df_sample = load_final_dataset()
    persons_by_row_df = load_persons_by_row()

    # Check if DataFrames are empty and handle gracefully
    if influencer_df is None or influencer_df.empty:
        influencer_df = pd.DataFrame()
    else:
        MAX_INFLUENCER_ROWS = 200_000
        if len(influencer_df) > MAX_INFLUENCER_ROWS:
            influencer_df = influencer_df.head(MAX_INFLUENCER_ROWS)
        influencer_df = with_sentiment_band(influencer_df)
    
    if final_df_sample is None or final_df_sample.empty:
        final_df_sample = pd.DataFrame()
    
    if persons_by_row_df is None or persons_by_row_df.empty:
        persons_by_row_df = pd.DataFrame()
        pbr_long = pd.DataFrame(columns=['row_index', 'person', 'person_norm_lc'])
        all_people = []
    else:
        pbr_long = explode_persons(persons_by_row_df)
        all_people = get_all_people_list(pbr_long)

    # --------------------------------------
    # Filters Section
    # --------------------------------------
    st.markdown('<div class="section-header" style="font-size:1.5rem;font-weight:700;color:#142536;margin-top:1.5rem;margin-bottom:1rem;">Filters</div>', unsafe_allow_html=True)

    # Cluster options
    cluster_col = None
    clusters = []
    if influencer_df is not None and not influencer_df.empty:
        try:
            if 'cluster_label' in influencer_df.columns:
                clusters = sorted(influencer_df['cluster_label'].dropna().astype(str).unique().tolist())
                cluster_col = 'cluster_label'
            elif 'cluster' in influencer_df.columns:
                clusters = sorted(influencer_df['cluster'].dropna().astype(str).unique().tolist())
                cluster_col = 'cluster'
        except Exception:
            clusters = []

    author_options = get_filter_options(final_df_sample, "author_name")
    publication_options = get_filter_options(final_df_sample, "publication_name")
    source_name_options = get_filter_options(final_df_sample, "source_name")
    channel_options = get_filter_options(final_df_sample, "channel_name")
    topic_options_global = get_filter_options(final_df_sample, "tag_name")

    # First row of filters (4 columns)
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        selected_clusters = st.multiselect(
            "Select Clusters", 
            clusters, 
            default=[], 
            key="tab1_select_clusters",
            help="Filter articles by cluster groups. Clusters represent groups of articles with similar characteristics."
        ) if clusters else []
    with col2:
        sentiment_bands_available = []
        if influencer_df is not None and not influencer_df.empty and 'sentiment_band' in influencer_df.columns:
            try:
                sentiment_bands_available = sorted(influencer_df['sentiment_band'].dropna().unique().tolist())
            except Exception:
                sentiment_bands_available = []
        selected_sentiment_bands_global = st.multiselect(
            "Sentiment Band", 
            sentiment_bands_available, 
            default=[], 
            key="tab1_sentiment_band",
            help="Filter by sentiment categories: Negative (< -10), Neutral (-10 to 10), or Positive (> 10) on a -100 to 100 scale"
        )
    with col3:
        selected_authors_global = st.multiselect(
            "Authors", 
            author_options, 
            key="tab1_select_authors",
            help="Filter articles by specific authors. Select multiple authors to see articles from any of them."
        )
    with col4:
        selected_publications_global = st.multiselect(
            "Publications", 
            publication_options, 
            key="tab1_select_publications",
            help="Filter articles by publication source. Shows articles from selected publications."
        )
    
    # Second row of filters (4 columns)
    col5, col6, col7, col8 = st.columns(4)
    with col5:
        selected_source_names_global = st.multiselect(
            "Source Names", 
            source_name_options, 
            key="tab1_select_source_names",
            help="Filter by source name (e.g., specific news agencies, organizations)."
        )
    with col6:
        selected_channels_global = st.multiselect(
            "Channels", 
            channel_options, 
            key="tab1_select_channels",
            help="Filter by distribution channel (e.g., social media, news wires, trade publications)."
        )
    with col7:
        selected_topics_global = st.multiselect(
            "Topics", 
            topic_options_global, 
            key="tab1_select_topics",
            help="Filter articles by topic tags. Topics represent the main themes or subjects of articles."
        )
    with col8:
        # Date range filter - get bounds from data
        date_min = None
        date_max = None
        default_start_date = pd.Timestamp('2024-01-01').date()
        
        if final_df_sample is not None and not final_df_sample.empty and 'published_datetime' in final_df_sample.columns:
            dates = pd.to_datetime(final_df_sample['published_datetime'], errors='coerce').dropna()
            if not dates.empty:
                date_min = dates.min().date()
                date_max = dates.max().date()
                # Ensure default_start_date is within range
                if default_start_date < date_min:
                    default_start_date = date_min
                if default_start_date > date_max:
                    default_start_date = date_min
        
        if date_min and date_max:
            date_range_global = st.date_input(
                "Date Range",
                value=(default_start_date, date_max),
                min_value=date_min,
                max_value=date_max,
                key="global_date_range",
                help="Filter articles by publication date range. Default starts from 2024."
            )
            
            # Handle date range input
            try:
                if isinstance(date_range_global, tuple):
                    if len(date_range_global) == 2:
                        selected_date_start, selected_date_end = date_range_global
                    elif len(date_range_global) == 1:
                        selected_date_start = date_range_global[0]
                        selected_date_end = date_max
                    else:
                        selected_date_start = default_start_date
                        selected_date_end = date_max
                else:
                    selected_date_start = date_range_global if date_range_global else default_start_date
                    selected_date_end = date_max
                
                if selected_date_start is None:
                    selected_date_start = default_start_date
                if selected_date_end is None:
                    selected_date_end = date_max
                
                selected_date_start_ts = pd.Timestamp(selected_date_start) if selected_date_start else None
                selected_date_end_ts = pd.Timestamp(selected_date_end) if selected_date_end else None
            except Exception:
                selected_date_start_ts = pd.Timestamp(default_start_date) if default_start_date else None
                selected_date_end_ts = pd.Timestamp(date_max) if date_max else None
        else:
            selected_date_start_ts = None
            selected_date_end_ts = None

    # Apply global filters to the article-level data for use in both sub-tabs
    final_df_filtered = None
    if final_df_sample is not None and not final_df_sample.empty:
        final_df_filtered = final_df_sample.copy()
        if 'row_index' not in final_df_filtered.columns:
            final_df_filtered = final_df_filtered.reset_index().rename(columns={'index': 'row_index'})

        if selected_authors_global and 'author_name' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered[
                final_df_filtered['author_name'].astype(str).str.strip().isin(selected_authors_global)
            ]
        if selected_publications_global and 'publication_name' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered[
                final_df_filtered['publication_name'].astype(str).str.strip().isin(selected_publications_global)
            ]
        if selected_source_names_global and 'source_name' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered[
                final_df_filtered['source_name'].astype(str).str.strip().isin(selected_source_names_global)
            ]
        if selected_channels_global and 'channel_name' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered[
                final_df_filtered['channel_name'].astype(str).str.strip().isin(selected_channels_global)
            ]
        if selected_topics_global and 'tag_name' in final_df_filtered.columns:
            final_df_filtered = final_df_filtered[
                final_df_filtered['tag_name'].astype(str).str.strip().isin(selected_topics_global)
            ]
        if selected_sentiment_bands_global:
            final_df_filtered = with_sentiment_band(final_df_filtered)
            final_df_filtered = final_df_filtered[
                final_df_filtered['sentiment_band'].isin(selected_sentiment_bands_global)
            ]
        
        # Apply date range filter if provided
        if selected_date_start_ts is not None or selected_date_end_ts is not None:
            if 'published_datetime' in final_df_filtered.columns:
                final_df_filtered['published_datetime'] = pd.to_datetime(final_df_filtered['published_datetime'], errors='coerce')
                # Convert to date for comparison to avoid timezone issues
                try:
                    if final_df_filtered['published_datetime'].dt.tz is not None:
                        final_df_filtered['date_only'] = final_df_filtered['published_datetime'].dt.tz_convert('UTC').dt.date
                    else:
                        final_df_filtered['date_only'] = final_df_filtered['published_datetime'].dt.date
                    
                    if selected_date_start_ts is not None:
                        date_start_date = selected_date_start_ts.date() if isinstance(selected_date_start_ts, pd.Timestamp) else pd.to_datetime(selected_date_start_ts).date()
                        final_df_filtered = final_df_filtered[final_df_filtered['date_only'] >= date_start_date]
                    
                    if selected_date_end_ts is not None:
                        date_end_date = selected_date_end_ts.date() if isinstance(selected_date_end_ts, pd.Timestamp) else pd.to_datetime(selected_date_end_ts).date()
                        final_df_filtered = final_df_filtered[final_df_filtered['date_only'] <= date_end_date]
                    
                    if 'date_only' in final_df_filtered.columns:
                        final_df_filtered = final_df_filtered.drop(columns=['date_only'])
                except Exception:
                    if 'date_only' in final_df_filtered.columns:
                        final_df_filtered = final_df_filtered.drop(columns=['date_only'])

    filtered_row_indices = None
    if final_df_filtered is not None and not final_df_filtered.empty and 'row_index' in final_df_filtered.columns:
        filtered_row_indices = final_df_filtered['row_index'].dropna().unique()

    if persons_by_row_df is None or persons_by_row_df.empty:
        persons_by_row_filtered = persons_by_row_df
    else:
        persons_by_row_filtered = persons_by_row_df.copy()
        if filtered_row_indices is not None and len(filtered_row_indices) > 0:
            persons_by_row_filtered = persons_by_row_filtered[
                persons_by_row_filtered['row_index'].isin(filtered_row_indices)
            ]
        if selected_topics_global and 'tag_name' in persons_by_row_filtered.columns:
            persons_by_row_filtered = persons_by_row_filtered[
                persons_by_row_filtered['tag_name'].astype(str).str.strip().isin(selected_topics_global)
            ]

    if persons_by_row_filtered is None or (
        isinstance(persons_by_row_filtered, pd.DataFrame) and persons_by_row_filtered.empty
    ):
        pbr_long_filtered = pd.DataFrame(columns=['row_index', 'person', 'person_norm_lc'])
    else:
        pbr_long_filtered = explode_persons(persons_by_row_filtered)

    all_people_filtered = None
    if pbr_long_filtered is not None and not pbr_long_filtered.empty:
        all_people_filtered = get_all_people_list(pbr_long_filtered)

    # Apply filters (cluster, band) to influencer view
    influencer_view = influencer_df.copy() if influencer_df is not None and not influencer_df.empty else pd.DataFrame()

    if not influencer_view.empty:
        if selected_clusters and cluster_col and cluster_col in influencer_view.columns:
            influencer_view = influencer_view[influencer_view[cluster_col].astype(str).isin(selected_clusters)]

        if selected_sentiment_bands_global and 'sentiment_band' in influencer_view.columns:
            # sentiment_band already computed in influencer_view (from influencer_df)
            influencer_view = influencer_view[influencer_view['sentiment_band'].isin(selected_sentiment_bands_global)]

    # Summary metrics
    st.markdown('<div class="section-header" style="font-size:1.5rem;font-weight:700;color:#142536;margin-top:1.5rem;margin-bottom:1rem;">Data Summary</div>', unsafe_allow_html=True)
    if influencer_view is not None and not influencer_view.empty:
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.metric(
                "Total Individuals", 
                f"{len(influencer_view):,}", 
                help="Number of unique individuals (people) found in articles matching your current filters"
            )
        with m2:
            total_mentions = influencer_view.get('mention_count', pd.Series([0]*len(influencer_view))).sum()
            st.metric(
                "Total Mentions", 
                f"{int(total_mentions):,}", 
                help="Total number of times individuals are mentioned across all filtered articles"
            )
        with m3:
            avg_sentiment = influencer_view.get('sentiment_score', pd.Series(dtype=float)).mean()
            st.metric(
                "Avg Sentiment Score", 
                f"{(avg_sentiment if pd.notna(avg_sentiment) else 0):.2f}", 
                help="Average sentiment score ranging from -1 (very negative) to +1 (very positive). Values near 0 are neutral."
            )
        with m4:
            avg_circ = influencer_view.get('circulation_size', pd.Series(dtype=float)).mean()
            st.metric(
                "Avg Circulation", 
                f"{(avg_circ if pd.notna(avg_circ) else 0):,.0f}", 
                help="Average circulation size (readership/reach) of publications in the filtered dataset"
            )
    else:
        st.info("No data available. Please check your filters or ensure data files are loaded.")

    show_overview = st.checkbox(
        "See General Overview", 
        value=False,
        help="Display high-level visualizations including cluster analysis and top individuals across all filtered data"
    )

    if show_overview:
        st.markdown("### General Overview")

        if cluster_col and influencer_view is not None and not influencer_view.empty:
            c1, c2 = st.columns(2)
            with c1:
                fig_sentiment = create_sentiment_by_cluster_chart(influencer_view, cluster_col)
                if fig_sentiment:
                    st.plotly_chart(fig_sentiment, width="stretch")
                else:
                    st.info("No cluster/sentiment data available after filters.")

            with c2:
                fig_circ = create_circulation_by_cluster_chart(influencer_view, cluster_col)
                if fig_circ:
                    st.plotly_chart(fig_circ, width="stretch")
                else:
                    st.info("No cluster/circulation data available after filters.")
        else:
            st.info("Cluster-level insights are unavailable for the current filters.")

        st.markdown("#### Top Individuals")
        n_top_general = st.slider(
            "Number of top individuals to show",
            min_value=5,
            max_value=50,
            value=20,
            step=5,
            key="top_people_slider_overview",
            help="Adjust the number of top individuals displayed in the bar chart, ranked by number of article mentions"
        )

        if pbr_long_filtered is None or pbr_long_filtered.empty:
            st.info("Person-by-row data is required to surface the top individuals.")
        else:
            top_people_counts = pbr_long_filtered['person'].value_counts().reset_index()
            top_people_counts.columns = ['person', 'article_count']

            if top_people_counts.empty:
                st.info("No individuals found for the current selection.")
            else:
                fig_top_people = create_top_people_bar_chart(pbr_long_filtered, n_top_general)
                if fig_top_people:
                    st.plotly_chart(fig_top_people, width="stretch")
                else:
                    st.info("No individuals found for the current selection.")

    # --------------------------------------
    # Main Tabs: People and Topics
    # --------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    people_tab, topic_tab = st.tabs(["People", "Topics"])

    # ------------------------------- People Tab ------------------------------- #
    with people_tab:
        st.markdown("### Search Individual")

        people_options = all_people_filtered if all_people_filtered else all_people
        col_person, col_person2, col_keyword = st.columns([2, 2, 1])
        with col_person:
            story_person = st.selectbox(
                "Select an Individual",
                options=[""] + people_options,
                index=0,
                help="Search and select a person to analyze. The search is case-insensitive and handles name variations automatically.",
                key="tab2_story_person"
            )
        with col_person2:
            story_person2 = st.selectbox(
                "Compare with (Optional)",
                options=[""] + people_options,
                index=0,
                help="Optionally select a second person to compare side-by-side. Both individuals will be shown in the same charts with different colors.",
                key="tab2_story_person2"
            )
        with col_keyword:
            story_keyword = st.text_input(
                "Optional Keyword Filter",
                placeholder="e.g., vaccine, policy...",
                key="tab2_story_keyword",
                help="Filter articles to only those containing your keyword in the headline or article body. Useful for focusing on specific topics."
            )

        if story_person and story_person.strip():
            with st.spinner("🔍 Searching for articles..."):
                if (
                    final_df_filtered is None or final_df_filtered.empty
                    or pbr_long_filtered is None or pbr_long_filtered.empty
                ):
                    st.info("No article/person data available for the current filters.")
                else:
                    final_df_people = final_df_filtered.copy()
                    if 'row_index' not in final_df_people.columns:
                        final_df_people = final_df_people.reset_index().rename(columns={'index': 'row_index'})

                    # Get articles for first person (including name variations)
                    needle = story_person.strip().lower()
                    search_variants = _get_name_search_variants(story_person)
                    matching_rows = pbr_long_filtered.loc[
                        pbr_long_filtered['person_norm_lc'].isin(search_variants), 'row_index'
                    ].unique()
                    person_articles = (
                        final_df_people[final_df_people['row_index'].isin(matching_rows)].copy()
                        if len(matching_rows) > 0 else pd.DataFrame()
                    )

                    # Get articles for second person if selected
                    person2_articles = pd.DataFrame()
                    story_person2_clean = story_person2.strip() if story_person2 and story_person2.strip() else None
                    if story_person2_clean and story_person2_clean.lower() != needle:
                        search_variants2 = _get_name_search_variants(story_person2_clean)
                        matching_rows2 = pbr_long_filtered.loc[
                            pbr_long_filtered['person_norm_lc'].isin(search_variants2), 'row_index'
                        ].unique()
                        person2_articles = (
                            final_df_people[final_df_people['row_index'].isin(matching_rows2)].copy()
                            if len(matching_rows2) > 0 else pd.DataFrame()
                        )

                    # Apply keyword filter to both
                    if story_keyword and story_keyword.strip():
                        keyword = story_keyword.strip().lower()
                        if not person_articles.empty:
                            keyword_mask = pd.Series(False, index=person_articles.index)
                            for col in ['headline', 'article_body']:
                                if col in person_articles.columns:
                                    keyword_mask = keyword_mask | person_articles[col].astype(str).str.lower().str.contains(
                                        keyword, regex=False
                                    )
                            person_articles = person_articles[keyword_mask]
                        
                        if not person2_articles.empty:
                            keyword_mask2 = pd.Series(False, index=person2_articles.index)
                            for col in ['headline', 'article_body']:
                                if col in person2_articles.columns:
                                    keyword_mask2 = keyword_mask2 | person2_articles[col].astype(str).str.lower().str.contains(
                                        keyword, regex=False
                                    )
                            person2_articles = person2_articles[keyword_mask2]

                    if person_articles.empty:
                        st.warning(
                            f"No article data found for {story_person}"
                            + (f" with keyword '{story_keyword}'" if story_keyword and story_keyword.strip() else "")
                            + " within the current filters."
                        )
                    else:
                        # Check if comparing
                        is_comparison = story_person2_clean and not person2_articles.empty
                        
                        st.markdown("### Summary")
                        if is_comparison:
                            # Comparison view - show all Person 1 metrics first, then all Person 2 metrics
                            st.markdown(f"**{story_person}** vs **{story_person2_clean}**")
                            
                            # Calculate all values first
                            num_articles1 = len(person_articles)
                            num_articles2 = len(person2_articles)
                            avg_sentiment = person_articles.get('sentiment_score', pd.Series(dtype=float)).mean()
                            avg_sentiment2 = person2_articles.get('sentiment_score', pd.Series(dtype=float)).mean()
                            total_circ = person_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            total_circ2 = person2_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            unique_pubs = person_articles.get('publication_name', pd.Series(dtype=str)).nunique()
                            unique_pubs2 = person2_articles.get('publication_name', pd.Series(dtype=str)).nunique()
                            
                            # Person 1 row 1 - main metrics
                            summary_cols1 = st.columns(4)
                            with summary_cols1[0]:
                                st.metric(
                                    f"{story_person} - Articles", 
                                    num_articles1,
                                    help="Total number of articles mentioning this person after applying all filters"
                                )
                            with summary_cols1[1]:
                                st.metric(
                                    f"{story_person} - Avg Sentiment", 
                                    f"{(avg_sentiment if pd.notna(avg_sentiment) else 0):.2f}",
                                    help="Average sentiment score for articles mentioning this person"
                                )
                            with summary_cols1[2]:
                                st.metric(
                                    f"{story_person} - Total Reach", 
                                    f"{int(total_circ) if pd.notna(total_circ) else 0:,}",
                                    help="Sum of circulation sizes across all publications mentioning this person"
                                )
                            with summary_cols1[3]:
                                st.metric(
                                    f"{story_person} - Publishers", 
                                    int(unique_pubs) if pd.notna(unique_pubs) else 0,
                                    help="Number of distinct publications that have mentioned this person"
                                )
                        else:
                            # Single person view
                            summary_cols = st.columns(4)
                            with summary_cols[0]:
                                st.metric(
                                    "Total Articles", 
                                    len(person_articles),
                                    help="Total number of articles mentioning this person after applying all filters"
                                )
                            with summary_cols[1]:
                                avg_sentiment = person_articles.get('sentiment_score', pd.Series(dtype=float)).mean()
                                st.metric(
                                    "Avg Sentiment", 
                                    f"{(avg_sentiment if pd.notna(avg_sentiment) else 0):.2f}",
                                    help="Average sentiment score for articles mentioning this person (-1 to +1)"
                                )
                            with summary_cols[2]:
                                total_circ = person_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                                st.metric(
                                    "Total Reach", 
                                    f"{int(total_circ) if pd.notna(total_circ) else 0:,}",
                                    help="Sum of circulation sizes across all publications mentioning this person"
                                )
                            with summary_cols[3]:
                                unique_pubs = person_articles.get('publication_name', pd.Series(dtype=str)).nunique()
                                st.metric(
                                    "Unique Publishers", 
                                    int(unique_pubs) if pd.notna(unique_pubs) else 0,
                                    help="Number of distinct publications that have mentioned this person"
                                )

                        # Extra metrics section
                        if is_comparison:
                            # Comparison view - show all Person 1 metrics first, then all Person 2 metrics
                            extra_metrics_list = []
                            
                            # Average Total Reach per Article
                            total_circ1 = person_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            num_articles1 = len(person_articles)
                            total_circ2 = person2_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            num_articles2 = len(person2_articles)
                            
                            if num_articles1 > 0 and pd.notna(total_circ1) and num_articles2 > 0 and pd.notna(total_circ2):
                                avg_reach1 = total_circ1 / num_articles1
                                avg_reach2 = total_circ2 / num_articles2
                                extra_metrics_list.append(("Average Total Reach per Article", avg_reach1, avg_reach2))
                            
                            # Average Hit Strength
                            if 'hit_strength' in person_articles.columns and 'hit_strength' in person2_articles.columns:
                                hit1 = person_articles['hit_strength'].mean()
                                hit2 = person2_articles['hit_strength'].mean()
                                if pd.notna(hit1) and pd.notna(hit2):
                                    extra_metrics_list.append(("Average Hit Strength", hit1, hit2))
                            
                            # Average VIPR Score
                            if 'vipr_score' in person_articles.columns and 'vipr_score' in person2_articles.columns:
                                vipr1 = person_articles['vipr_score'].mean()
                                vipr2 = person2_articles['vipr_score'].mean()
                                if pd.notna(vipr1) and pd.notna(vipr2):
                                    extra_metrics_list.append(("Average VIPR Score", vipr1, vipr2))
                            
                            # Average VIPR Weight
                            if 'vipr_weight' in person_articles.columns and 'vipr_weight' in person2_articles.columns:
                                weight1 = person_articles['vipr_weight'].mean()
                                weight2 = person2_articles['vipr_weight'].mean()
                                if pd.notna(weight1) and pd.notna(weight2):
                                    extra_metrics_list.append(("Average VIPR Weight", weight1, weight2))
                            
                            # Person 1 row 2 - extra metrics (right after main metrics)
                            if extra_metrics_list:
                                metric_cols1 = st.columns(len(extra_metrics_list))
                                for col_obj, (label, value1, value2) in zip(metric_cols1, extra_metrics_list):
                                    with col_obj:
                                        st.metric(
                                            f"{story_person} - {label}",
                                            f"{value1:,.0f}" if pd.notna(value1) else "N/A",
                                            help=f"{label} for {story_person}"
                                        )
                            
                            # Person 2 row 1 - main metrics with deltas
                            summary_cols2 = st.columns(4)
                            with summary_cols2[0]:
                                st.metric(
                                    f"{story_person2_clean} - Articles", 
                                    num_articles2,
                                    delta=num_articles2 - num_articles1,
                                    help="Total number of articles mentioning the comparison person"
                                )
                            with summary_cols2[1]:
                                delta_sentiment = (avg_sentiment2 - avg_sentiment) if pd.notna(avg_sentiment2) and pd.notna(avg_sentiment) else None
                                st.metric(
                                    f"{story_person2_clean} - Avg Sentiment", 
                                    f"{(avg_sentiment2 if pd.notna(avg_sentiment2) else 0):.2f}",
                                    delta=f"{delta_sentiment:.2f}" if delta_sentiment is not None else None,
                                    help="Average sentiment score for articles mentioning the comparison person"
                                )
                            with summary_cols2[2]:
                                delta_circ = (total_circ2 - total_circ) if pd.notna(total_circ2) and pd.notna(total_circ) else None
                                st.metric(
                                    f"{story_person2_clean} - Total Reach", 
                                    f"{int(total_circ2) if pd.notna(total_circ2) else 0:,}",
                                    delta=f"{int(delta_circ):,}" if delta_circ is not None else None,
                                    help="Sum of circulation sizes for the comparison person"
                                )
                            with summary_cols2[3]:
                                delta_pubs = (unique_pubs2 - unique_pubs) if pd.notna(unique_pubs2) and pd.notna(unique_pubs) else None
                                st.metric(
                                    f"{story_person2_clean} - Publishers", 
                                    int(unique_pubs2) if pd.notna(unique_pubs2) else 0,
                                    delta=delta_pubs if delta_pubs is not None else None,
                                    help="Number of distinct publications mentioning the comparison person"
                                )
                            
                            # Person 2 row 2 - extra metrics with deltas
                            if extra_metrics_list:
                                metric_cols2 = st.columns(len(extra_metrics_list))
                                for col_obj, (label, value1, value2) in zip(metric_cols2, extra_metrics_list):
                                    with col_obj:
                                        delta_val = value2 - value1 if pd.notna(value1) and pd.notna(value2) else None
                                        if label == "Average Total Reach per Article":
                                            delta_display = f"{int(delta_val):,}" if delta_val is not None else None
                                        else:
                                            delta_display = f"{delta_val:,.0f}" if delta_val is not None else None
                                        st.metric(
                                            f"{story_person2_clean} - {label}",
                                            f"{value2:,.0f}" if pd.notna(value2) else "N/A",
                                            delta=delta_display,
                                            help=f"{label} for {story_person2_clean}"
                                        )
                        else:
                            # Single person view
                            extra_metrics = []
                            # Calculate Average Total Reach per Article
                            total_circ_for_avg = person_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            num_articles = len(person_articles)
                            if num_articles > 0 and pd.notna(total_circ_for_avg):
                                avg_reach_per_article = total_circ_for_avg / num_articles
                                extra_metrics.append(("Average Total Reach per Article", avg_reach_per_article))
                            
                            if 'hit_strength' in person_articles.columns:
                                extra_metrics.append(("Average Hit Strength", person_articles['hit_strength'].mean()))
                            if 'vipr_score' in person_articles.columns:
                                extra_metrics.append(("Average VIPR Score", person_articles['vipr_score'].mean()))
                            if 'vipr_weight' in person_articles.columns:
                                extra_metrics.append(("Average VIPR Weight", person_articles['vipr_weight'].mean()))
                            if extra_metrics:
                                metric_cols = st.columns(len(extra_metrics))
                                for col_obj, (label, value) in zip(metric_cols, extra_metrics):
                                    with col_obj:
                                        display_val = f"{value:,.0f}" if pd.notna(value) else "N/A"
                                        st.metric(label, display_val)

                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            with st.container(border=True):
                                if is_comparison:
                                    st.markdown("#### Emotion Distribution")
                                    st.caption(f"Comparison of emotions: {story_person} vs {story_person2_clean}")
                                else:
                                    st.markdown("#### Emotion Distribution")
                                    st.caption("Shows the distribution of emotions detected in articles mentioning this person")
                                fig_emotions = create_person_emotion_chart(
                                    person_articles, 
                                    story_person,
                                    person2_articles if is_comparison else None,
                                    story_person2_clean if is_comparison else None
                                )
                                if fig_emotions:
                                    st.plotly_chart(fig_emotions, width="stretch")
                                else:
                                    st.info("No emotion data available.")

                        with viz_col2:
                            with st.container(border=True):
                                if is_comparison:
                                    st.markdown("#### Sentiment Distribution")
                                    st.caption(f"Comparison of sentiment: {story_person} vs {story_person2_clean}")
                                else:
                                    st.markdown("#### Sentiment Distribution")
                                    st.caption("Shows how sentiment is distributed across articles (Negative, Neutral, Positive)")
                                if 'sentiment_band' not in person_articles.columns:
                                    person_articles = with_sentiment_band(person_articles)
                                if is_comparison and 'sentiment_band' not in person2_articles.columns:
                                    person2_articles = with_sentiment_band(person2_articles)
                                fig_sentiment_person = create_person_sentiment_chart(
                                    person_articles, 
                                    story_person,
                                    person2_articles if is_comparison else None,
                                    story_person2_clean if is_comparison else None
                                )
                                if fig_sentiment_person:
                                    st.plotly_chart(fig_sentiment_person, width="stretch")
                                else:
                                    st.info("No sentiment distribution available.")

                        with st.container(border=True):
                            st.markdown("### Mentions Over Time")
                            if is_comparison:
                                st.caption(f"Timeline comparison: {story_person} vs {story_person2_clean}. Helps identify trends and peak coverage periods for both individuals.")
                            else:
                                st.caption("Timeline showing when articles mentioning this person were published. Helps identify trends and peak coverage periods.")
                            
                            # Use the global date range filter
                            fig_mentions = create_person_mentions_over_time_chart(
                                person_articles,
                                story_person,
                                person2_articles if is_comparison else None,
                                story_person2_clean if is_comparison else None,
                                date_start=selected_date_start_ts,
                                date_end=selected_date_end_ts
                            )
                            if fig_mentions:
                                st.plotly_chart(fig_mentions, width="stretch")
                            else:
                                st.info("No published dates available to plot mentions over time.")

                        with st.container(border=True):
                            st.markdown("### Network Analysis")
                            st.caption("Visualize connections between this person and related entities (publications, authors, sources, channels, tags, sentiment). Nodes represent entities, edges show co-occurrence in articles. Larger nodes indicate more articles.")
                            
                            # Toggle between interactive and static views
                            use_interactive = st.checkbox(
                                "Use Interactive Network (Draggable Nodes)",
                                value=False,
                                key="person_network_interactive",
                                help="Enable interactive mode to drag and rearrange nodes. Requires streamlit-agraph package. In interactive mode, you can click and drag nodes to explore connections."
                            )
                            
                            if use_interactive and AGraph_AVAILABLE:
                                try:
                                    from streamlit_agraph import agraph
                                    network_data = build_person_network_graph_interactive(person_articles, story_person)
                                    if network_data:
                                        nodes, edges, config = network_data
                                        agraph(nodes=nodes, edges=edges, config=config)
                                        
                                        # Create legend for node types
                                        st.markdown("**Legend:**")
                                        legend_cols = st.columns(7)
                                        legend_items = [
                                            ("Person", "#12715D"),
                                            ("Publication", "#4AB48E"),
                                            ("Source Name/Type", "#2A9D8F"),
                                            ("Channel", "#D4A115"),
                                            ("Author", "#D94841"),
                                            ("Tag", "#9467BD"),
                                            ("Sentiment", "#8C564B"),
                                        ]
                                        for col, (label, color) in zip(legend_cols, legend_items):
                                            with col:
                                                st.markdown(f'<span style="color: {color};">●</span> {label}', unsafe_allow_html=True)
                                        
                                        st.caption("💡 **Tip:** Click and drag nodes to rearrange them. Hover over nodes for details.")
                                    else:
                                        st.info("Network graph not available for this selection.")
                                except Exception as e:
                                    st.warning(f"Interactive network unavailable: {e}. Falling back to static view.")
                                    network_fig = build_person_network_graph(person_articles, story_person)
                                    if network_fig:
                                        st.plotly_chart(network_fig, width="stretch")
                                    else:
                                        st.info("Network graph not available for this selection.")
                            else:
                                if not AGraph_AVAILABLE:
                                    st.info("💡 Install streamlit-agraph for interactive networks: `pip install streamlit-agraph`")
                                network_fig = build_person_network_graph(person_articles, story_person)
                                if network_fig:
                                    st.plotly_chart(
                                        network_fig,
                                        width="stretch",
                                        config={
                                            "modeBarButtonsToRemove": [
                                                "select2d", "lasso2d"
                                            ],
                                            "displayModeBar": True,
                                            "displaylogo": False,
                                            "doubleClick": "reset",
                                            "toImageButtonOptions": {
                                                "format": "png",
                                                "filename": "network_graph",
                                                "height": 600,
                                                "width": 1200,
                                                "scale": 1
                                            }
                                        }
                                    )
                                else:
                                    st.info("Network graph not available for this selection.")

                        st.markdown("---")
                        st.markdown("### Articles Mentioning This Person")
                        st.caption("Detailed table of all articles mentioning this person. Sortable columns show headline, publication, sentiment, emotion, circulation, source information, and topics.")

                        display_cols = [
                            'headline', 'publication_name',
                            'sentiment_score', 'emotion_body', 'circulation_size',
                            'source_type', 'source_name', 'channel_name',
                            'tag_name',
                        ]
                        display_cols = [col for col in display_cols if col in person_articles.columns]

                        if 'circulation_size' in person_articles.columns:
                            person_articles = person_articles.sort_values('circulation_size', ascending=False)

                        display_df = person_articles[display_cols].copy()
                        if 'sentiment_score' in display_df.columns:
                            display_df['sentiment_score'] = display_df['sentiment_score'].apply(
                                lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                            )
                        if 'circulation_size' in display_df.columns:
                            display_df['circulation_size'] = display_df['circulation_size'].apply(
                                lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                            )
                        if 'headline' in display_df.columns:
                            display_df['headline'] = display_df['headline'].apply(
                                lambda x: (str(x)[:80] + "...") if pd.notna(x) and len(str(x)) > 80 else x
                            )

                        safe_table(display_df, max_rows=2000, height=420)

                        csv_export = person_articles.to_csv(index=False)
                        export_name = f"{story_person.replace(' ', '_')}"
                        if story_keyword and story_keyword.strip():
                            export_name += f"_{story_keyword.strip().replace(' ', '_')}"
                        export_name += "_articles.csv"
                        st.download_button(
                            label="📥 Export Articles Table to CSV",
                            data=csv_export,
                            file_name=export_name,
                            mime="text/csv",
                            key="export_people_data",
                            help="Download the articles table as a CSV file for further analysis in Excel or other tools"
                        )
                        st.caption(f"Exporting {len(person_articles):,} articles with current filters.")

    # ------------------------------- Topics Tab ------------------------------- #
    with topic_tab:
        st.markdown("### Explore by Topic")

        if (
            final_df_filtered is None or final_df_filtered.empty
            or persons_by_row_filtered is None or persons_by_row_filtered.empty
        ):
            st.info("Final dataset and persons_by_row are required for topic search.")
        else:
            tag_counts = final_df_filtered['tag_name'].dropna().astype(str).value_counts()
            tag_options = tag_counts.head(500).index.tolist()
            topic_col, keyword_col = st.columns([3, 1])
            with topic_col:
                selected_tag = st.selectbox(
                    "Select Topic (tag_name)",
                    options=[""] + tag_options,
                    index=0,
                    key="topic_select",
                    help="Select a topic to analyze. Topics are sorted by frequency (most mentioned topics first). Shows analysis of people, sentiment, emotions, and network connections related to this topic."
                )
            with keyword_col:
                topic_keyword = st.text_input(
                    "Keyword Filter",
                    placeholder="e.g., vaccine, policy...",
                    key="topic_keyword",
                    help="Further filter articles within the selected topic by searching for specific keywords in headlines or article text"
                )

            if selected_tag:
                final_df_topic = final_df_filtered[final_df_filtered['tag_name'].astype(str) == selected_tag].copy()
                pbr_topic = persons_by_row_filtered[
                    persons_by_row_filtered['tag_name'].astype(str) == selected_tag
                ].copy()

                if 'row_index' not in final_df_topic.columns:
                    final_df_topic = final_df_topic.reset_index().rename(columns={'index': 'row_index'})

                if topic_keyword and topic_keyword.strip():
                    keyword = topic_keyword.strip().lower()
                    keyword_mask = pd.Series(False, index=final_df_topic.index)
                    for col in ['headline', 'article_body']:
                        keyword_mask = keyword_mask | final_df_topic[col].astype(str).str.lower().str.contains(
                            keyword, regex=False
                        )
                    final_df_topic = final_df_topic[keyword_mask].copy()

                    matching_rows = final_df_topic['row_index'].unique()
                    pbr_topic = pbr_topic[pbr_topic['row_index'].isin(matching_rows)].copy()

                if final_df_topic.empty:
                    message = f"No articles found for topic '{selected_tag}'"
                    if topic_keyword and topic_keyword.strip():
                        message += f" with keyword '{topic_keyword}'"
                    st.info(message + ".")
                else:
                    pbr_long_topic = explode_persons(pbr_topic)
                    num_people = st.slider(
                        "Number of top people to show",
                        min_value=5,
                        max_value=50,
                        value=20,
                        step=5,
                        key="top_people_slider",
                        help="Adjust how many top people (by mention count) are displayed in the charts below. Higher numbers show more individuals but may reduce chart clarity."
                    )

                    if not pbr_long_topic.empty:
                        top_n_people = pbr_long_topic['person'].value_counts().head(num_people).index.tolist()
                    else:
                        top_n_people = []

                    sentiment_scores = final_df_topic['sentiment_score'].dropna()
                    has_sentiment = len(sentiment_scores) > 0
                    if has_sentiment and 'sentiment_band' not in final_df_topic.columns:
                        # Use appropriate thresholds for sentiment_score scale (-100 to 100)
                        # Negative: < -10, Neutral: -10 to 10, Positive: > 10
                        final_df_topic['sentiment_band'] = pd.cut(
                            final_df_topic['sentiment_score'],
                            bins=[-float('inf'), -10, 10, float('inf')],
                            labels=['negative', 'neutral', 'positive']
                        )

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        with st.container(border=True):
                            st.markdown(f"#### Top {num_people} People by Sentiment")
                            st.caption("Sentiment analysis for top people mentioned in this topic. Shows how sentiment varies across different individuals.")

                            if has_sentiment and not pbr_long_topic.empty and len(top_n_people) > 0:
                                fig_sent = create_topic_sentiment_by_people_chart(
                                    pbr_long_topic, final_df_topic, top_n_people, num_people
                                )
                                if fig_sent:
                                    st.plotly_chart(fig_sent, width="stretch")
                                else:
                                    st.info("No sentiment data available for the top people in this topic.")
                            else:
                                if not has_sentiment:
                                    st.info("No sentiment_score column available for this topic.")
                                elif pbr_long_topic.empty:
                                    st.info("No people detected for this topic.")
                                elif len(top_n_people) == 0:
                                    st.info("No people found for this topic.")
                                else:
                                    st.info("No sentiment data available for this topic.")

                    with chart_col2:
                        with st.container(border=True):
                            st.markdown(f"#### Top {num_people} People by Emotion")
                            st.caption("Emotion distribution for top people in this topic. Shows which emotions are most associated with each person.")

                            if not pbr_long_topic.empty and len(top_n_people) > 0:
                                fig_emotion = create_topic_emotion_by_people_chart(
                                    pbr_long_topic, final_df_topic, top_n_people, num_people
                                )
                                if fig_emotion:
                                    st.plotly_chart(fig_emotion, width="stretch")
                                else:
                                    st.info("No emotion data available.")
                            else:
                                st.info("No people detected for this topic.")

                    with st.container(border=True):
                        st.markdown(f"#### Mentions Over Time - Top {num_people} People")
                        st.caption("Timeline showing when top people were mentioned in articles about this topic. Helps identify trends and peak coverage periods for each person.")

                        if not pbr_long_topic.empty and len(top_n_people) > 0:
                            fig_time = create_topic_mentions_over_time_chart(
                                pbr_long_topic, final_df_topic, top_n_people
                            )
                            if fig_time:
                                st.plotly_chart(fig_time, width="stretch")
                            else:
                                st.info("No data available for mentions over time.")
                        else:
                            st.info("No people detected for this topic.")

                    with st.container(border=True):
                        st.markdown("#### Circulation Quartile Distribution")
                        st.caption("Shows the distribution of article circulation sizes (readership/reach) for this topic. Quartiles help understand the reach profile of coverage.")
                        fig_circ = create_circulation_quartile_chart(final_df_topic)
                        if fig_circ:
                            st.plotly_chart(fig_circ, width="stretch")
                        else:
                            st.info("No circulation data available.")

                    with st.container(border=True):
                        st.markdown("#### Network Analysis")
                        st.caption("Visualize connections between this topic and related entities (people, publications, authors, sources, channels, tags, sentiment). Nodes represent entities, edges show co-occurrence in articles. Green/purple edges connect to the topic center, grey edges show connections between other entities.")
                        
                        # Toggle between interactive and static views
                        use_interactive = st.checkbox(
                            "Use Interactive Network (Draggable Nodes)",
                            value=False,
                            key="topic_network_interactive",
                            help="Enable interactive mode to drag and rearrange nodes. Requires streamlit-agraph package. In interactive mode, you can click and drag nodes to explore connections and see labels inside nodes."
                        )
                        
                        if use_interactive and AGraph_AVAILABLE:
                            try:
                                from streamlit_agraph import agraph
                                network_data = build_topic_categorical_network_graph_interactive(
                                    selected_tag,
                                    final_df_topic,
                                    pbr_long_topic
                                )
                                if network_data:
                                    nodes, edges, config = network_data
                                    agraph(nodes=nodes, edges=edges, config=config)
                                    
                                    # Create legend for node types
                                    st.markdown("**Legend:**")
                                    legend_cols = st.columns(8)
                                    legend_items = [
                                        ("Topic", "#9467BD"),
                                        ("Person", "#12715D"),
                                        ("Publication", "#4AB48E"),
                                        ("Source Name/Type", "#2A9D8F"),
                                        ("Channel", "#D4A115"),
                                        ("Author", "#D94841"),
                                        ("Tag", "#9467BD"),
                                        ("Sentiment", "#8C564B"),
                                    ]
                                    for col, (label, color) in zip(legend_cols, legend_items):
                                        with col:
                                            st.markdown(f'<span style="color: {color}; font-size: 1.2em;">●</span> {label}', unsafe_allow_html=True)
                                    
                                    st.caption(
                                        f"💡 **Tip:** Click and drag nodes to rearrange them. Hover over nodes for details. "
                                        f"Network showing connections between topic '{selected_tag}' and all categorical attributes. "
                                        "Categorical columns with more than 20 options are filtered to top 20. "
                                        "Node size represents number of articles."
                                    )
                                else:
                                    st.info("Categorical network graph not available for this topic.")
                            except Exception as e:
                                st.warning(f"Interactive network unavailable: {e}. Falling back to static view.")
                                categorical_network_fig = build_topic_categorical_network_graph(
                                    selected_tag,
                                    final_df_topic,
                                    pbr_long_topic
                                )
                                if categorical_network_fig:
                                    st.plotly_chart(categorical_network_fig, width="stretch")
                                else:
                                    st.info("Categorical network graph not available for this topic.")
                        else:
                            if not AGraph_AVAILABLE:
                                st.info("💡 Install streamlit-agraph for interactive networks: `pip install streamlit-agraph`")
                            categorical_network_fig = build_topic_categorical_network_graph(
                                selected_tag,
                                final_df_topic,
                                pbr_long_topic
                            )
                            if categorical_network_fig:
                                st.plotly_chart(
                                    categorical_network_fig,
                                    width="stretch",
                                    config={
                                        "modeBarButtonsToRemove": [
                                            "select2d", "lasso2d"
                                        ],
                                        "displayModeBar": True,
                                        "displaylogo": False,
                                        "doubleClick": "reset",
                                        "toImageButtonOptions": {
                                            "format": "png",
                                            "filename": "network_graph",
                                            "height": 600,
                                            "width": 1200,
                                            "scale": 1
                                        }
                                    }
                                )
                                st.caption(
                                    f"Network showing connections between topic '{selected_tag}' and all categorical attributes. "
                                    "Categorical columns with more than 20 options are filtered to top 20. "
                                    "Node size represents number of articles."
                                )
                            else:
                                st.info("Categorical network graph not available for this topic.")

    render_footer()


def render_footer():
    st.markdown("---")
    st.markdown(
    """
    <div style="font-size: 0.85rem; text-align: left; margin-top: 2rem; color: #475569;">
        Built by Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammad Waqas, Mark Saba, Posy Olivetti
    </div>
    """,
    unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()