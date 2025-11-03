"""
Streamlit App for Attribution and PCA Analysis
Shows how individuals were being talked about in the dataset
"""

from __future__ import annotations

# -------------------- MUST be first Streamlit call --------------------
import streamlit as st

# Add global error handler
def handle_error(e, context=""):
    """Handle errors with detailed information"""
    st.error(f"Error {context}: {str(e)}")
    st.error(f"Error type: {type(e).__name__}")
    import traceback
    st.code(traceback.format_exc())
    st.stop()

# Load external CSS file
import os
from pathlib import Path
css_path = os.path.join(os.path.dirname(__file__), "style.css")
try:
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
except FileNotFoundError:
    # Fallback: try relative path
    try:
        with open("style.css") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. App will run without custom styling.")

# -------------------- Imports --------------------
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import altair as alt
from collections import Counter
from typing import Dict, Optional, Tuple, Set
import re
from pathlib import Path

# Try to import streamlit_extras, fallback to custom styling if not available
try:
    from streamlit_extras.metric_cards import style_metric_cards
    STREAMLIT_EXTRAS_AVAILABLE = True
except ImportError:
    STREAMLIT_EXTRAS_AVAILABLE = False
    
    # Custom metric styling function as fallback
    def style_metric_cards(background_color="#fafafa", border_color="#fafafa", border_left_color="#12715D"):
        """Custom metric card styling when streamlit-extras is not available"""
        st.markdown(f"""
        <style>
        [data-testid="metric-container"] {{
            background-color: {background_color};
            border-left: 4px solid {border_left_color};
            padding: 0rem;
            border-radius: 0.5rem;
        }}
        </style>
        """, unsafe_allow_html=True)

# Import from modular files
from data_loaders import (
    load_influencer_table,
    load_attribution_dataset,
    load_final_dataset,
    load_persons_by_row
)
from data_processors import (
    clean_bin_column,
    extract_clean_names,
    is_likely_person_name
)
from charts import (
    create_sentiment_analysis_chart,
    create_visibility_chart,
    create_top_individuals_chart,
    create_cluster_distribution_chart,
    create_metrics_comparison,
    create_article_length_analysis,
    create_emotion_chart
)
from network_analysis import (
    get_network_data,
    build_content_network_edges,
    build_content_graph,
    community_map_content,
    create_interactive_network_visualization
)

# Add parent directory to path for data access
import sys
sys.path.append(str(Path(__file__).parent.parent))

# ---- Brand defaults for charts ----
# Penta brand colors (matching CSS variables)
PENTA_COLORS = ["#12715D", "#4AB48E", "#142536", "#D4A115", "#2A9D8F", "#D94841"]
PENTA_PRIMARY = "#12715D"
PENTA_ACCENT = "#4AB48E" 
PENTA_DARK = "#142536"
PENTA_GOLD = "#D4A115"

# Set Plotly Express defaults
px.defaults.color_discrete_sequence = PENTA_COLORS

def shorten(label: str, max_len: int = 28) -> str:
    """Shorten label for display."""
    s = str(label).strip()
    if len(s) <= max_len:
        return s
    # Try to break at word boundaries for better readability
    if ' ' in s:
        words = s.split()
        result = ""
        for word in words:
            if len(result + " " + word) <= max_len - 1:
                result += (" " + word) if result else word
            else:
                break
        if result:
            return result + "…"
    return s[:max_len-1] + "…"

# Custom CSS (additional inline styles if needed)
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #12715D;
        margin-bottom: 1rem;
        text-align: left;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #142536;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #fafafa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #12715D;
    }
</style>
""", unsafe_allow_html=True)


# Main app
def main():
    st.markdown('<div class="main-header">PolicyPath 🏛️</div>', unsafe_allow_html=True)
    st.markdown("""Your indispensable guide to healthcare policy influence""")

    # Load data
    with st.spinner("Loading data..."):
        influencer_df = load_influencer_table()
        # attribution_df = load_attribution_dataset()  # Not currently used in UI
        final_df = load_final_dataset()
        persons_by_row_df = load_persons_by_row()
    
    # Clean bin columns if they exist
    if influencer_df is not None:
        if 'circulation_size_bin' in influencer_df.columns:
            influencer_df = clean_bin_column(influencer_df, 'circulation_size_bin')
        if 'sentiment_score_bin' in influencer_df.columns:
            influencer_df = clean_bin_column(influencer_df, 'sentiment_score_bin')
        
        # Keep all persons (no filtering)
    
    if final_df is not None:
        if 'circulation_size_bin' in final_df.columns:
            final_df = clean_bin_column(final_df, 'circulation_size_bin')
        if 'sentiment_score_bin' in final_df.columns:
            final_df = clean_bin_column(final_df, 'sentiment_score_bin')
    
    # Emotion data is now pre-computed in the final_dataset_with_attribution.parquet as 'emotion_body' column
    
    if influencer_df is None:
        st.error("Unable to load influencer table. Please ensure the data files are in the correct location.")
        st.stop()
    
    # Start with unfiltered data - filters will be applied in Pulse tab
    filtered_df = influencer_df.copy()
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["PolicyPath", "People"])
    
    with tab1:
        st.markdown('<div class="section-header">PolicyPath Pulse</div>', unsafe_allow_html=True)
        st.markdown("Monitor the pulse of healthcare policy influence with interactive KPIs and charts.")
        
        # Load attribution data if available
        attribution_df = load_attribution_dataset()
        
        # Filters Section
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

        # Person search - normalize names for display (combine Trump variations, etc.)
        def normalize_trump(name):
            """Normalize Trump variations to 'Donald Trump', excluding family members"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Exclude family members (check for exact matches or names containing these)
            family_members = ['lady trump', 'eric trump', 'melania trump', 'meliana trump', 'barron trump', 'ivanka trump', 'tiffany trump']
            for family in family_members:
                if family in name_lower:
                    return name  # Keep family members as-is
            
            # Check if it's a Trump variation - ANY name containing "trump" (except family members)
            if 'trump' in name_lower:
                # If it already has "donald" in it, normalize to "Donald Trump"
                if 'donald' in name_lower:
                    return "Donald Trump"
                # If it's just "trump" or contains "trump" without "donald", normalize to "Donald Trump"
                else:
                    return "Donald Trump"
            return name
        
        def normalize_elon_musk(name):
            """Normalize Elon Musk variations to 'Elon Musk'"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Check if it's an Elon Musk variation
            if 'elon' in name_lower and 'musk' in name_lower:
                return "Elon Musk"
            elif 'musk' in name_lower and 'elon' not in name_lower:
                # Check if it's part of "Elon Musk" (like "Musk's", "Musker", "Musk-Led")
                if name_lower.startswith('musk') or 'musk' in name_lower:
                    return "Elon Musk"
            return name
        
        def normalize_anthony_fauci(name):
            """Normalize Anthony Fauci variations to 'Anthony Fauci'"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Remove common prefixes
            name_clean = re.sub(r'^(dr\.?|doctor)\s*', '', name_lower)
            # Remove trailing punctuation
            name_clean = name_clean.rstrip('.,;:')
            # Check if it's an Anthony Fauci variation
            if 'anthony' in name_clean and 'fauci' in name_clean:
                return "Anthony Fauci"
            elif 'fauci' in name_clean and 'anthony' not in name_clean:
                # Check for variations like "Faucci" (typo)
                if 'fauci' in name_clean or 'faucci' in name_clean:
                    return "Anthony Fauci"
            return name
        
        def normalize_kamala_harris(name):
            """Normalize Harris variations to 'Kamala Harris'"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Check if it's a Kamala Harris variation
            if 'kamala' in name_lower and 'harris' in name_lower:
                return "Kamala Harris"
            elif 'harris' in name_lower and 'kamala' not in name_lower:
                # Check if it's just "Harris" or variations - normalize to "Kamala Harris"
                # But exclude if it's clearly someone else (like "Harrison", "Harris Eyre", etc.)
                # Only normalize if it's just "Harris" or short variations
                if name_lower == 'harris' or name_lower.startswith('harris '):
                    # Check if it's followed by a first name (like "Harris Eyre" = different person)
                    words = name_lower.split()
                    if len(words) == 1:
                        # Just "Harris" - normalize to "Kamala Harris"
                        return "Kamala Harris"
                    elif len(words) == 2 and words[0] == 'harris':
                        # "Harris Something" - check if it's a known different person
                        # Common patterns that are NOT Kamala Harris
                        different_persons = ['harris eyre', 'harris goes', 'harris poll', 'harris time', 'harris walt', 'harris darby']
                        if name_lower not in different_persons:
                            # Could be Kamala Harris, normalize it
                            return "Kamala Harris"
            return name
        
        def normalize_robert_kennedy(name_str):
            """Normalize Robert Kennedy variations to single name"""
            if pd.isna(name_str):
                return name_str
            name_lower = str(name_str).lower().strip()
            # Check for Robert Kennedy variations
            if 'robert' in name_lower and ('kennedy' in name_lower or 'junior' in name_lower):
                return "Robert Kennedy"
            return name_str
        
        # Get all unique person names and normalize them
        raw_persons = influencer_df['person_list'].dropna().astype(str).unique().tolist()
        # Filter out non-person names using is_likely_person_name
        # Normalize all names and filter
        normalized_persons = []
        for person in raw_persons:
            # Only include if it looks like a person name
            if is_likely_person_name(person):
                normalized = normalize_robert_kennedy(person)
                normalized = normalize_trump(normalized)
                normalized = normalize_elon_musk(normalized)
                normalized = normalize_anthony_fauci(normalized)
                normalized = normalize_kamala_harris(normalized)
                normalized_persons.append(normalized)
        
        # Get unique normalized names (after combining variations)
        all_persons = sorted(list(set(normalized_persons)))
        variant_to_canonical = {}
        
        # Cluster filter
        if 'cluster_label' in influencer_df.columns:
            clusters = (
                influencer_df['cluster_label']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            clusters = sorted(clusters)
            cluster_col = 'cluster_label'
        elif 'cluster' in influencer_df.columns:
            clusters = (
                influencer_df['cluster']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            clusters = sorted(clusters)
            cluster_col = 'cluster'
        else:
            clusters = []
            cluster_col = None
            
        # Filter inputs in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            selected_persons = st.multiselect(
                "Select Individuals",
                all_persons,
                default=[]
            )
        
        with filter_col2:
            if clusters:
                selected_clusters = st.multiselect(
                    "Select Clusters",
                    clusters,
                    default=clusters
                )
            else:
                selected_clusters = []
        
        with filter_col3:
            # Sentiment Band filter
            if 'sentiment_score' in influencer_df.columns:
                # Create sentiment bands if not already present
                if 'sentiment_band' not in influencer_df.columns:
                    pulse_filtered_df_temp = influencer_df.copy()
                    pulse_filtered_df_temp['sentiment_band'] = pd.cut(
                        pulse_filtered_df_temp['sentiment_score'],
                        bins=[-float('inf'), -0.1, 0.1, float('inf')],
                        labels=['Negative', 'Neutral', 'Positive']
                    )
                    sentiment_bands = sorted(pulse_filtered_df_temp['sentiment_band'].dropna().unique().tolist())
                else:
                    sentiment_bands = sorted(influencer_df['sentiment_band'].dropna().unique().tolist())
                # Always show multiselect when sentiment_score exists
                selected_sentiment_bands = st.multiselect(
                    "Sentiment Band",
                    sentiment_bands,
                    default=sentiment_bands,
                    key="filter_sentiment_band"
                )
            else:
                selected_sentiment_bands = []
        
        # Additional filters row - Date, Publications, Source Types
        filter_row2_col1, filter_row2_col2, filter_row2_col3 = st.columns(3)
        
        # Get available columns from final_df for filtering
        date_range = None
        selected_publications = []
        selected_channels = []
        selected_authors = []
        selected_source_types = []
        
        if final_df is not None and not final_df.empty:
            final_df_cols = final_df.columns.tolist()
            
            with filter_row2_col1:
                # Date range filter
                date_col = next((c for c in final_df_cols if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)
                if date_col:
                    try:
                        final_df_copy_date = final_df.copy()
                        final_df_copy_date[date_col] = pd.to_datetime(final_df_copy_date[date_col], errors="coerce")
                        min_d = final_df_copy_date[date_col].min()
                        max_d = final_df_copy_date[date_col].max()
                        if pd.notna(min_d) and pd.notna(max_d):
                            date_range = st.date_input(
                                "Date Range",
                                value=(min_d.date(), max_d.date()),
                                min_value=min_d.date(),
                                max_value=max_d.date(),
                                key="filter_date_range"
                            )
                    except Exception:
                        date_range = None
            
            with filter_row2_col2:
                # Publications filter
                pub_cols = [c for c in final_df_cols if "publication" in c.lower()]
                if pub_cols:
                    pub_col = pub_cols[0]
                    pubs = sorted(final_df[pub_col].dropna().unique().tolist())[:50]
                    selected_publications = st.multiselect(
                        "Publications",
                        pubs,
                        default=[],
                        key="filter_publications"
                    )
            
            with filter_row2_col3:
                # Channels/Source Types filter
                source_cols = [c for c in final_df_cols if "source" in c.lower() and "type" in c.lower()]
                if source_cols:
                    source_col = source_cols[0]
                    sources = sorted(final_df[source_col].dropna().unique().tolist())[:20]
                    selected_source_types = st.multiselect(
                        "Source Types",
                        sources,
                        default=[],
                        key="filter_source_types"
                    )
        
        # Additional filters row - Authors, Sentiment Band (if available in final_df)
        filter_row3_col1, filter_row3_col2 = st.columns(2)
        
        if final_df is not None and not final_df.empty:
            with filter_row3_col1:
                # Authors filter
                author_cols = [c for c in final_df_cols if "author" in c.lower()]
                if author_cols:
                    author_col = author_cols[0]
                    authors = sorted(final_df[author_col].dropna().unique().tolist())[:50]
                    selected_authors = st.multiselect(
                        "Authors",
                        authors,
                        default=[],
                        key="filter_authors"
                    )
        
        # Apply filters
        # First, filter article-level data if filters are applied
        filtered_article_indices = None
        if final_df is not None and not final_df.empty and persons_by_row_df is not None:
            final_df_filtered = final_df.copy()
            if 'row_index' not in final_df_filtered.columns:
                final_df_filtered = final_df_filtered.reset_index().rename(columns={'index': 'row_index'})
            
            # Apply article-level filters
            mask = pd.Series(True, index=final_df_filtered.index)
            
            # Date range filter
            if date_range and len(date_range) == 2:
                date_col = next((c for c in final_df_filtered.columns if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)
                if date_col:
                    try:
                        final_df_filtered[date_col] = pd.to_datetime(final_df_filtered[date_col], errors="coerce")
                        mask &= (final_df_filtered[date_col].dt.date >= date_range[0]) & (final_df_filtered[date_col].dt.date <= date_range[1])
                    except Exception:
                        pass
            
            # Publication filter
            if selected_publications:
                pub_cols = [c for c in final_df_filtered.columns if "publication" in c.lower()]
                if pub_cols:
                    mask &= final_df_filtered[pub_cols[0]].isin(selected_publications)
            
            # Source type filter
            if selected_source_types:
                source_cols = [c for c in final_df_filtered.columns if "source" in c.lower() and "type" in c.lower()]
                if source_cols:
                    mask &= final_df_filtered[source_cols[0]].isin(selected_source_types)
            
            # Author filter
            if selected_authors:
                author_cols = [c for c in final_df_filtered.columns if "author" in c.lower()]
                if author_cols:
                    mask &= final_df_filtered[author_cols[0]].isin(selected_authors)
            
            # Get filtered article indices
            filtered_article_indices = set(final_df_filtered[mask]['row_index'].values)
        
        # Now filter influencer_df based on person selections and article-level filters
        # Use vectorized operations where possible for speed
        filter_status = st.empty()
        filter_status.info("🔄 Applying filters...")
        
        # Start with a view - only copy when we need to modify
        pulse_filtered_df = influencer_df
        
        # If we have article-level filters, we need to find which persons are mentioned in those articles
        if filtered_article_indices is not None and persons_by_row_df is not None:
            # Find persons mentioned in filtered articles
            filtered_person_rows = persons_by_row_df[persons_by_row_df['row_index'].isin(filtered_article_indices)]
            if not filtered_person_rows.empty:
                # Extract all persons from filtered articles (vectorized)
                filtered_persons_set = set()
                persons_series = filtered_person_rows['persons'].dropna().astype(str)
                # Use vectorized split for better performance
                all_persons_split = persons_series.str.split(',')
                for persons_list in all_persons_split:
                    if isinstance(persons_list, list):
                        filtered_persons_set.update([p.strip() for p in persons_list if p.strip()])
                
                # Filter influencer_df using vectorized string operations (faster than apply)
                if filtered_persons_set:
                    # Create a regex pattern for faster matching
                    person_pattern = '|'.join([re.escape(p.lower()) for p in filtered_persons_set if p])
                    if person_pattern:
                        # Only copy when we need to filter
                        if pulse_filtered_df is influencer_df:
                            pulse_filtered_df = influencer_df.copy()
                        mask = pulse_filtered_df['person_list'].astype(str).str.lower().str.contains(
                            person_pattern, case=False, na=False, regex=True
                        )
                        pulse_filtered_df = pulse_filtered_df[mask]
        
        if selected_persons:
            # Normalize name variations (same as above)
            def normalize_trump(name):
                """Normalize Trump variations to 'Donald Trump', excluding family members"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                family_members = ['lady trump', 'eric trump', 'melania trump', 'meliana trump', 'barron trump', 'ivanka trump', 'tiffany trump']
                for family in family_members:
                    if family in name_lower:
                        return name
                if 'trump' in name_lower:
                    return "Donald Trump"
                return name
            
            def normalize_elon_musk(name):
                """Normalize Elon Musk variations to 'Elon Musk'"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                if 'elon' in name_lower and 'musk' in name_lower:
                    return "Elon Musk"
                elif 'musk' in name_lower and 'elon' not in name_lower:
                    if name_lower.startswith('musk') or 'musk' in name_lower:
                        return "Elon Musk"
                return name
            
            def normalize_anthony_fauci(name):
                """Normalize Anthony Fauci variations to 'Anthony Fauci'"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                name_clean = re.sub(r'^(dr\.?|doctor)\s*', '', name_lower)
                name_clean = name_clean.rstrip('.,;:')
                if 'anthony' in name_clean and 'fauci' in name_clean:
                    return "Anthony Fauci"
                elif 'fauci' in name_clean or 'faucci' in name_clean:
                    return "Anthony Fauci"
                return name
            
            def normalize_kamala_harris(name):
                """Normalize Harris variations to 'Kamala Harris'"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                if 'kamala' in name_lower and 'harris' in name_lower:
                    return "Kamala Harris"
                elif 'harris' in name_lower and 'kamala' not in name_lower:
                    words = name_lower.split()
                    if len(words) == 1:
                        return "Kamala Harris"
                    elif len(words) == 2 and words[0] == 'harris':
                        different_persons = ['harris eyre', 'harris goes', 'harris poll', 'harris time', 'harris walt', 'harris darby']
                        if name_lower not in different_persons:
                            return "Kamala Harris"
                return name
            
            # Normalize selected persons
            normalized_selected = []
            for sp in selected_persons:
                normalized = normalize_trump(sp)
                normalized = normalize_elon_musk(normalized)
                normalized = normalize_anthony_fauci(normalized)
                normalized = normalize_kamala_harris(normalized)
                normalized_selected.append(normalized)
            
            # Use vectorized string operations for LIKE/contains matching (much faster than apply)
            # Create a pattern from normalized selected persons
            selected_pattern = '|'.join([re.escape(sp.lower()) for sp in normalized_selected if sp])
            if selected_pattern:
                # Only copy when we need to filter
                if pulse_filtered_df is influencer_df:
                    pulse_filtered_df = influencer_df.copy()
                
                # Also normalize the person_list column for matching
                pulse_filtered_df = pulse_filtered_df.copy()
                
                # Apply all normalizations
                pulse_filtered_df['person_list_normalized'] = pulse_filtered_df['person_list'].apply(normalize_trump)
                pulse_filtered_df['person_list_normalized'] = pulse_filtered_df['person_list_normalized'].apply(normalize_elon_musk)
                pulse_filtered_df['person_list_normalized'] = pulse_filtered_df['person_list_normalized'].apply(normalize_anthony_fauci)
                pulse_filtered_df['person_list_normalized'] = pulse_filtered_df['person_list_normalized'].apply(normalize_kamala_harris)
                
                # Use vectorized string contains (much faster)
                mask = pulse_filtered_df['person_list_normalized'].astype(str).str.lower().str.contains(
                    selected_pattern, case=False, na=False, regex=True
                )
                pulse_filtered_df = pulse_filtered_df[mask]
        
        if selected_clusters and cluster_col:
            # Only copy when we need to filter
            if pulse_filtered_df is influencer_df:
                pulse_filtered_df = influencer_df.copy()
            # Convert cluster column to string for comparison
            pulse_filtered_df = pulse_filtered_df[
                pulse_filtered_df[cluster_col].astype(str).isin(selected_clusters)
            ]
        
        # Sentiment band filter
        if selected_sentiment_bands:
            # Only copy when we need to filter
            if pulse_filtered_df is influencer_df:
                pulse_filtered_df = influencer_df.copy()
            # Create sentiment bands if not already present
            if 'sentiment_band' not in pulse_filtered_df.columns and 'sentiment_score' in pulse_filtered_df.columns:
                pulse_filtered_df['sentiment_band'] = pd.cut(
                    pulse_filtered_df['sentiment_score'],
                    bins=[-float('inf'), -0.1, 0.1, float('inf')],
                    labels=['Negative', 'Neutral', 'Positive']
                )
            if 'sentiment_band' in pulse_filtered_df.columns:
                pulse_filtered_df = pulse_filtered_df[
                    pulse_filtered_df['sentiment_band'].isin(selected_sentiment_bands)
                ]
        
        # Clear the filter status
        filter_status.empty()
        
        # Summary Statistics (same format as before)
        st.markdown('<div class="section-header">Data Summary</div>', unsafe_allow_html=True)
        
        # Show metrics immediately (fast computation)
        if not pulse_filtered_df.empty:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_individuals = len(pulse_filtered_df)
                st.metric("Total Individuals", f"{total_individuals:,}")
            
            with col2:
                total_mentions = pulse_filtered_df['mention_count'].sum() if 'mention_count' in pulse_filtered_df.columns else 0
                st.metric("Total Mentions", f"{total_mentions:,}")
            
            with col3:
                avg_sentiment = pulse_filtered_df['sentiment_score'].mean() if 'sentiment_score' in pulse_filtered_df.columns else 0
                st.metric("Avg Sentiment Score", f"{avg_sentiment:.2f}")
            
            with col4:
                avg_circulation = pulse_filtered_df['circulation_size'].mean() if 'circulation_size' in pulse_filtered_df.columns else 0
                st.metric("Avg Circulation", f"{avg_circulation:,.0f}")
    
        # Policy Influence Analytics
        # Show content immediately with a placeholder if filtering takes time
        if not pulse_filtered_df.empty:
            # Determine grouping dimension
            group_by_options = []
            if cluster_col in pulse_filtered_df.columns:
                group_by_options.append(('Cluster', cluster_col))
            if 'sentiment_score' in pulse_filtered_df.columns:
                # Create sentiment bins
                pulse_filtered_df = pulse_filtered_df.copy()
                pulse_filtered_df['sentiment_band'] = pd.cut(
                    pulse_filtered_df['sentiment_score'],
                    bins=[-float('inf'), -0.1, 0.1, float('inf')],
                    labels=['Negative', 'Neutral', 'Positive']
                )
                group_by_options.append(('Sentiment Band', 'sentiment_band'))
            
            if group_by_options:
                    dim_label, dim_col = group_by_options[0]  # Use first available
                    dim = st.selectbox("Group charts by", [label for label, _ in group_by_options], index=0)
                    dim_col = next(col for label, col in group_by_options if label == dim)
                    
                    # Aggregate data
                    agg_dict = {'mention_count': 'sum'}
                    if 'sentiment_score' in pulse_filtered_df.columns:
                        agg_dict['sentiment_score'] = 'mean'
                    if 'circulation_size' in pulse_filtered_df.columns:
                        agg_dict['circulation_size'] = 'mean'
                    
                    agg = (
                        pulse_filtered_df
                        .groupby(dim_col)
                        .agg(agg_dict)
                        .reset_index()
                        .rename(columns={dim_col: "dim"})
                    )
                    agg = agg[agg["dim"].notna()]
                    
                    # Top N Slider
                    top_n = st.slider("Top N", 5, 30, 10, 1)
                    
                    # Row 1: Charts
                    col1, col2 = st.columns(2)

                    with col1:
                        # Sentiment Analysis by Cluster
                        if cluster_col in pulse_filtered_df.columns and 'sentiment_score' in pulse_filtered_df.columns:
                            sentiment_by_cluster = pulse_filtered_df.groupby(cluster_col)['sentiment_score'].mean().reset_index()
                            fig_sentiment = go.Figure(data=[
                                go.Bar(
                                    x=sentiment_by_cluster[cluster_col],
                                    y=sentiment_by_cluster['sentiment_score'],
                                    marker_color=PENTA_COLORS[0]
                                )
                            ])
                            fig_sentiment.update_layout(
                                title="Sentiment Distribution by Cluster",
                                xaxis_title="Cluster",
                                yaxis_title="Average Sentiment",
                                height=300,
                                showlegend=False
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        else:
                            st.info("No cluster/sentiment data available")
                    
                    with col2:
                        # Circulation Analysis
                        if cluster_col in pulse_filtered_df.columns and 'circulation_size' in pulse_filtered_df.columns:
                            circ_by_cluster = pulse_filtered_df.groupby(cluster_col)['circulation_size'].mean().nlargest(8).reset_index()
                            fig_circ = go.Figure(data=[
                                go.Bar(
                                    x=circ_by_cluster[cluster_col],
                                    y=circ_by_cluster['circulation_size'],
                                    marker_color=PENTA_COLORS[1]
                                )
                            ])
                            fig_circ.update_layout(
                                title="Top Clusters by Circulation",
                                xaxis_title="Cluster",
                                yaxis_title="Average Circulation",
                                height=300,
                                showlegend=False
                            )
                            st.plotly_chart(fig_circ, use_container_width=True)
                        else:
                            st.info("No cluster/circulation data available")
                    
                    # Emotion chart with slider
                    st.markdown('<div class="section-header">Top Individuals by Emotion</div>', unsafe_allow_html=True)
                    
                    # Slider for the emotion chart
                    n_top = st.slider("Number of top individuals to show", 10, 50, 20, key="emotion_slider")
                    
                    # Debug: Check if data is available
                    if final_df is None:
                        st.info("Final dataset not loaded. Emotion analysis requires final_dataset_with_attribution.parquet.")
                    elif persons_by_row_df is None or persons_by_row_df.empty:
                        st.info("Person mapping data not available. Emotion analysis requires persons_by_row.csv.")
                    elif 'emotion_body' not in final_df.columns:
                        st.info(f"Emotion data not available. Available columns: {', '.join(final_df.columns[:10])}...")
                    else:
                        # Show loading placeholder while creating chart
                        with st.spinner("Generating emotion analysis..."):
                            emotion_chart = create_emotion_chart(
                                pulse_filtered_df, 
                                final_df=final_df, 
                                persons_by_row_df=persons_by_row_df,
                                n=n_top,
                                selected_persons=selected_persons if selected_persons else None
                            )
                        if emotion_chart:
                            st.plotly_chart(emotion_chart, use_container_width=True)
                        else:
                            st.info("Emotion data not available. No articles found with emotion labels for the top individuals.")

    with tab2:
        st.markdown('<div class="section-header">People</div>', unsafe_allow_html=True)
        st.markdown("**Search by individual with optional keyword to explore what story is being told - see articles, topics, publishers, and sentiment.**")
        
        # Filters Section (exact replica of tab 1)
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)
        
        # Person search - normalize names for display (combine Trump variations, etc.)
        def normalize_trump(name):
            """Normalize Trump variations to 'Donald Trump', excluding family members"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Exclude family members (check for exact matches or names containing these)
            family_members = ['lady trump', 'eric trump', 'melania trump', 'meliana trump', 'barron trump', 'ivanka trump', 'tiffany trump']
            for family in family_members:
                if family in name_lower:
                    return name  # Keep family members as-is
            
            # Check if it's a Trump variation - ANY name containing "trump" (except family members)
            if 'trump' in name_lower:
                # If it already has "donald" in it, normalize to "Donald Trump"
                if 'donald' in name_lower:
                    return "Donald Trump"
                # If it's just "trump" or contains "trump" without "donald", normalize to "Donald Trump"
                else:
                    return "Donald Trump"
            return name
        
        def normalize_elon_musk(name):
            """Normalize Elon Musk variations to 'Elon Musk'"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Check if it's an Elon Musk variation
            if 'elon' in name_lower and 'musk' in name_lower:
                return "Elon Musk"
            elif 'musk' in name_lower and 'elon' not in name_lower:
                # Check if it's part of "Elon Musk" (like "Musk's", "Musker", "Musk-Led")
                if name_lower.startswith('musk') or 'musk' in name_lower:
                    return "Elon Musk"
            return name
        
        def normalize_anthony_fauci(name):
            """Normalize Anthony Fauci variations to 'Anthony Fauci'"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Remove common prefixes
            name_clean = re.sub(r'^(dr\.?|doctor)\s*', '', name_lower)
            # Remove trailing punctuation
            name_clean = name_clean.rstrip('.,;:')
            # Check if it's an Anthony Fauci variation
            if 'anthony' in name_clean and 'fauci' in name_clean:
                return "Anthony Fauci"
            elif 'fauci' in name_clean and 'anthony' not in name_clean:
                # Check for variations like "Faucci" (typo)
                if 'fauci' in name_clean or 'faucci' in name_clean:
                    return "Anthony Fauci"
            return name
        
        def normalize_kamala_harris(name):
            """Normalize Harris variations to 'Kamala Harris'"""
            if pd.isna(name):
                return name
            name_str = str(name).strip()
            if not name_str:
                return name
            name_lower = name_str.lower()
            # Check if it's a Kamala Harris variation
            if 'kamala' in name_lower and 'harris' in name_lower:
                return "Kamala Harris"
            elif 'harris' in name_lower and 'kamala' not in name_lower:
                # Check if it's just "Harris" or variations - normalize to "Kamala Harris"
                # But exclude if it's clearly someone else (like "Harrison", "Harris Eyre", etc.)
                # Only normalize if it's just "Harris" or short variations
                if name_lower == 'harris' or name_lower.startswith('harris '):
                    # Check if it's followed by a first name (like "Harris Eyre" = different person)
                    words = name_lower.split()
                    if len(words) == 1:
                        # Just "Harris" - normalize to "Kamala Harris"
                        return "Kamala Harris"
                    elif len(words) == 2 and words[0] == 'harris':
                        # "Harris Something" - check if it's a known different person
                        # Common patterns that are NOT Kamala Harris
                        different_persons = ['harris eyre', 'harris goes', 'harris poll', 'harris time', 'harris walt', 'harris darby']
                        if name_lower not in different_persons:
                            # Could be Kamala Harris, normalize it
                            return "Kamala Harris"
            return name
        
        def normalize_robert_kennedy(name_str):
            """Normalize Robert Kennedy variations to single name"""
            if pd.isna(name_str):
                return name_str
            name_lower = str(name_str).lower().strip()
            # Check for Robert Kennedy variations
            if 'robert' in name_lower and ('kennedy' in name_lower or 'junior' in name_lower):
                return "Robert Kennedy"
            return name_str
        
        # Get all unique person names and normalize them
        raw_persons = influencer_df['person_list'].dropna().astype(str).unique().tolist()
        # Filter out non-person names using is_likely_person_name
        # Normalize all names and filter
        normalized_persons = []
        for person in raw_persons:
            # Only include if it looks like a person name
            if is_likely_person_name(person):
                normalized = normalize_robert_kennedy(person)
                normalized = normalize_trump(normalized)
                normalized = normalize_elon_musk(normalized)
                normalized = normalize_anthony_fauci(normalized)
                normalized = normalize_kamala_harris(normalized)
                normalized_persons.append(normalized)
        
        # Get unique normalized names (after combining variations)
        all_persons = sorted(list(set(normalized_persons)))
        variant_to_canonical = {}
        
        # Cluster filter
        if 'cluster_label' in influencer_df.columns:
            clusters = (
                influencer_df['cluster_label']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            clusters = sorted(clusters)
            cluster_col = 'cluster_label'
        elif 'cluster' in influencer_df.columns:
            clusters = (
                influencer_df['cluster']
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            clusters = sorted(clusters)
            cluster_col = 'cluster'
        else:
            clusters = []
            cluster_col = None
            
        # Filter inputs in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            selected_persons_tab2 = st.multiselect(
                "Select Individuals",
                all_persons,
                default=[],
                key="tab2_select_individuals"
            )
        
        with filter_col2:
            if clusters:
                selected_clusters_tab2 = st.multiselect(
                    "Select Clusters",
                    clusters,
                    default=clusters,
                    key="tab2_select_clusters"
                )
            else:
                selected_clusters_tab2 = []
        
        with filter_col3:
            # Sentiment Band filter
            if 'sentiment_score' in influencer_df.columns:
                # Create sentiment bands if not already present
                if 'sentiment_band' not in influencer_df.columns:
                    pulse_filtered_df_temp = influencer_df.copy()
                    pulse_filtered_df_temp['sentiment_band'] = pd.cut(
                        pulse_filtered_df_temp['sentiment_score'],
                        bins=[-float('inf'), -0.1, 0.1, float('inf')],
                        labels=['Negative', 'Neutral', 'Positive']
                    )
                    sentiment_bands = sorted(pulse_filtered_df_temp['sentiment_band'].dropna().unique().tolist())
                else:
                    sentiment_bands = sorted(influencer_df['sentiment_band'].dropna().unique().tolist())
                selected_sentiment_bands_tab2 = st.multiselect(
                    "Sentiment Band",
                    sentiment_bands,
                    default=sentiment_bands,
                    key="tab2_filter_sentiment_band"
                )
            else:
                selected_sentiment_bands_tab2 = []
        
        # Additional filters row - Date, Publications, Source Types
        filter_row2_col1, filter_row2_col2, filter_row2_col3 = st.columns(3)
        
        # Get available columns from final_df for filtering
        date_range_tab2 = None
        selected_publications_tab2 = []
        selected_channels_tab2 = []
        selected_authors_tab2 = []
        selected_source_types_tab2 = []
        
        if final_df is not None and not final_df.empty:
            final_df_cols = final_df.columns.tolist()
            
            with filter_row2_col1:
                # Date range filter
                date_col = next((c for c in final_df_cols if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)
                if date_col:
                    try:
                        final_df_copy_date = final_df.copy()
                        final_df_copy_date[date_col] = pd.to_datetime(final_df_copy_date[date_col], errors="coerce")
                        min_d = final_df_copy_date[date_col].min()
                        max_d = final_df_copy_date[date_col].max()
                        if pd.notna(min_d) and pd.notna(max_d):
                            date_range_tab2 = st.date_input(
                                "Date Range",
                                value=(min_d.date(), max_d.date()),
                                min_value=min_d.date(),
                                max_value=max_d.date(),
                                key="tab2_filter_date_range"
                            )
                    except Exception:
                        date_range_tab2 = None
            
            with filter_row2_col2:
                # Publications filter
                pub_cols = [c for c in final_df_cols if "publication" in c.lower()]
                if pub_cols:
                    pub_col = pub_cols[0]
                    pubs = sorted(final_df[pub_col].dropna().unique().tolist())[:50]
                    selected_publications_tab2 = st.multiselect(
                        "Publications",
                        pubs,
                        default=[],
                        key="tab2_filter_publications"
                    )
            
            with filter_row2_col3:
                # Channels/Source Types filter
                source_cols = [c for c in final_df_cols if "source" in c.lower() and "type" in c.lower()]
                if source_cols:
                    source_col = source_cols[0]
                    sources = sorted(final_df[source_col].dropna().unique().tolist())[:20]
                    selected_source_types_tab2 = st.multiselect(
                        "Source Types",
                        sources,
                        default=[],
                        key="tab2_filter_source_types"
                    )
        
        # Additional filters row - Authors, Sentiment Band (if available in final_df)
        filter_row3_col1, filter_row3_col2 = st.columns(2)
        
        if final_df is not None and not final_df.empty:
            with filter_row3_col1:
                # Authors filter
                author_cols = [c for c in final_df_cols if "author" in c.lower()]
                if author_cols:
                    author_col = author_cols[0]
                    authors = sorted(final_df[author_col].dropna().unique().tolist())[:50]
                    selected_authors_tab2 = st.multiselect(
                        "Authors",
                        authors,
                        default=[],
                        key="tab2_filter_authors"
                    )
        
        # Optional keyword filter for individual search
        st.markdown("---")
        
        # Apply filters first (same logic as tab 1)
        filtered_article_indices_tab2 = None
        if final_df is not None and not final_df.empty and persons_by_row_df is not None:
            final_df_filtered_tab2 = final_df.copy()
            if 'row_index' not in final_df_filtered_tab2.columns:
                final_df_filtered_tab2 = final_df_filtered_tab2.reset_index().rename(columns={'index': 'row_index'})
            
            # Apply article-level filters
            mask_tab2 = pd.Series(True, index=final_df_filtered_tab2.index)
            
            # Date range filter
            if date_range_tab2 and len(date_range_tab2) == 2:
                date_col = next((c for c in final_df_filtered_tab2.columns if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)
                if date_col:
                    try:
                        final_df_filtered_tab2[date_col] = pd.to_datetime(final_df_filtered_tab2[date_col], errors="coerce")
                        mask_tab2 &= (final_df_filtered_tab2[date_col].dt.date >= date_range_tab2[0]) & (final_df_filtered_tab2[date_col].dt.date <= date_range_tab2[1])
                    except Exception:
                        pass
            
            # Publication filter
            if selected_publications_tab2:
                pub_cols = [c for c in final_df_filtered_tab2.columns if "publication" in c.lower()]
                if pub_cols:
                    mask_tab2 &= final_df_filtered_tab2[pub_cols[0]].isin(selected_publications_tab2)
            
            # Source type filter
            if selected_source_types_tab2:
                source_cols = [c for c in final_df_filtered_tab2.columns if "source" in c.lower() and "type" in c.lower()]
                if source_cols:
                    mask_tab2 &= final_df_filtered_tab2[source_cols[0]].isin(selected_source_types_tab2)
            
            # Author filter
            if selected_authors_tab2:
                author_cols = [c for c in final_df_filtered_tab2.columns if "author" in c.lower()]
                if author_cols:
                    mask_tab2 &= final_df_filtered_tab2[author_cols[0]].isin(selected_authors_tab2)
            
            # Get filtered article indices
            filtered_article_indices_tab2 = set(final_df_filtered_tab2[mask_tab2]['row_index'].values)
        
        # Filter influencer_df based on selections
        influencer_df_filtered_tab2 = influencer_df.copy()
        
        # Apply cluster filter
        if selected_clusters_tab2 and cluster_col and cluster_col in influencer_df_filtered_tab2.columns:
            influencer_df_filtered_tab2 = influencer_df_filtered_tab2[influencer_df_filtered_tab2[cluster_col].isin(selected_clusters_tab2)]
        
        # Apply sentiment band filter
        if selected_sentiment_bands_tab2:
            if 'sentiment_band' not in influencer_df_filtered_tab2.columns:
                influencer_df_filtered_tab2['sentiment_band'] = pd.cut(
                    influencer_df_filtered_tab2['sentiment_score'],
                    bins=[-float('inf'), -0.1, 0.1, float('inf')],
                    labels=['Negative', 'Neutral', 'Positive']
                )
            influencer_df_filtered_tab2 = influencer_df_filtered_tab2[influencer_df_filtered_tab2['sentiment_band'].isin(selected_sentiment_bands_tab2)]
        
        # Apply person filter
        if selected_persons_tab2:
            # Normalize name variations (same as above)
            def normalize_trump_filter(name):
                """Normalize Trump variations to 'Donald Trump', excluding family members"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                family_members = ['lady trump', 'eric trump', 'melania trump', 'meliana trump', 'barron trump', 'ivanka trump', 'tiffany trump']
                for family in family_members:
                    if family in name_lower:
                        return name
                if 'trump' in name_lower:
                    return "Donald Trump"
                return name
            
            def normalize_elon_musk_filter(name):
                """Normalize Elon Musk variations to 'Elon Musk'"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                if 'elon' in name_lower and 'musk' in name_lower:
                    return "Elon Musk"
                elif 'musk' in name_lower and 'elon' not in name_lower:
                    if name_lower.startswith('musk') or 'musk' in name_lower:
                        return "Elon Musk"
                return name
            
            def normalize_anthony_fauci_filter(name):
                """Normalize Anthony Fauci variations to 'Anthony Fauci'"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                name_clean = re.sub(r'^(dr\.?|doctor)\s*', '', name_lower)
                name_clean = name_clean.rstrip('.,;:')
                if 'anthony' in name_clean and 'fauci' in name_clean:
                    return "Anthony Fauci"
                elif 'fauci' in name_clean or 'faucci' in name_clean:
                    return "Anthony Fauci"
                return name
            
            def normalize_kamala_harris_filter(name):
                """Normalize Harris variations to 'Kamala Harris'"""
                if pd.isna(name):
                    return name
                name_str = str(name).strip()
                if not name_str:
                    return name
                name_lower = name_str.lower()
                if 'kamala' in name_lower and 'harris' in name_lower:
                    return "Kamala Harris"
                elif 'harris' in name_lower and 'kamala' not in name_lower:
                    words = name_lower.split()
                    if len(words) == 1:
                        return "Kamala Harris"
                    elif len(words) == 2 and words[0] == 'harris':
                        different_persons = ['harris eyre', 'harris goes', 'harris poll', 'harris time', 'harris walt', 'harris darby']
                        if name_lower not in different_persons:
                            return "Kamala Harris"
                return name
            
            # Normalize selected persons
            normalized_selected_tab2 = []
            for sp in selected_persons_tab2:
                normalized = normalize_trump_filter(sp)
                normalized = normalize_elon_musk_filter(normalized)
                normalized = normalize_anthony_fauci_filter(normalized)
                normalized = normalize_kamala_harris_filter(normalized)
                normalized_selected_tab2.append(normalized)
            
            # Create pattern for matching
            selected_pattern_tab2 = '|'.join([re.escape(sp.lower()) for sp in normalized_selected_tab2 if sp])
            if selected_pattern_tab2:
                influencer_df_filtered_tab2['person_list_normalized'] = influencer_df_filtered_tab2['person_list'].apply(normalize_trump_filter)
                influencer_df_filtered_tab2['person_list_normalized'] = influencer_df_filtered_tab2['person_list_normalized'].apply(normalize_elon_musk_filter)
                influencer_df_filtered_tab2['person_list_normalized'] = influencer_df_filtered_tab2['person_list_normalized'].apply(normalize_anthony_fauci_filter)
                influencer_df_filtered_tab2['person_list_normalized'] = influencer_df_filtered_tab2['person_list_normalized'].apply(normalize_kamala_harris_filter)
                
                mask_person_tab2 = influencer_df_filtered_tab2['person_list_normalized'].astype(str).str.lower().str.contains(
                    selected_pattern_tab2, case=False, na=False, regex=True
                )
                influencer_df_filtered_tab2 = influencer_df_filtered_tab2[mask_person_tab2]
        
        # If we have article-level filters, filter persons_by_row_df
        if filtered_article_indices_tab2 is not None:
            persons_by_row_df_filtered_tab2 = persons_by_row_df[
                persons_by_row_df['row_index'].isin(filtered_article_indices_tab2)
            ].copy() if persons_by_row_df is not None and not persons_by_row_df.empty else pd.DataFrame()
        else:
            persons_by_row_df_filtered_tab2 = persons_by_row_df.copy() if persons_by_row_df is not None else pd.DataFrame()
        
        search_col1, search_col2 = st.columns([2, 1])
        with search_col1:
            story_person = st.selectbox(
                "Select an Individual",
                options=[""] + all_persons,
                index=0,
                help="Fast, case-insensitive, partial-match people search",
                key="tab2_story_person"
            )
        with search_col2:
            story_keyword = st.text_input(
                "Optional Keyword Filter",
                placeholder="e.g., vaccine, policy...",
                help="Filter results to articles also containing this keyword (e.g., see how 'donald trump' is talked about in terms of 'vaccine')",
                key="tab2_story_keyword"
            )
        
        # Show search results
        if story_person:
            # Get row indices where this person is mentioned (from filtered data)
            person_rows = persons_by_row_df_filtered_tab2[
                persons_by_row_df_filtered_tab2['persons'].astype(str).str.contains(story_person, case=False, na=False, regex=False)
            ] if not persons_by_row_df_filtered_tab2.empty else pd.DataFrame()
            
            if not person_rows.empty and final_df is not None and not final_df.empty:
                # Merge with final_df to get article data
                # Work with a copy to avoid modifying cached data
                final_df_copy = final_df.copy()
                
                # First, ensure we have row_index in final_df
                if 'row_index' not in final_df_copy.columns:
                    final_df_copy = final_df_copy.reset_index().rename(columns={'index': 'row_index'})
                
                # Apply article-level filters if they exist
                if filtered_article_indices_tab2 is not None:
                    final_df_copy = final_df_copy[final_df_copy['row_index'].isin(filtered_article_indices_tab2)]
                
                person_articles = final_df_copy.merge(
                    person_rows[['row_index']],
                    on='row_index',
                    how='inner'
                )
                
                # Apply keyword filter if provided (before showing summary)
                if story_keyword and story_keyword.strip():
                    text_cols = ['headline', 'article_body']
                    available_text_cols = [c for c in text_cols if c in person_articles.columns]
                    if available_text_cols:
                        mask = pd.Series(False, index=person_articles.index)
                        for col in available_text_cols:
                            mask |= person_articles[col].astype(str).str.contains(
                                story_keyword.strip(), case=False, na=False, regex=False
                            )
                        person_articles = person_articles[mask].copy()
                
                if not person_articles.empty:
                    # Store original for filter options
                    original_articles = person_articles.copy()
                    
                    # Filter Options by Categorical Columns (apply filters first, then show summary)
                    st.markdown("### Filter Options")
                    
                    # Source Type filter
                    if 'source_type' in person_articles.columns:
                        source_types = sorted(person_articles['source_type'].dropna().unique().tolist())
                        if source_types:
                            selected_sources = st.multiselect(
                                "Source Type",
                                source_types,
                                default=source_types,
                                key="filter_source_type"
                            )
                            if selected_sources:
                                person_articles = person_articles[person_articles['source_type'].isin(selected_sources)]
                    
                    # Publication filter
                    if 'publication_name' in person_articles.columns:
                        publications = sorted(original_articles['publication_name'].dropna().unique().tolist())
                        if len(publications) > 1:
                            selected_pubs = st.multiselect(
                                "Publication",
                                publications,
                                default=publications[:min(20, len(publications))] if len(publications) > 20 else publications,
                                key="filter_publication"
                            )
                            if selected_pubs:
                                person_articles = person_articles[person_articles['publication_name'].isin(selected_pubs)]
                    
                    # Cluster filter
                    cluster_col = 'cluster_label' if 'cluster_label' in original_articles.columns else 'cluster'
                    if cluster_col in original_articles.columns:
                        clusters = sorted(original_articles[cluster_col].dropna().unique().tolist())
                        if clusters:
                            selected_clusters = st.multiselect(
                                "Cluster",
                                clusters,
                                default=clusters,
                                key="filter_cluster"
                            )
                            if selected_clusters:
                                person_articles = person_articles[person_articles[cluster_col].isin(selected_clusters)]
                    
                    # Summary metrics (after filters are applied)
                    st.markdown("### Summary")
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Articles", len(person_articles))
                    with col2:
                        avg_sent = person_articles['sentiment_score'].mean() if 'sentiment_score' in person_articles.columns else 0
                        st.metric("Avg Sentiment", f"{avg_sent:.2f}")
                    with col3:
                        total_reach = person_articles['circulation_size'].sum() if 'circulation_size' in person_articles.columns else 0
                        st.metric("Total Reach", f"{total_reach:,.0f}")
                    with col4:
                        unique_pubs = person_articles['publication_name'].nunique() if 'publication_name' in person_articles.columns else 0
                        st.metric("Unique Publishers", unique_pubs)
                    
                    # Emotion distribution chart if available (using pre-computed emotion_body)
                    if 'emotion_body' in person_articles.columns:
                        emotion_counts = person_articles['emotion_body'].value_counts().dropna()
                        if not emotion_counts.empty:
                            st.markdown("### Emotion Distribution")
                            fig_emotions = px.bar(
                                x=emotion_counts.values,
                                y=emotion_counts.index,
                                orientation='h',
                                title=f'Emotions in Articles About {story_person}',
                                labels={'x': 'Number of Articles', 'y': 'Emotion'},
                                color=emotion_counts.values,
                                color_continuous_scale='RdYlBu'
                            )
                            fig_emotions.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                            st.plotly_chart(fig_emotions, use_container_width=True)
                    
                    # Articles/Headlines section - Table format
                    st.markdown("### Articles Mentioning This Person")
                    
                    # Prepare display columns for table
                    display_cols = []
                    if 'headline' in person_articles.columns:
                        display_cols.append('headline')
                    if 'publication_name' in person_articles.columns:
                        display_cols.append('publication_name')
                    if 'author_name' in person_articles.columns:
                        display_cols.append('author_name')
                    if 'sentiment_score' in person_articles.columns:
                        display_cols.append('sentiment_score')
                    if 'emotion_body' in person_articles.columns:
                        display_cols.append('emotion_body')
                    if 'circulation_size' in person_articles.columns:
                        display_cols.append('circulation_size')
                    if 'source_type' in person_articles.columns:
                        display_cols.append('source_type')
                    if 'article_url' in person_articles.columns:
                        display_cols.append('article_url')
                    
                    # Sort by circulation to show most impactful first (before formatting)
                    sort_col = 'circulation_size' if 'circulation_size' in person_articles.columns else None
                    if sort_col and sort_col in person_articles.columns:
                        person_articles_sorted = person_articles.nlargest(len(person_articles), sort_col)
                    else:
                        person_articles_sorted = person_articles.copy()
                    
                    # Create display dataframe from sorted data
                    display_df = person_articles_sorted[display_cols].copy()
                    
                    # Format sentiment_score for display
                    if 'sentiment_score' in display_df.columns:
                        display_df['sentiment_score'] = display_df['sentiment_score'].apply(
                            lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
                        )
                    
                    # Format circulation_size for display
                    if 'circulation_size' in display_df.columns:
                        display_df['circulation_size'] = display_df['circulation_size'].apply(
                            lambda x: f"{x:,.0f}" if pd.notna(x) else "N/A"
                        )
                    
                    # Truncate long headlines
                    if 'headline' in display_df.columns:
                        display_df['headline'] = display_df['headline'].apply(
                            lambda x: (str(x)[:80] + "...") if pd.notna(x) and len(str(x)) > 80 else x
                        )
                    
                    # Display table
                    st.dataframe(
                        display_df,
                        use_container_width=True,
                        height=400
                    )
                    
                    # Export CSV button
                    csv = person_articles.to_csv(index=False)
                    export_filename = f"{story_person.replace(' ', '_')}"
                    if story_keyword and story_keyword.strip():
                        export_filename += f"_{story_keyword.strip().replace(' ', '_')}"
                    export_filename += "_articles.csv"
                    
                    st.download_button(
                        label="📥 Export Articles Table to CSV",
                        data=csv,
                        file_name=export_filename,
                        mime="text/csv",
                        key="export_people_data"
                    )
                    st.caption(f"Exporting {len(person_articles):,} articles with all applied filters.")
                    
                    st.markdown("---")
                    
                    # Network Analysis Visualization
                    st.markdown("### Network Analysis")
                    st.markdown("**See how the person is being talked about through different channels and publications.**")
                    
                    # Check if we have source_type and publication_name columns
                    if 'source_type' in person_articles.columns and 'publication_name' in person_articles.columns:
                        # Aggregate data for network: Channel → Publication
                        channel_pub_counts = person_articles.groupby(['source_type', 'publication_name']).size().reset_index(name='count')
                        channel_pub_counts = channel_pub_counts.sort_values('count', ascending=False)
                        
                        # Limit to top channels and publications for readability
                        top_channels = person_articles['source_type'].value_counts().head(10).index.tolist()
                        top_pubs = person_articles['publication_name'].value_counts().head(10).index.tolist()
                        
                        filtered_edges = channel_pub_counts[
                            (channel_pub_counts['source_type'].isin(top_channels)) &
                            (channel_pub_counts['publication_name'].isin(top_pubs))
                        ]
                        
                        if not filtered_edges.empty:
                            try:
                                from streamlit_agraph import agraph, Node, Edge, Config
                                
                                # Create nodes
                                nodes = []
                                seen_ids = set()
                                
                                # Add channel nodes
                                for channel in top_channels[:6]:  # Limit to top 6
                                    channel_id = f"channel_{channel}"
                                    if channel_id not in seen_ids:
                                        seen_ids.add(channel_id)
                                        label = channel[:15] + "..." if len(channel) > 15 else channel
                                        nodes.append(Node(
                                            id=channel_id,
                                            label=label,
                                            size=30,
                                            color=PENTA_COLORS[0],  # Penta primary
                                            shape="circle",
                                            title=f"{channel}<br>Type: Channel"
                                        ))
                                
                                # Add publication nodes
                                for pub in top_pubs[:6]:  # Limit to top 6
                                    pub_id = f"pub_{pub}"
                                    if pub_id not in seen_ids:
                                        seen_ids.add(pub_id)
                                        label = pub[:15] + "..." if len(pub) > 15 else pub
                                        nodes.append(Node(
                                            id=pub_id,
                                            label=label,
                                            size=30,
                                            color=PENTA_COLORS[1],  # Penta green
                                            shape="triangle",
                                            title=f"{pub}<br>Type: Publication"
                                        ))
                                
                                # Create edges
                                edges = []
                                for _, row in filtered_edges.head(20).iterrows():  # Limit to top 20 edges
                                    source_id = f"channel_{row['source_type']}"
                                    target_id = f"pub_{row['publication_name']}"
                                    if source_id in seen_ids and target_id in seen_ids:
                                        edges.append(Edge(
                                            source=source_id,
                                            target=target_id,
                                            width=max(2, int(row['count'] * 2)),
                                            color="#999999",
                                            label=str(row['count'])
                                        ))
                                
                                # Configuration
                                config = Config(
                                    width="100%",
                                    height=400,
                                    directed=True,
                                    physics={
                                        "enabled": True,
                                        "stabilization": {"enabled": True, "iterations": 50},
                                        "barnesHut": {
                                            "gravitationalConstant": -4000,
                                            "centralGravity": 0.2,
                                            "springLength": 100,
                                            "springConstant": 0.02,
                                            "damping": 0.1
                                        }
                                    },
                                    hierarchical=False,
                                    nodeHighlightBehavior=True,
                                    highlightColor="#F7A7A6",
                                    collapsible=False,
                                    node={'labelProperty': 'label'},
                                    link={'labelProperty': 'label', 'renderLabel': False}
                                )
                                
                                agraph(nodes=nodes, edges=edges, config=config)
                                
                                with st.expander("What to Look For", expanded=False):
                                    st.markdown("""
                                    - **Blue Circles**: Channels (source types)
                                    - **Green Triangles**: Publications
                                    - **Connections**: Lines show channel-publication relationships
                                    - **Thick Lines**: Stronger relationships (more articles)
                                    - **Hub Nodes**: Channels/publications connected to many others
                                    """)
                                
                            except ImportError:
                                st.warning("streamlit-agraph not available. Install with: pip install streamlit-agraph")
                                # Fallback to text-based visualization
                                st.info("Showing top channel-publication relationships:")
                                st.dataframe(filtered_edges.head(20))
                            except Exception as e:
                                st.error(f"Error creating network visualization: {e}")
                                st.info("Showing top channel-publication relationships:")
                                st.dataframe(filtered_edges.head(20))
                        else:
                            st.info("Not enough data for network visualization.")
                    else:
                        st.info("Network analysis requires 'source_type' and 'publication_name' columns.")
                    
                    st.markdown("---")
                    
                    # Sentiment vs Circulation Quadrant Plot
                    if 'sentiment_score' in person_articles.columns and 'circulation_size' in person_articles.columns:
                        st.markdown("### Sentiment vs Circulation Analysis")
                        st.markdown(f"**Showing where {story_person} lands in sentiment vs circulation space (quadrant analysis).**")
                        
                        # Calculate person's average metrics
                        person_avg_sentiment = person_articles['sentiment_score'].mean()
                        person_avg_circulation = person_articles['circulation_size'].mean()
                        
                        # Determine thresholds for quadrants
                        sentiment_threshold = 0.0  # Sentiment threshold at 0 (positive vs negative)
                        
                        # Get all articles for comparison (if available) to determine circulation threshold
                        if final_df is not None and not final_df.empty:
                            comparison_df = final_df[['sentiment_score', 'circulation_size']].dropna()
                            comparison_df = comparison_df[
                                (comparison_df['sentiment_score'].notna()) & 
                                (comparison_df['circulation_size'].notna())
                            ]
                            
                            if not comparison_df.empty:
                                circulation_threshold = comparison_df['circulation_size'].median()
                                
                                # Create quadrant plot
                                fig_sent_circ = go.Figure()
                                
                                # Add quadrant background rectangles
                                x_min = comparison_df['sentiment_score'].min()
                                x_max = comparison_df['sentiment_score'].max()
                                y_min = comparison_df['circulation_size'].min()
                                y_max = comparison_df['circulation_size'].max()
                                
                                # Quadrant 1: High Sentiment, High Circulation (top-right)
                                fig_sent_circ.add_shape(
                                    type="rect",
                                    x0=max(sentiment_threshold, x_min), y0=max(circulation_threshold, y_min),
                                    x1=x_max, y1=y_max,
                                    fillcolor="rgba(74, 180, 142, 0.1)",  # Light green
                                    line=dict(width=0)
                                )
                                
                                # Quadrant 2: Low Sentiment, High Circulation (top-left)
                                fig_sent_circ.add_shape(
                                    type="rect",
                                    x0=x_min, y0=max(circulation_threshold, y_min),
                                    x1=min(sentiment_threshold, x_max), y1=y_max,
                                    fillcolor="rgba(217, 72, 65, 0.1)",  # Light red
                                    line=dict(width=0)
                                )
                                
                                # Quadrant 3: Low Sentiment, Low Circulation (bottom-left)
                                fig_sent_circ.add_shape(
                                    type="rect",
                                    x0=x_min, y0=y_min,
                                    x1=min(sentiment_threshold, x_max), y1=min(circulation_threshold, y_max),
                                    fillcolor="rgba(142, 54, 107, 0.1)",  # Light purple
                                    line=dict(width=0)
                                )
                                
                                # Quadrant 4: High Sentiment, Low Circulation (bottom-right)
                                fig_sent_circ.add_shape(
                                    type="rect",
                                    x0=max(sentiment_threshold, x_min), y0=y_min,
                                    x1=x_max, y1=min(circulation_threshold, y_max),
                                    fillcolor="rgba(212, 161, 21, 0.1)",  # Light gold
                                    line=dict(width=0)
                                )
                                
                                # Add dividing lines
                                fig_sent_circ.add_shape(
                                    type="line",
                                    x0=sentiment_threshold, y0=y_min,
                                    x1=sentiment_threshold, y1=y_max,
                                    line=dict(color="black", width=2, dash="dash")
                                )
                                
                                fig_sent_circ.add_shape(
                                    type="line",
                                    x0=x_min, y0=circulation_threshold,
                                    x1=x_max, y1=circulation_threshold,
                                    line=dict(color="black", width=2, dash="dash")
                                )
                                
                                # Add quadrant labels
                                fig_sent_circ.add_annotation(
                                    x=(sentiment_threshold + x_max) / 2,
                                    y=(circulation_threshold + y_max) / 2,
                                    text="High Sentiment<br>High Circulation",
                                    showarrow=False,
                                    font=dict(size=12, color=PENTA_DARK),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor=PENTA_COLORS[1],
                                    borderwidth=1
                                )
                                
                                fig_sent_circ.add_annotation(
                                    x=(x_min + sentiment_threshold) / 2,
                                    y=(circulation_threshold + y_max) / 2,
                                    text="Low Sentiment<br>High Circulation",
                                    showarrow=False,
                                    font=dict(size=12, color=PENTA_DARK),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor=PENTA_COLORS[5],
                                    borderwidth=1
                                )
                                
                                fig_sent_circ.add_annotation(
                                    x=(x_min + sentiment_threshold) / 2,
                                    y=(y_min + circulation_threshold) / 2,
                                    text="Low Sentiment<br>Low Circulation",
                                    showarrow=False,
                                    font=dict(size=12, color=PENTA_DARK),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor=PENTA_COLORS[3],
                                    borderwidth=1
                                )
                                
                                fig_sent_circ.add_annotation(
                                    x=(sentiment_threshold + x_max) / 2,
                                    y=(y_min + circulation_threshold) / 2,
                                    text="High Sentiment<br>Low Circulation",
                                    showarrow=False,
                                    font=dict(size=12, color=PENTA_DARK),
                                    bgcolor="rgba(255,255,255,0.8)",
                                    bordercolor=PENTA_COLORS[4],
                                    borderwidth=1
                                )
                                
                                # Add all articles as background points
                                fig_sent_circ.add_trace(go.Scatter(
                                    x=comparison_df['sentiment_score'],
                                    y=comparison_df['circulation_size'],
                                    mode='markers',
                                    name='All Articles',
                                    marker=dict(
                                        color='lightgray',
                                        size=5,
                                        opacity=0.3
                                    ),
                                    hovertemplate='<b>All Articles</b><br>Sentiment: %{x:.2f}<br>Circulation: %{y:,.0f}<extra></extra>'
                                ))
                                
                                # Add person's articles
                                fig_sent_circ.add_trace(go.Scatter(
                                    x=person_articles['sentiment_score'],
                                    y=person_articles['circulation_size'],
                                    mode='markers',
                                    name=f'{story_person} Articles',
                                    marker=dict(
                                        color=PENTA_COLORS[0],
                                        size=8,
                                        opacity=0.7
                                    ),
                                    hovertemplate=f'<b>{story_person}</b><br>Sentiment: %{{x:.2f}}<br>Circulation: %{{y:,.0f}}<extra></extra>'
                                ))
                                
                                # Add average point
                                fig_sent_circ.add_trace(go.Scatter(
                                    x=[person_avg_sentiment],
                                    y=[person_avg_circulation],
                                    mode='markers',
                                    name=f'{story_person} Average',
                                    marker=dict(
                                        color=PENTA_COLORS[1],
                                        size=15,
                                        symbol='star'
                                    ),
                                    hovertemplate=f'<b>{story_person} Average</b><br>Sentiment: {person_avg_sentiment:.2f}<br>Circulation: {person_avg_circulation:,.0f}<extra></extra>'
                                ))
                                
                                fig_sent_circ.update_layout(
                                    title=f"Sentiment vs Circulation Quadrant Analysis: {story_person}",
                                    xaxis_title="Sentiment Score",
                                    yaxis_title="Circulation Size",
                                    height=600,
                                    hovermode='closest',
                                    font=dict(family="Poppins, sans-serif", size=12, color=PENTA_DARK),
                                    title_font=dict(family="Poppins, sans-serif", size=18, color=PENTA_DARK),
                                    showlegend=True
                                )
                                
                                st.plotly_chart(fig_sent_circ, use_container_width=True)
                                
                                # Show quadrant summary
                                person_articles_clean = person_articles[
                                    person_articles['sentiment_score'].notna() & 
                                    person_articles['circulation_size'].notna()
                                ]
                                
                                q1_count = len(person_articles_clean[
                                    (person_articles_clean['sentiment_score'] > sentiment_threshold) &
                                    (person_articles_clean['circulation_size'] > circulation_threshold)
                                ])
                                q2_count = len(person_articles_clean[
                                    (person_articles_clean['sentiment_score'] <= sentiment_threshold) &
                                    (person_articles_clean['circulation_size'] > circulation_threshold)
                                ])
                                q3_count = len(person_articles_clean[
                                    (person_articles_clean['sentiment_score'] <= sentiment_threshold) &
                                    (person_articles_clean['circulation_size'] <= circulation_threshold)
                                ])
                                q4_count = len(person_articles_clean[
                                    (person_articles_clean['sentiment_score'] > sentiment_threshold) &
                                    (person_articles_clean['circulation_size'] <= circulation_threshold)
                                ])
                                
                                col1, col2, col3, col4 = st.columns(4)
                                with col1:
                                    st.metric("High Sentiment / High Circulation", q1_count)
                                with col2:
                                    st.metric("Low Sentiment / High Circulation", q2_count)
                                with col3:
                                    st.metric("Low Sentiment / Low Circulation", q3_count)
                                with col4:
                                    st.metric("High Sentiment / Low Circulation", q4_count)
                                
                            else:
                                st.info("No comparison data available.")
                        else:
                            # Just show person's articles with quadrants
                            circulation_threshold = person_articles['circulation_size'].median()
                            sentiment_threshold = 0.0
                            
                            fig_sent_circ = go.Figure()
                            
                            x_min = person_articles['sentiment_score'].min()
                            x_max = person_articles['sentiment_score'].max()
                            y_min = person_articles['circulation_size'].min()
                            y_max = person_articles['circulation_size'].max()
                            
                            # Add quadrant backgrounds and lines (similar to above)
                            fig_sent_circ.add_shape(
                                type="rect",
                                x0=max(sentiment_threshold, x_min), y0=max(circulation_threshold, y_min),
                                x1=x_max, y1=y_max,
                                fillcolor="rgba(74, 180, 142, 0.1)",
                                line=dict(width=0)
                            )
                            
                            fig_sent_circ.add_shape(
                                type="line",
                                x0=sentiment_threshold, y0=y_min,
                                x1=sentiment_threshold, y1=y_max,
                                line=dict(color="black", width=2, dash="dash")
                            )
                            
                            fig_sent_circ.add_shape(
                                type="line",
                                x0=x_min, y0=circulation_threshold,
                                x1=x_max, y1=circulation_threshold,
                                line=dict(color="black", width=2, dash="dash")
                            )
                            
                            fig_sent_circ.add_trace(go.Scatter(
                                x=person_articles['sentiment_score'],
                                y=person_articles['circulation_size'],
                                mode='markers',
                                name=f'{story_person} Articles',
                                marker=dict(
                                    color=PENTA_COLORS[0],
                                    size=8,
                                    opacity=0.7
                                ),
                                hovertemplate=f'<b>{story_person}</b><br>Sentiment: %{{x:.2f}}<br>Circulation: %{{y:,.0f}}<extra></extra>'
                            ))
                            
                            fig_sent_circ.add_trace(go.Scatter(
                                x=[person_avg_sentiment],
                                y=[person_avg_circulation],
                                mode='markers',
                                name=f'{story_person} Average',
                                marker=dict(
                                    color=PENTA_COLORS[1],
                                    size=15,
                                    symbol='star'
                                ),
                                hovertemplate=f'<b>{story_person} Average</b><br>Sentiment: {person_avg_sentiment:.2f}<br>Circulation: {person_avg_circulation:,.0f}<extra></extra>'
                            ))
                            
                            fig_sent_circ.update_layout(
                                title=f"Sentiment vs Circulation Quadrant Analysis: {story_person}",
                                xaxis_title="Sentiment Score",
                                yaxis_title="Circulation Size",
                                height=600,
                                hovermode='closest',
                                font=dict(family="Poppins, sans-serif", size=12, color=PENTA_DARK),
                                title_font=dict(family="Poppins, sans-serif", size=18, color=PENTA_DARK)
                            )
                            
                            st.plotly_chart(fig_sent_circ, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Emotions vs Circulation Quadrant Plot
                    if 'emotion_body' in person_articles.columns and 'circulation_size' in person_articles.columns:
                        st.markdown("### Emotions vs Circulation Analysis")
                        st.markdown(f"**Showing where {story_person} lands in emotions vs circulation space (quadrant analysis).**")
                        
                        # Categorize emotions: positive emotions (joy, surprise) vs negative emotions (sadness, anger, fear, disgust)
                        positive_emotions = ['joy', 'surprise']
                        negative_emotions = ['sadness', 'anger', 'fear', 'disgust']
                        
                        # Prepare data - map emotions to numeric values: 1 for positive, 0 for negative
                        def emotion_to_numeric(emotion):
                            emotion_str = str(emotion).lower().strip()
                            return 1 if emotion_str in positive_emotions else 0
                        
                        person_articles_emotion = person_articles[
                            person_articles['emotion_body'].notna() & 
                            person_articles['circulation_size'].notna()
                        ].copy()
                        
                        if not person_articles_emotion.empty:
                            person_articles_emotion['emotion_positive'] = person_articles_emotion['emotion_body'].apply(emotion_to_numeric)
                            
                            # Get all articles for comparison (if available) to determine circulation threshold
                            if final_df is not None and not final_df.empty:
                                comparison_df_emotion = final_df[
                                    ['emotion_body', 'circulation_size']
                                ].dropna()
                                comparison_df_emotion = comparison_df_emotion[
                                    (comparison_df_emotion['emotion_body'].notna()) & 
                                    (comparison_df_emotion['circulation_size'].notna())
                                ]
                                
                                if not comparison_df_emotion.empty:
                                    comparison_df_emotion['emotion_positive'] = comparison_df_emotion['emotion_body'].apply(emotion_to_numeric)
                                    circulation_threshold = comparison_df_emotion['circulation_size'].median()
                                    emotion_threshold = 0.5  # Split between positive (1) and negative (0)
                                    
                                    # Create quadrant plot
                                    fig_emotion_circ = go.Figure()
                                    
                                    # Add quadrant background rectangles
                                    x_min = comparison_df_emotion['circulation_size'].min()
                                    x_max = comparison_df_emotion['circulation_size'].max()
                                    y_min = -0.2
                                    y_max = 1.2
                                    
                                    # Quadrant 1: Positive Emotions, High Circulation (top-right)
                                    fig_emotion_circ.add_shape(
                                        type="rect",
                                        x0=max(circulation_threshold, x_min), y0=max(emotion_threshold, y_min),
                                        x1=x_max, y1=y_max,
                                        fillcolor="rgba(74, 180, 142, 0.1)",  # Light green
                                        line=dict(width=0)
                                    )
                                    
                                    # Quadrant 2: Negative Emotions, High Circulation (top-left)
                                    fig_emotion_circ.add_shape(
                                        type="rect",
                                        x0=x_min, y0=max(emotion_threshold, y_min),
                                        x1=min(circulation_threshold, x_max), y1=y_max,
                                        fillcolor="rgba(217, 72, 65, 0.1)",  # Light red
                                        line=dict(width=0)
                                    )
                                    
                                    # Quadrant 3: Negative Emotions, Low Circulation (bottom-left)
                                    fig_emotion_circ.add_shape(
                                        type="rect",
                                        x0=x_min, y0=y_min,
                                        x1=min(circulation_threshold, x_max), y1=min(emotion_threshold, y_max),
                                        fillcolor="rgba(142, 54, 107, 0.1)",  # Light purple
                                        line=dict(width=0)
                                    )
                                    
                                    # Quadrant 4: Positive Emotions, Low Circulation (bottom-right)
                                    fig_emotion_circ.add_shape(
                                        type="rect",
                                        x0=max(circulation_threshold, x_min), y0=y_min,
                                        x1=x_max, y1=min(emotion_threshold, y_max),
                                        fillcolor="rgba(212, 161, 21, 0.1)",  # Light gold
                                        line=dict(width=0)
                                    )
                                    
                                    # Add dividing lines
                                    fig_emotion_circ.add_shape(
                                        type="line",
                                        x0=circulation_threshold, y0=y_min,
                                        x1=circulation_threshold, y1=y_max,
                                        line=dict(color="black", width=2, dash="dash")
                                    )
                                    
                                    fig_emotion_circ.add_shape(
                                        type="line",
                                        x0=x_min, y0=emotion_threshold,
                                        x1=x_max, y1=emotion_threshold,
                                        line=dict(color="black", width=2, dash="dash")
                                    )
                                    
                                    # Add quadrant labels
                                    fig_emotion_circ.add_annotation(
                                        x=(circulation_threshold + x_max) / 2,
                                        y=(emotion_threshold + y_max) / 2,
                                        text="Positive Emotions<br>High Circulation",
                                        showarrow=False,
                                        font=dict(size=12, color=PENTA_DARK),
                                        bgcolor="rgba(255,255,255,0.8)",
                                        bordercolor=PENTA_COLORS[1],
                                        borderwidth=1
                                    )
                                    
                                    fig_emotion_circ.add_annotation(
                                        x=(x_min + circulation_threshold) / 2,
                                        y=(emotion_threshold + y_max) / 2,
                                        text="Negative Emotions<br>High Circulation",
                                        showarrow=False,
                                        font=dict(size=12, color=PENTA_DARK),
                                        bgcolor="rgba(255,255,255,0.8)",
                                        bordercolor=PENTA_COLORS[5],
                                        borderwidth=1
                                    )
                                    
                                    fig_emotion_circ.add_annotation(
                                        x=(x_min + circulation_threshold) / 2,
                                        y=(y_min + emotion_threshold) / 2,
                                        text="Negative Emotions<br>Low Circulation",
                                        showarrow=False,
                                        font=dict(size=12, color=PENTA_DARK),
                                        bgcolor="rgba(255,255,255,0.8)",
                                        bordercolor=PENTA_COLORS[3],
                                        borderwidth=1
                                    )
                                    
                                    fig_emotion_circ.add_annotation(
                                        x=(circulation_threshold + x_max) / 2,
                                        y=(y_min + emotion_threshold) / 2,
                                        text="Positive Emotions<br>Low Circulation",
                                        showarrow=False,
                                        font=dict(size=12, color=PENTA_DARK),
                                        bgcolor="rgba(255,255,255,0.8)",
                                        bordercolor=PENTA_COLORS[4],
                                        borderwidth=1
                                    )
                                    
                                    # Add all articles as background points
                                    fig_emotion_circ.add_trace(go.Scatter(
                                        x=comparison_df_emotion['circulation_size'],
                                        y=comparison_df_emotion['emotion_positive'],
                                        mode='markers',
                                        name='All Articles',
                                        marker=dict(
                                            color='lightgray',
                                            size=5,
                                            opacity=0.3
                                        ),
                                        hovertemplate='<b>All Articles</b><br>Emotion: %{text}<br>Circulation: %{x:,.0f}<extra></extra>',
                                        text=comparison_df_emotion['emotion_body']
                                    ))
                                    
                                    # Add person's articles
                                    fig_emotion_circ.add_trace(go.Scatter(
                                        x=person_articles_emotion['circulation_size'],
                                        y=person_articles_emotion['emotion_positive'],
                                        mode='markers',
                                        name=f'{story_person} Articles',
                                        marker=dict(
                                            color=PENTA_COLORS[0],
                                            size=8,
                                            opacity=0.7
                                        ),
                                        hovertemplate=f'<b>{story_person}</b><br>Emotion: %{{text}}<br>Circulation: %{{x:,.0f}}<extra></extra>',
                                        text=person_articles_emotion['emotion_body']
                                    ))
                                    
                                    # Calculate and add average point
                                    person_avg_emotion_positive = person_articles_emotion['emotion_positive'].mean()
                                    person_avg_circulation_emotion = person_articles_emotion['circulation_size'].mean()
                                    
                                    fig_emotion_circ.add_trace(go.Scatter(
                                        x=[person_avg_circulation_emotion],
                                        y=[person_avg_emotion_positive],
                                        mode='markers',
                                        name=f'{story_person} Average',
                                        marker=dict(
                                            color=PENTA_COLORS[1],
                                            size=15,
                                            symbol='star'
                                        ),
                                        hovertemplate=f'<b>{story_person} Average</b><br>Circulation: {person_avg_circulation_emotion:,.0f}<extra></extra>'
                                    ))
                                    
                                    fig_emotion_circ.update_layout(
                                        title=f"Emotions vs Circulation Quadrant Analysis: {story_person}",
                                        xaxis_title="Circulation Size",
                                        yaxis_title="Emotion Type",
                                        yaxis=dict(
                                            tickmode='array',
                                            tickvals=[0, 1],
                                            ticktext=['Negative Emotions', 'Positive Emotions']
                                        ),
                                        height=600,
                                        hovermode='closest',
                                        font=dict(family="Poppins, sans-serif", size=12, color=PENTA_DARK),
                                        title_font=dict(family="Poppins, sans-serif", size=18, color=PENTA_DARK),
                                        showlegend=True
                                    )
                                    
                                    st.plotly_chart(fig_emotion_circ, use_container_width=True)
                                    
                                    # Show quadrant summary
                                    person_articles_clean = person_articles_emotion[
                                        person_articles_emotion['emotion_positive'].notna() & 
                                        person_articles_emotion['circulation_size'].notna()
                                    ]
                                    
                                    q1_count = len(person_articles_clean[
                                        (person_articles_clean['emotion_positive'] > emotion_threshold) &
                                        (person_articles_clean['circulation_size'] > circulation_threshold)
                                    ])
                                    q2_count = len(person_articles_clean[
                                        (person_articles_clean['emotion_positive'] <= emotion_threshold) &
                                        (person_articles_clean['circulation_size'] > circulation_threshold)
                                    ])
                                    q3_count = len(person_articles_clean[
                                        (person_articles_clean['emotion_positive'] <= emotion_threshold) &
                                        (person_articles_clean['circulation_size'] <= circulation_threshold)
                                    ])
                                    q4_count = len(person_articles_clean[
                                        (person_articles_clean['emotion_positive'] > emotion_threshold) &
                                        (person_articles_clean['circulation_size'] <= circulation_threshold)
                                    ])
                                    
                                    col1, col2, col3, col4 = st.columns(4)
                                    with col1:
                                        st.metric("Positive Emotions / High Circulation", q1_count)
                                    with col2:
                                        st.metric("Negative Emotions / High Circulation", q2_count)
                                    with col3:
                                        st.metric("Negative Emotions / Low Circulation", q3_count)
                                    with col4:
                                        st.metric("Positive Emotions / Low Circulation", q4_count)
                                    
                                else:
                                    st.info("No comparison data available.")
                            else:
                                # Just show person's articles with quadrants
                                circulation_threshold = person_articles_emotion['circulation_size'].median()
                                emotion_threshold = 0.5
                                
                                fig_emotion_circ = go.Figure()
                                
                                x_min = person_articles_emotion['circulation_size'].min()
                                x_max = person_articles_emotion['circulation_size'].max()
                                y_min = -0.2
                                y_max = 1.2
                                
                                # Add quadrant backgrounds and lines
                                fig_emotion_circ.add_shape(
                                    type="rect",
                                    x0=max(circulation_threshold, x_min), y0=max(emotion_threshold, y_min),
                                    x1=x_max, y1=y_max,
                                    fillcolor="rgba(74, 180, 142, 0.1)",
                                    line=dict(width=0)
                                )
                                
                                fig_emotion_circ.add_shape(
                                    type="line",
                                    x0=circulation_threshold, y0=y_min,
                                    x1=circulation_threshold, y1=y_max,
                                    line=dict(color="black", width=2, dash="dash")
                                )
                                
                                fig_emotion_circ.add_shape(
                                    type="line",
                                    x0=x_min, y0=emotion_threshold,
                                    x1=x_max, y1=emotion_threshold,
                                    line=dict(color="black", width=2, dash="dash")
                                )
                                
                                fig_emotion_circ.add_trace(go.Scatter(
                                    x=person_articles_emotion['circulation_size'],
                                    y=person_articles_emotion['emotion_positive'],
                                    mode='markers',
                                    name=f'{story_person} Articles',
                                    marker=dict(
                                        color=PENTA_COLORS[0],
                                        size=8,
                                        opacity=0.7
                                    ),
                                    hovertemplate=f'<b>{story_person}</b><br>Emotion: %{{text}}<br>Circulation: %{{x:,.0f}}<extra></extra>',
                                    text=person_articles_emotion['emotion_body']
                                ))
                                
                                person_avg_emotion_positive = person_articles_emotion['emotion_positive'].mean()
                                person_avg_circulation_emotion = person_articles_emotion['circulation_size'].mean()
                                
                                fig_emotion_circ.add_trace(go.Scatter(
                                    x=[person_avg_circulation_emotion],
                                    y=[person_avg_emotion_positive],
                                    mode='markers',
                                    name=f'{story_person} Average',
                                    marker=dict(
                                        color=PENTA_COLORS[1],
                                        size=15,
                                        symbol='star'
                                    ),
                                    hovertemplate=f'<b>{story_person} Average</b><br>Circulation: {person_avg_circulation_emotion:,.0f}<extra></extra>'
                                ))
                                
                                fig_emotion_circ.update_layout(
                                    title=f"Emotions vs Circulation Quadrant Analysis: {story_person}",
                                    xaxis_title="Circulation Size",
                                    yaxis_title="Emotion Type",
                                    yaxis=dict(
                                        tickmode='array',
                                        tickvals=[0, 1],
                                        ticktext=['Negative Emotions', 'Positive Emotions']
                                    ),
                                    height=600,
                                    hovermode='closest',
                                    font=dict(family="Poppins, sans-serif", size=12, color=PENTA_DARK),
                                    title_font=dict(family="Poppins, sans-serif", size=18, color=PENTA_DARK)
                                )
                                
                                st.plotly_chart(fig_emotion_circ, use_container_width=True)
                    
                    st.markdown("---")
                    
                    # Emotion and Sentiment Breakdown by Topic/Cluster
                    cluster_col = 'cluster_label' if 'cluster_label' in person_articles.columns else ('cluster' if 'cluster' in person_articles.columns else None)
                    
                    if cluster_col and cluster_col in person_articles.columns:
                        st.markdown("### Emotion and Sentiment Breakdown by Topic")
                        st.markdown(f"**Showing how {story_person} is discussed across different topics/clusters.**")
                        
                        # Prepare data for breakdown
                        breakdown_data = []
                        
                        for cluster in person_articles[cluster_col].dropna().unique():
                            cluster_data = person_articles[person_articles[cluster_col] == cluster]
                            
                            # Emotion breakdown
                            if 'emotion_body' in cluster_data.columns:
                                emotion_counts = cluster_data['emotion_body'].value_counts().to_dict()
                                for emotion, count in emotion_counts.items():
                                    breakdown_data.append({
                                        'Cluster': str(cluster),
                                        'Type': 'Emotion',
                                        'Category': str(emotion).title(),
                                        'Count': count,
                                        'Avg Sentiment': cluster_data['sentiment_score'].mean() if 'sentiment_score' in cluster_data.columns else 0
                                    })
                            
                            # Sentiment breakdown
                            if 'sentiment_score' in cluster_data.columns:
                                sentiment_bands = pd.cut(
                                    cluster_data['sentiment_score'],
                                    bins=[-float('inf'), -0.1, 0.1, float('inf')],
                                    labels=['Negative', 'Neutral', 'Positive']
                                )
                                sentiment_counts = sentiment_bands.value_counts().to_dict()
                                for sentiment, count in sentiment_counts.items():
                                    breakdown_data.append({
                                        'Cluster': str(cluster),
                                        'Type': 'Sentiment',
                                        'Category': str(sentiment),
                                        'Count': count,
                                        'Avg Sentiment': cluster_data['sentiment_score'].mean() if 'sentiment_score' in cluster_data.columns else 0
                                    })
                        
                        if breakdown_data:
                            breakdown_df = pd.DataFrame(breakdown_data)
                            
                            # Create grouped bar chart
                            fig_breakdown = go.Figure()
                            
                            clusters = sorted(breakdown_df['Cluster'].unique())
                            colors_emotion = {e: PENTA_COLORS[i % len(PENTA_COLORS)] for i, e in enumerate(breakdown_df[breakdown_df['Type'] == 'Emotion']['Category'].unique())}
                            colors_sentiment = {s: PENTA_COLORS[i % len(PENTA_COLORS)] for i, s in enumerate(['Negative', 'Neutral', 'Positive'])}
                            
                            # Add emotion bars
                            emotion_data = breakdown_df[breakdown_df['Type'] == 'Emotion']
                            if not emotion_data.empty:
                                for emotion in emotion_data['Category'].unique():
                                    emotion_subset = emotion_data[emotion_data['Category'] == emotion]
                                    fig_breakdown.add_trace(go.Bar(
                                        name=f"Emotion: {emotion}",
                                        x=emotion_subset['Cluster'],
                                        y=emotion_subset['Count'],
                                        marker_color=colors_emotion.get(emotion, PENTA_COLORS[0]),
                                        hovertemplate=f'<b>{emotion}</b><br>Cluster: %{{x}}<br>Count: %{{y}}<extra></extra>'
                                    ))
                            
                            # Add sentiment bars
                            sentiment_data = breakdown_df[breakdown_df['Type'] == 'Sentiment']
                            if not sentiment_data.empty:
                                for sentiment in ['Negative', 'Neutral', 'Positive']:
                                    sentiment_subset = sentiment_data[sentiment_data['Category'] == sentiment]
                                    if not sentiment_subset.empty:
                                        fig_breakdown.add_trace(go.Bar(
                                            name=f"Sentiment: {sentiment}",
                                            x=sentiment_subset['Cluster'],
                                            y=sentiment_subset['Count'],
                                            marker_color=colors_sentiment.get(sentiment, PENTA_COLORS[0]),
                                            hovertemplate=f'<b>{sentiment}</b><br>Cluster: %{{x}}<br>Count: %{{y}}<extra></extra>'
                                        ))
                            
                            fig_breakdown.update_layout(
                                title=f"Emotion and Sentiment Breakdown by Topic: {story_person}",
                                xaxis_title="Topic/Cluster",
                                yaxis_title="Count",
                                barmode='group',
                                height=500,
                                hovermode='closest',
                                font=dict(family="Poppins, sans-serif", size=12, color=PENTA_DARK),
                                title_font=dict(family="Poppins, sans-serif", size=18, color=PENTA_DARK)
                            )
                            
                            st.plotly_chart(fig_breakdown, use_container_width=True)
                        else:
                            st.info("No breakdown data available.")
                    else:
                        st.info("Topic/Cluster breakdown requires cluster data.")
                    
                else:
                    st.warning(f"No article data found for {story_person}" + (f" with keyword '{story_keyword}'" if story_keyword and story_keyword.strip() else "") + ".")
            else:
                if person_rows.empty:
                    st.info(f"No articles found mentioning '{story_person}'. The person may not be in the persons_by_row data.")
                else:
                    st.info(f"Found mentions of {story_person}, but unable to link to article data. Ensure final_dataset_with_attribution.parquet is available.")
        else:
                st.info("👆 Select a person from the dropdown above to see the story being told about them.")

if __name__ == "__main__":
    main()

# -------------------- Dataset Footnote --------------------
st.markdown("---")
st.markdown(
    """
    <div style="font-size: 0.8em; text-align: left; margin-top: 2rem;">
        Built by Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammad Waqas, Mark Saba, Posy Olivetti
    </div>
    """,
    unsafe_allow_html=True
)
