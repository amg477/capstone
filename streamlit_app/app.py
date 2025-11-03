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
    load_final_dataset,
    load_persons_by_row
)
from data_processors import (
    clean_bin_column,
    extract_clean_names,
    is_likely_person_name
)
from charts import (
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

@st.cache_data(show_spinner=False)
def get_normalized_person_list(influencer_df):
    """Get normalized person list - cached for performance"""
    if influencer_df is None or influencer_df.empty or 'person_list' not in influencer_df.columns:
        return []
    
    def normalize_robert_kennedy(name_str):
        if pd.isna(name_str):
            return name_str
        name_lower = str(name_str).lower().strip()
        if 'robert' in name_lower and ('kennedy' in name_lower or 'junior' in name_lower):
            return "Robert Kennedy"
        return name_str
    
    def normalize_trump(name):
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
    
    # Get person counts and filter to only include persons with count > 1
    person_counts = influencer_df['person_list'].dropna().astype(str).value_counts()
    
    # Filter to persons appearing more than once
    persons_with_multiple = person_counts[person_counts > 1]
    
    # Limit to top persons for performance (if too many)
    if len(persons_with_multiple) > 5000:
        # Take top N by frequency
        raw_persons = persons_with_multiple.head(5000).index.tolist()
    else:
        raw_persons = persons_with_multiple.index.tolist()
    
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
    return sorted(list(set(normalized_persons)))


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

    # Load essential data upfront (cached, so first load only)
    influencer_df = load_influencer_table()
    
    # Load tab-specific data lazily (only when needed)
    # These will be loaded when user switches to People tab
    final_df = None
    persons_by_row_df = None
    
    # Clean bin columns if they exist
    if influencer_df is not None:
        if 'circulation_size_bin' in influencer_df.columns:
            influencer_df = clean_bin_column(influencer_df, 'circulation_size_bin')
        if 'sentiment_score_bin' in influencer_df.columns:
            influencer_df = clean_bin_column(influencer_df, 'sentiment_score_bin')
    
    # Emotion data is now pre-computed in the final_dataset_with_attribution.parquet as 'emotion_body' column
    
    if influencer_df is None:
        st.error("Unable to load influencer table. Please ensure the data files are in the correct location.")
        st.stop()
    
    # Start with unfiltered data - filters will be applied in Pulse tab
    filtered_df = influencer_df.copy()
    
    # Tabs for different views
    tab1, tab2 = st.tabs(["PolicyPath", "People"])
    
    with tab1:
        # Lazy load final_df if needed for filters (only if not already loaded)
        if final_df is None:
            final_df = load_final_dataset()
            if final_df is not None:
                if 'circulation_size_bin' in final_df.columns:
                    final_df = clean_bin_column(final_df, 'circulation_size_bin')
                if 'sentiment_score_bin' in final_df.columns:
                    final_df = clean_bin_column(final_df, 'sentiment_score_bin')
        
        # Lazy load persons_by_row_df if needed for emotion analysis
        if persons_by_row_df is None:
            persons_by_row_df = load_persons_by_row()
        
        # Filters Section
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

        # Get normalized person list (cached for performance)
        try:
            all_persons = get_normalized_person_list(influencer_df)
            if not all_persons:
                all_persons = []
        except Exception as e:
            st.warning(f"Error loading person list: {e}")
            all_persons = []
        
        # Cluster filter
        try:
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
        except Exception as e:
            st.warning(f"Error loading clusters: {e}")
            clusters = []
            cluster_col = None
            
        # Filter inputs in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            try:
                selected_persons = st.multiselect(
                    "Select Individuals",
                    all_persons,
                    default=[],
                    key="filter_persons"
                )
            except Exception as e:
                st.error(f"Error showing person filter: {e}")
                selected_persons = []
        
        with filter_col2:
            if clusters:
                selected_clusters = st.multiselect(
                    "Select Clusters",
                    clusters,
                    default=[],
                    key="filter_clusters"
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
                    default=[],
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
        # Lazy load data for People tab (only when tab is accessed)
        if final_df is None:
            with st.spinner("Loading People tab data..."):
                final_df = load_final_dataset()
                persons_by_row_df = load_persons_by_row()
            
            # Clean bin columns if needed
            if final_df is not None:
                if 'circulation_size_bin' in final_df.columns:
                    final_df = clean_bin_column(final_df, 'circulation_size_bin')
                if 'sentiment_score_bin' in final_df.columns:
                    final_df = clean_bin_column(final_df, 'sentiment_score_bin')
        
        st.markdown('<div class="section-header">People</div>', unsafe_allow_html=True)
        
        # Individual Search Section - Move to top for visibility
        st.markdown("### Search Individual")
        search_col1, search_col2 = st.columns([2, 1])
        
        # Get normalized person list (cached for performance)
        try:
            all_persons = get_normalized_person_list(influencer_df)
            if not all_persons:
                all_persons = []
        except Exception as e:
            st.warning(f"Error loading person list: {e}")
            all_persons = []
        
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
        
        st.markdown("---")
        
        # Filter Options Section - Consolidate all filters here
        st.markdown("### Filter Options")
        
        # Get cluster information for filter
        try:
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
        except Exception as e:
            st.warning(f"Error loading clusters: {e}")
            clusters = []
            cluster_col = None
        
        # Filter inputs in columns
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        
        with filter_col1:
            # Authors filter
            if final_df is not None and not final_df.empty:
                author_cols = [c for c in final_df.columns if "author" in c.lower()]
                if author_cols:
                    author_col = author_cols[0]
                    authors = sorted(final_df[author_col].dropna().unique().tolist())[:50]
                    selected_authors_tab2 = st.multiselect(
                        "Select Authors",
                        authors,
                        default=[],
                        key="tab2_filter_authors"
                    )
                else:
                    selected_authors_tab2 = []
            else:
                selected_authors_tab2 = []
        
        with filter_col2:
            # Cluster filter
            if clusters:
                selected_clusters_tab2 = st.multiselect(
                    "Select Clusters",
                    clusters,
                    default=[],
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
                    default=[],
                    key="tab2_filter_sentiment_band"
                )
            else:
                selected_sentiment_bands_tab2 = []
        
        # Additional filters row - Date, Publications, Source Types
        filter_row2_col1, filter_row2_col2, filter_row2_col3 = st.columns(3)
        
        # Get available columns from final_df for filtering
        date_range_tab2 = None
        selected_publications_tab2 = []
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
        
        st.markdown("---")
        
        # Person search - normalize names for display (combine Trump variations, etc.)
        
        # Apply filters first (same logic as tab 1)
        filtered_article_indices_tab2 = None
        if final_df is not None and not final_df.empty and persons_by_row_df is not None:
            # Only copy if we need to modify (check if row_index exists first)
            needs_copy = 'row_index' not in final_df.columns
            if needs_copy:
                final_df_filtered_tab2 = final_df.reset_index().rename(columns={'index': 'row_index'})
            else:
                final_df_filtered_tab2 = final_df  # Use view, copy only when modifying
            
            # Apply article-level filters
            mask_tab2 = pd.Series(True, index=final_df_filtered_tab2.index)
            
            # Date range filter
            if date_range_tab2 and len(date_range_tab2) == 2:
                date_col = next((c for c in final_df_filtered_tab2.columns if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)
                if date_col:
                    try:
                        # Only convert dates if not already datetime
                        if final_df_filtered_tab2[date_col].dtype != 'datetime64[ns]':
                            if final_df_filtered_tab2 is final_df:
                                final_df_filtered_tab2 = final_df.copy()
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
        
        # If we have article-level filters, filter persons_by_row_df
        if filtered_article_indices_tab2 is not None:
            persons_by_row_df_filtered_tab2 = persons_by_row_df[
                persons_by_row_df['row_index'].isin(filtered_article_indices_tab2)
            ].copy() if persons_by_row_df is not None and not persons_by_row_df.empty else pd.DataFrame()
        else:
            persons_by_row_df_filtered_tab2 = persons_by_row_df.copy() if persons_by_row_df is not None else pd.DataFrame()
        
        # Show search results
        if story_person and story_person.strip():
            with st.spinner("🔍 Searching for articles..."):
                # Get row indices where this person is mentioned (from filtered data)
                if not persons_by_row_df_filtered_tab2.empty and 'persons' in persons_by_row_df_filtered_tab2.columns:
                    person_rows = persons_by_row_df_filtered_tab2[
                        persons_by_row_df_filtered_tab2['persons'].astype(str).str.contains(story_person.strip(), case=False, na=False, regex=False)
                    ]
                else:
                    person_rows = pd.DataFrame()
                
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
                        
                        # Attribution Analysis - Bin Rankings
                        st.markdown("### Attribution Analysis Rankings")
                        st.markdown("**Average bin rankings (1-5 scale) for articles mentioning this person**")
                        rank_col1, rank_col2 = st.columns(2)
                        
                        with rank_col1:
                            if 'circulation_size_bin' in person_articles.columns:
                                # Calculate average circulation bin (1-5)
                                avg_circ_bin = person_articles['circulation_size_bin'].dropna().mean()
                                if pd.notna(avg_circ_bin):
                                    st.metric("Avg Circulation Bin", f"{avg_circ_bin:.2f}")
                                    st.caption("1=Lowest, 5=Highest")
                                else:
                                    st.metric("Avg Circulation Bin", "N/A")
                            else:
                                st.metric("Avg Circulation Bin", "N/A")
                        
                        with rank_col2:
                            if 'sentiment_score_bin' in person_articles.columns:
                                # Calculate average sentiment bin (1-5)
                                avg_sent_bin = person_articles['sentiment_score_bin'].dropna().mean()
                                if pd.notna(avg_sent_bin):
                                    st.metric("Avg Sentiment Bin", f"{avg_sent_bin:.2f}")
                                    st.caption("1=Most Negative, 5=Most Positive")
                                else:
                                    st.metric("Avg Sentiment Bin", "N/A")
                            else:
                                st.metric("Avg Sentiment Bin", "N/A")
                        
                        # Emotion and Sentiment distribution charts side by side
                        chart_col1, chart_col2 = st.columns(2)
                        
                        with chart_col1:
                            # Emotion distribution chart if available (using pre-computed emotion_body)
                            if 'emotion_body' in person_articles.columns:
                                emotion_counts = person_articles['emotion_body'].value_counts().dropna()
                                if not emotion_counts.empty:
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
                        
                        with chart_col2:
                            # Sentiment distribution chart if available
                            if 'sentiment_score' in person_articles.columns:
                                # Create sentiment bands
                                person_articles_sent = person_articles.copy()
                                person_articles_sent['sentiment_band'] = pd.cut(
                                    person_articles_sent['sentiment_score'],
                                    bins=[-float('inf'), -0.1, 0.1, float('inf')],
                                    labels=['Negative', 'Neutral', 'Positive']
                                )
                                sentiment_counts = person_articles_sent['sentiment_band'].value_counts().dropna()
                                if not sentiment_counts.empty:
                                    fig_sentiment = px.bar(
                                        x=sentiment_counts.values,
                                        y=sentiment_counts.index,
                                        orientation='h',
                                        title=f'Sentiment in Articles About {story_person}',
                                        labels={'x': 'Number of Articles', 'y': 'Sentiment'},
                                        color=sentiment_counts.values,
                                        color_continuous_scale='RdYlBu'
                                    )
                                    fig_sentiment.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig_sentiment, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Network Analysis Visualization
                        st.markdown("### Network Analysis")
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
                                    
                                    # Add channel nodes (Blue Circles)
                                    for channel in top_channels[:6]:  # Limit to top 6
                                        channel_id = f"channel_{channel}"
                                        if channel_id not in seen_ids:
                                            seen_ids.add(channel_id)
                                            label = channel[:15] + "..." if len(channel) > 15 else channel
                                            nodes.append(Node(
                                                id=channel_id,
                                                label=label,
                                                size=30,
                                                color="#0066CC",  # Blue for channels (not brand color)
                                                shape="circle",
                                                title=f"{channel}<br>Type: Channel"
                                            ))
                                    
                                    # Add publication nodes (Green Triangles)
                                    for pub in top_pubs[:6]:  # Limit to top 6
                                        pub_id = f"pub_{pub}"
                                        if pub_id not in seen_ids:
                                            seen_ids.add(pub_id)
                                            label = pub[:15] + "..." if len(pub) > 15 else pub
                                            nodes.append(Node(
                                                id=pub_id,
                                                label=label,
                                                size=30,
                                                color="#228B22",  # Green for publications (not brand color)
                                                shape="triangle",
                                                title=f"{pub}<br>Type: Publication"
                                            ))
                                    
                                    # Create edges (lines showing relationships)
                                    edges = []
                                    for _, row in filtered_edges.head(20).iterrows():  # Limit to top 20 edges
                                        source_id = f"channel_{row['source_type']}"
                                        target_id = f"pub_{row['publication_name']}"
                                        if source_id in seen_ids and target_id in seen_ids:
                                            # Thick lines for stronger relationships (more articles)
                                            edge_width = max(2, min(10, int(row['count'] * 0.5)))
                                            edges.append(Edge(
                                                source=source_id,
                                                target=target_id,
                                                width=edge_width,
                                                color="#666666",  # Gray for connections
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
                        
                        # Articles/Headlines section - Table format (moved to end)
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
                        
                    else:
                        st.warning(f"No article data found for {story_person}" + (f" with keyword '{story_keyword}'" if story_keyword and story_keyword.strip() else "") + ".")
                else:
                    if person_rows.empty:
                        st.info(f"No articles found mentioning '{story_person}'. The person may not be in the persons_by_row data.")
                    else:
                        st.info(f"Found mentions of {story_person}, but unable to link to article data. Ensure final_dataset_with_attribution.parquet is available.")
        else:
            ""

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
