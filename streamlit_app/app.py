#!/usr/bin/env python3
"""
Streamlit App for Attribution and PCA Analysis
Compatible with your existing data_loaders.py / data_processors.py / charts.py.
Adds safe caps, set-based people search (no giant regex), and robust imports.
"""

from __future__ import annotations

# -------------------- MUST be first Streamlit call --------------------
import streamlit as st

# -------------------- Imports --------------------
import os
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Ensure current folder is importable (Streamlit Cloud packaging quirks)
APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

# Load the Data 
from data_loaders import (
    load_influencer_table,  
    load_final_dataset,    
    load_persons_by_row,     
)
# Note: data_processors functions are not currently used
# from data_processors import (
#     clean_bin_column,
#     extract_clean_names,
#     is_likely_person_name,
# )
from charts import create_emotion_chart
# Optional: Network analysis (restores Network tab if data present)
try:
    import network_analysis as net
except Exception:
    net = None

# Try to import helpers; if missing, define fallbacks inline
# ---- Inline helpers (no extra files needed) ----
import streamlit as st
import pandas as pd
import re

@st.cache_data(show_spinner=False)
def explode_persons(persons_by_row_df: pd.DataFrame) -> pd.DataFrame:
    """Explode comma-separated persons into long form for fast set-based matching."""
    if persons_by_row_df is None or persons_by_row_df.empty:
        return pd.DataFrame(columns=["row_index", "person", "person_lc"])
    df = persons_by_row_df[['row_index', 'persons']].dropna().copy()
    df['persons'] = df['persons'].astype(str)
    df = df.assign(person=df['persons'].str.split(',')).explode('person')
    df['person'] = df['person'].str.strip()
    df = df[df['person'] != '']
    df['person_lc'] = df['person'].str.lower()
    if df['row_index'].dtype.kind not in ('i', 'u'):
        df['row_index'] = pd.to_numeric(df['row_index'], errors='coerce').astype('Int64')
        df = df.dropna(subset=['row_index']).copy()
        df['row_index'] = df['row_index'].astype('int64')
    return df[['row_index', 'person', 'person_lc']]

def _norm_one(name: str) -> str:
    s = str(name).strip()
    low = s.lower()
    if not s:
        return s
    def any_in(parts): return any(p in low for p in parts)
    if ('kennedy' in low) and ('robert' in low or 'junior' in low):
        return "Robert Kennedy"
    if 'trump' in low and not any_in(['ivanka', 'eric ', 'tiffany', 'melania', 'meliana', 'barron', 'lady trump']):
        return "Donald Trump"
    if 'musk' in low:
        return "Elon Musk"
    clean = re.sub(r'^(dr\.?|doctor)\s*', '', low).rstrip('.,;:')
    if 'fauci' in clean or ('anthony' in clean and 'fauci' in clean):
        return "Anthony Fauci"
    if 'kamala' in low and 'harris' in low:
        return "Kamala Harris"
    if low == 'harris':
        return "Kamala Harris"
    return s

@st.cache_data(show_spinner=False)
def add_normalized_person(pbr_long: pd.DataFrame) -> pd.DataFrame:
    """Add normalized name columns to exploded persons frame."""
    if pbr_long is None or pbr_long.empty:
        return pd.DataFrame(columns=['row_index', 'person', 'person_lc', 'person_norm', 'person_norm_lc'])
    df = pbr_long.copy()
    df['person_norm'] = df['person'].map(_norm_one)
    df['person_norm_lc'] = df['person_norm'].str.lower()
    return df
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
    st.dataframe(df, use_container_width=True, height=height)

@st.cache_data(show_spinner=False)
def with_sentiment_band(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if 'sentiment_score' in out.columns and 'sentiment_band' not in out.columns:
        out['sentiment_band'] = pd.cut(
            out['sentiment_score'],
            bins=[-float('inf'), -0.1, 0.1, float('inf')],
            labels=['Negative', 'Neutral', 'Positive']
        )
    return out

# -------------------- Main app --------------------
def main():
    st.markdown('<div class="main-header" style="font-size:2rem;font-weight:700;color:#12715D;margin-bottom:0.5rem;">PolicyPath 🏛️</div>', unsafe_allow_html=True)
    st.markdown("Your indispensable guide to healthcare policy influence")

    _load_css()

    # Load base data (your existing loaders)
    with st.spinner("Loading influencer table..."):
        influencer_df = load_influencer_table()

    if influencer_df is None or (hasattr(influencer_df, "empty") and influencer_df.empty):
        st.error("⚠️ Unable to load influencer table. Check file locations.")
        st.info("""
        Expected data file locations:
        - data_storage/final_data/influencer_table.parquet (or .csv)
        - data_storage/final_data/final_dataset_with_attribution.parquet
        - data_storage/final_data/persons_by_row.parquet (or .csv)
        """)
        st.stop()

    # Light optimization: cap rows in memory-heavy views (keeps behavior but avoids OOM)
    MAX_INFLUENCER_ROWS = 200_000
    if len(influencer_df) > MAX_INFLUENCER_ROWS:
        influencer_df = influencer_df.head(MAX_INFLUENCER_ROWS)

    # Preload smaller article/person data (no params per your loaders)
    with st.spinner("Preparing people index..."):
        final_df_sample = load_final_dataset()  # you load the full file; we’ll guard/cap below
        persons_by_row_df = load_persons_by_row()

        if persons_by_row_df is None or persons_by_row_df.empty:
            pbr_long = pd.DataFrame(columns=['row_index', 'person', 'person_lc', 'person_norm', 'person_norm_lc'])
        else:
            pbr_long = add_normalized_person(explode_persons(persons_by_row_df))

        # Build dropdown list (cap to keep UI fast)
        if not pbr_long.empty:
            all_people = pbr_long['person_norm'].value_counts().head(5000).index.tolist()
        else:
            all_people = []

    tab1, tab2, tab3, tab4 = st.tabs(["PolicyPath", "People", "Topics", "Network"])

    # --------------------------------------
    # Tab 1: Pulse / Aggregates
    # --------------------------------------
    with tab1:
        st.markdown('<div class="section-header" style="font-size:1.25rem;font-weight:700;color:#142536;margin-top:1rem;">Filters</div>', unsafe_allow_html=True)

        # Cluster options
        cluster_col = None
        try:
            if 'cluster_label' in influencer_df.columns:
                clusters = sorted(influencer_df['cluster_label'].dropna().astype(str).unique().tolist())
                cluster_col = 'cluster_label'
            elif 'cluster' in influencer_df.columns:
                clusters = sorted(influencer_df['cluster'].dropna().astype(str).unique().tolist())
                cluster_col = 'cluster'
            else:
                clusters = []
        except Exception:
            clusters = []

        c1, c2, c3 = st.columns(3)
        with c1:
            selected_persons = st.multiselect("Select Individuals", all_people, default=[], key="tab1_select_persons")
        with c2:
            selected_clusters = st.multiselect("Select Clusters", clusters, default=[], key="tab1_select_clusters") if clusters else []
        with c3:
            tmp = with_sentiment_band(influencer_df)
            bands = sorted(tmp['sentiment_band'].dropna().unique().tolist()) if 'sentiment_band' in tmp.columns else []
            selected_sentiment_bands = st.multiselect("Sentiment Band", bands, default=[], key="tab1_sentiment_band")

        # Apply filters (people, cluster, band)
        influencer_view = influencer_df

        # Filter by persons (set-join, no regex)
        if selected_persons:
            needles = {p.strip().lower() for p in selected_persons if p.strip()}
            if not pbr_long.empty:
                rows_for_people = pbr_long[pbr_long['person_norm'].str.lower().isin(needles)]['row_index'].unique()
                # If influencer_df had a direct person key, we'd use that. Here, we conservatively filter by text incl.
                # If your influencer_df has a better join key, replace this contains with a join.
                keys = {p for p in selected_persons}
                influencer_view = influencer_view[influencer_view['person_list'].astype(str).str.lower().apply(
                    lambda s: any(k.lower() in s for k in keys)
                )]

        if selected_clusters and cluster_col and cluster_col in influencer_view.columns:
            influencer_view = influencer_view[influencer_view[cluster_col].astype(str).isin(selected_clusters)]

        if selected_sentiment_bands:
            influencer_view = with_sentiment_band(influencer_view)
            if 'sentiment_band' in influencer_view.columns:
                influencer_view = influencer_view[influencer_view['sentiment_band'].isin(selected_sentiment_bands)]

        # Summary metrics
        st.markdown('<div class="section-header" style="font-size:1.25rem;font-weight:700;color:#142536;margin-top:1rem;">Data Summary</div>', unsafe_allow_html=True)
        if influencer_view is not None and not influencer_view.empty:
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Total Individuals", f"{len(influencer_view):,}")
            with m2:
                total_mentions = influencer_view.get('mention_count', pd.Series([0]*len(influencer_view))).sum()
                st.metric("Total Mentions", f"{int(total_mentions):,}")
            with m3:
                avg_sentiment = influencer_view.get('sentiment_score', pd.Series(dtype=float)).mean()
                st.metric("Avg Sentiment Score", f"{(avg_sentiment if pd.notna(avg_sentiment) else 0):.2f}")
            with m4:
                avg_circ = influencer_view.get('circulation_size', pd.Series(dtype=float)).mean()
                st.metric("Avg Circulation", f"{(avg_circ if pd.notna(avg_circ) else 0):,.0f}")

        # Grouped charts (top-N)
        if influencer_view is not None and not influencer_view.empty:
            group_by_options = []
            if cluster_col and cluster_col in influencer_view.columns:
                group_by_options.append(('Cluster', cluster_col))
            if 'sentiment_score' in influencer_view.columns:
                influencer_view = with_sentiment_band(influencer_view)
                group_by_options.append(('Sentiment Band', 'sentiment_band'))

            if group_by_options:
                dim = st.selectbox("Group charts by", [label for label, _ in group_by_options], index=0)
                dim_col = next(col for label, col in group_by_options if label == dim)

                agg_dict = {'mention_count': 'sum'}
                if 'sentiment_score' in influencer_view.columns:
                    agg_dict['sentiment_score'] = 'mean'
                if 'circulation_size' in influencer_view.columns:
                    agg_dict['circulation_size'] = 'mean'

                agg = (
                    influencer_view
                    .groupby(dim_col)
                    .agg(agg_dict)
                    .reset_index()
                    .rename(columns={dim_col: "dim"})
                )
                agg = agg[agg["dim"].notna()]

                top_n = st.slider("Top N", 5, 30, 10, 1)

                c1, c2 = st.columns(2)
                with c1:
                    if cluster_col and 'sentiment_score' in influencer_view.columns:
                        sentiment_by_cluster = (
                            influencer_view.groupby(cluster_col)['sentiment_score']
                            .mean()
                            .reset_index()
                            .sort_values('sentiment_score', ascending=False)
                            .head(top_n)
                        )
                        fig_sentiment = go.Figure(data=[
                            go.Bar(x=sentiment_by_cluster[cluster_col], y=sentiment_by_cluster['sentiment_score'])
                        ])
                        fig_sentiment.update_layout(
                            title="Sentiment Distribution by Cluster (Top)",
                            xaxis_title="Cluster",
                            yaxis_title="Average Sentiment",
                            height=300,
                            showlegend=False
                        )
                        st.plotly_chart(fig_sentiment, use_container_width=True)
                    else:
                        st.info("No cluster/sentiment data available")

                with c2:
                    if cluster_col and 'circulation_size' in influencer_view.columns:
                        circ_by_cluster = (
                            influencer_view.groupby(cluster_col)['circulation_size']
                            .mean()
                            .reset_index()
                            .sort_values('circulation_size', ascending=False)
                            .head(top_n)
                        )
                        fig_circ = go.Figure(data=[
                            go.Bar(x=circ_by_cluster[cluster_col], y=circ_by_cluster['circulation_size'])
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
                # Emotion chart (uses your charts.create_emotion_chart)
                st.markdown('<div class="section-header" style="font-size:1.25rem;font-weight:700;color:#142536;margin-top:1rem;">Top Individuals by Emotion</div>', unsafe_allow_html=True)
                n_top = st.slider("Number of top individuals to show", 10, 50, 20, key="emotion_slider_tab1")

                # Use consistent variable names
                final_df_for_emotions = final_df_sample
                persons_by_row_for_emotions = persons_by_row_df

                if final_df_for_emotions is None or persons_by_row_for_emotions is None:
                    st.info("Emotion analysis requires final dataset and persons_by_row.")
                elif 'emotion_body' not in final_df_for_emotions.columns:
                    st.info("Emotion data not available in final dataset.")
                else:
                    emotion_chart = create_emotion_chart(
                        influencer_view,
                        final_df=final_df_for_emotions,
                        persons_by_row_df=persons_by_row_for_emotions,
                        n=n_top,
                        selected_persons=selected_persons if selected_persons else None
                    )
                    if emotion_chart:
                        st.plotly_chart(emotion_chart, use_container_width=True)
                    else:
                        st.info("Emotion data not available for the current selection.")
    # --------------------------------------
    # Tab 2: People — individual exploration
    # --------------------------------------
    with tab2:
        st.markdown('<div class="section-header" style="font-size:1.25rem;font-weight:700;color:#142536;margin-top:1rem;">People</div>', unsafe_allow_html=True)
        st.markdown("### Search Individual")

        # Build dropdown from normalized names from earlier
        s1, s2 = st.columns([2, 1])
        with s1:
            story_person = st.selectbox(
                "Select an Individual",
                options=[""] + all_people,
                index=0,
                help="Fast, case-insensitive, normalized people search",
                key="tab2_story_person"
            )
        with s2:
            story_keyword = st.text_input(
                "Optional Keyword Filter",
                placeholder="e.g., vaccine, policy...",
                key="tab2_story_keyword"
            )

        st.markdown("---")
        st.markdown("### Filter Options")

        # Cluster/sentiment
        cluster_col2 = None
        try:
            if 'cluster_label' in influencer_df.columns:
                clusters2 = sorted(influencer_df['cluster_label'].dropna().astype(str).unique().tolist())
                cluster_col2 = 'cluster_label'
            elif 'cluster' in influencer_df.columns:
                clusters2 = sorted(influencer_df['cluster'].dropna().astype(str).unique().tolist())
                cluster_col2 = 'cluster'
            else:
                clusters2 = []
        except Exception:
            clusters2 = []

        f1, f2, f3 = st.columns(3)
        with f1:
            # Authors from final_df_sample (if available)
            if final_df_sample is not None and not final_df_sample.empty and 'author_name' in final_df_sample.columns:
                authors = sorted(final_df_sample['author_name'].dropna().unique().tolist())[:50]
                selected_authors_tab2 = st.multiselect("Select Authors", authors, default=[], key="tab2_select_authors")
            else:
                selected_authors_tab2 = []

        with f2:
            selected_clusters_tab2 = st.multiselect("Select Clusters", clusters2, default=[], key="tab2_select_clusters") if clusters2 else []

        with f3:
            tmp2 = with_sentiment_band(influencer_df)
            bands2 = sorted(tmp2['sentiment_band'].dropna().unique().tolist()) if 'sentiment_band' in tmp2.columns else []
            selected_sentiment_bands_tab2 = st.multiselect("Sentiment Band", bands2, default=[], key="tab2_sentiment_band")

        fr2c1, fr2c2, fr2c3 = st.columns(3)
        # Date/publication/source filters are not pushed down (your loader has no args), so we skip widgets here
        # If you later add loader params, we can wire them up.

        st.markdown("---")

        if story_person and story_person.strip():
            with st.spinner("🔍 Searching for articles..."):
                if final_df_sample is None or final_df_sample.empty or persons_by_row_df is None or persons_by_row_df.empty:
                    st.info("No article/person data available.")
                else:
                    # row_index ensure
                    final_df2 = final_df_sample.copy()
                    if 'row_index' not in final_df2.columns:
                        final_df2 = final_df2.reset_index().rename(columns={'index': 'row_index'})

                    # Person match via normalized set join
                    pbr2 = add_normalized_person(explode_persons(persons_by_row_df))
                    needle = story_person.strip().lower()
                    rows = pbr2.loc[pbr2['person_norm'].str.lower() == needle, 'row_index'].unique()

                    person_articles = final_df2[final_df2['row_index'].isin(rows)].copy()

                    # Optional keyword filter
                    if story_keyword and story_keyword.strip():
                        kw = story_keyword.strip().lower()
                        mask = pd.Series(False, index=person_articles.index)
                        for col in ['headline', 'article_body']:
                            if col in person_articles.columns:
                                mask = mask | person_articles[col].astype(str).str.lower().str.contains(kw, regex=False)
                        person_articles = person_articles[mask]

                    if selected_clusters_tab2 and cluster_col2 and cluster_col2 in influencer_df.columns:
                        # If your articles also carry cluster linkage, you can filter here.
                        pass

                    if selected_sentiment_bands_tab2:
                        person_articles = with_sentiment_band(person_articles)
                        if 'sentiment_band' in person_articles.columns:
                            person_articles = person_articles[person_articles['sentiment_band'].isin(selected_sentiment_bands_tab2)]

                    if not person_articles.empty:
                        # Summary metrics
                        st.markdown("### Summary")
                        m1, m2, m3, m4 = st.columns(4)
                        with m1:
                            st.metric("Total Articles", len(person_articles))
                        with m2:
                            avg_sent = person_articles.get('sentiment_score', pd.Series(dtype=float)).mean()
                            st.metric("Avg Sentiment", f"{(avg_sent if pd.notna(avg_sent) else 0):.2f}")
                        with m3:
                            total_reach = person_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            st.metric("Total Reach", f"{int(total_reach) if pd.notna(total_reach) else 0:,}")
                        with m4:
                            uniq_pubs = person_articles.get('publication_name', pd.Series(dtype=str)).nunique()
                            st.metric("Unique Publishers", int(uniq_pubs) if pd.notna(uniq_pubs) else 0)

                        # Distributions
                        c1, c2 = st.columns(2)
                        with c1:
                            if 'emotion_body' in person_articles.columns:
                                counts = person_articles['emotion_body'].value_counts().dropna().head(12)
                                if not counts.empty:
                                    fig = px.bar(
                                        x=counts.values, y=counts.index,
                                        orientation='h',
                                        title=f'Emotions in Articles About {story_person}',
                                        labels={'x': 'Number of Articles', 'y': 'Emotion'}
                                    )
                                    fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig, use_container_width=True)
                        with c2:
                            if 'sentiment_score' in person_articles.columns:
                                tmp = with_sentiment_band(person_articles)
                                sc = tmp['sentiment_band'].value_counts().dropna()
                                if not sc.empty:
                                    fig = px.bar(
                                        x=sc.values, y=sc.index,
                                        orientation='h',
                                        title=f'Sentiment in Articles About {story_person}',
                                        labels={'x': 'Number of Articles', 'y': 'Sentiment'}
                                    )
                                    fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig, use_container_width=True)

                        st.markdown("---")
                        st.markdown("### Articles Mentioning This Person")

                        display_cols = [c for c in [
                            'headline', 'publication_name', 'author_name',
                            'sentiment_score', 'emotion_body', 'circulation_size',
                            'source_type', 'article_url'
                        ] if c in person_articles.columns]

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
                        st.caption(f"Exporting {len(person_articles):,} articles with current filters.")
                    else:
                        st.warning(f"No article data found for {story_person}" + (f" with keyword '{story_keyword}'" if story_keyword and story_keyword.strip() else "") + ".")

    # --------------------------------------
    # Tab 3: Topics — search by tag_name
    # --------------------------------------
    with tab3:
        st.markdown('<div class="section-header" style="font-size:1.25rem;font-weight:700;color:#142536;margin-top:1rem;">Topics</div>', unsafe_allow_html=True)
        if final_df_sample is None or final_df_sample.empty or persons_by_row_df is None or persons_by_row_df.empty:
            st.info("Final dataset and persons_by_row are required for topic search.")
        else:
            # Topic list from tag_name (cap to keep UI responsive)
            tag_counts = final_df_sample['tag_name'].dropna().astype(str).value_counts()
            tag_options = tag_counts.head(500).index.tolist()
            col_t1, col_t2 = st.columns([3, 1])
            with col_t1:
                selected_tag = st.selectbox("Select Topic (tag_name)", options=[""] + tag_options, index=0, key="topic_select")
            with col_t2:
                top_n_topic = st.slider("Top N People", 5, 50, 20, key="topic_topn")

            if selected_tag:
                # Filter article and person mappings to this topic
                final_df_topic = final_df_sample[final_df_sample['tag_name'].astype(str) == selected_tag].copy()
                pbr_topic = persons_by_row_df[persons_by_row_df['tag_name'].astype(str) == selected_tag].copy()

                # Explode persons and normalize names
                pbr_long_topic = add_normalized_person(explode_persons(pbr_topic))
                if pbr_long_topic.empty:
                    st.info("No people detected for this topic.")
                else:
                    # Top people by mention count within the topic
                    counts = pbr_long_topic['person_norm'].value_counts()
                    top_people = counts.head(top_n_topic)
                    influencer_topic = pd.DataFrame({
                        'person_list': top_people.index.tolist(),
                        'mention_count': top_people.values.tolist()
                    })

                    # Prepare sentiment aggregation for re-use
                    final_df_topic2 = final_df_topic.copy()
                    if 'row_index' not in final_df_topic2.columns:
                        final_df_topic2 = final_df_topic2.reset_index().rename(columns={'index': 'row_index'})
                    join_df = pbr_long_topic[['row_index', 'person_norm']].merge(
                        final_df_topic2[['row_index', 'sentiment_score']],
                        on='row_index', how='inner'
                    )
                    sent_agg = (
                        join_df.groupby('person_norm')['sentiment_score']
                        .mean().reset_index()
                    )
                    sent_agg = sent_agg[sent_agg['person_norm'].isin(influencer_topic['person_list'])]
                    sent_agg = sent_agg.sort_values('sentiment_score', ascending=False)

                    # Sub-tabs for a cleaner layout
                    t_overview, t_sentiment, t_emotion = st.tabs(["Overview", "Sentiment", "Emotion"])

                    with t_overview:
                        with st.container(border=True):
                            st.markdown(f"#### Top People in “{selected_tag}”")
                            fig_top = px.bar(
                                x=influencer_topic['mention_count'],
                                y=influencer_topic['person_list'],
                                orientation='h',
                                labels={'x': 'Mentions', 'y': 'Person'},
                                template='simple_white'
                            )
                            fig_top.update_layout(
                                margin=dict(l=10, r=10, t=10, b=10),
                                height=max(320, len(influencer_topic) * 22),
                                yaxis={'categoryorder': 'total ascending'}
                            )
                            st.plotly_chart(fig_top, use_container_width=True)

                    with t_sentiment:
                        with st.container(border=True):
                            st.markdown(f"#### Average Sentiment by Person")
                            if not sent_agg.empty:
                                # Diverging color by sign
                                colors = sent_agg['sentiment_score'].apply(lambda v: "#4AB48E" if pd.notna(v) and v >= 0 else "#D94841")
                                fig_sent = go.Figure(data=[
                                    go.Bar(
                                        x=sent_agg['sentiment_score'],
                                        y=sent_agg['person_norm'],
                                        orientation='h',
                                        marker_color=colors
                                    )
                                ])
                                fig_sent.update_layout(
                                    template='simple_white',
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=max(320, len(sent_agg) * 22),
                                    xaxis=dict(zeroline=True, zerolinecolor="#cccccc", title="Avg Sentiment"),
                                    yaxis=dict(title="Person", categoryorder='total ascending')
                                )
                                st.plotly_chart(fig_sent, use_container_width=True)
                            else:
                                st.info("No sentiment data available for the selected topic.")

                    with t_emotion:
                        with st.container(border=True):
                            st.markdown("#### Emotion Distribution (Top People)")
                            # Reuse existing emotion chart with filtered datasets and selected persons
                            if 'emotion_body' in final_df_topic.columns:
                                emotion_chart = create_emotion_chart(
                                    influencer_topic,
                                    final_df=final_df_topic,
                                    persons_by_row_df=pbr_topic,
                                    n=len(influencer_topic),
                                    selected_persons=influencer_topic['person_list'].tolist()
                                )
                                if emotion_chart:
                                    emotion_chart.update_layout(
                                        template='simple_white',
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        height=max(360, len(influencer_topic) * 18),
                                    )
                                    st.plotly_chart(emotion_chart, use_container_width=True)
                                else:
                                    st.info("Emotion data not available for this topic.")
                            else:
                                st.info("Final dataset lacks emotion_body for this topic.")

    # --------------------------------------
    # Tab 4: Network — restore visualization
    # --------------------------------------
    with tab4:
        st.markdown('<div class="section-header" style="font-size:1.25rem;font-weight:700;color:#142536;margin-top:1rem;">Network</div>', unsafe_allow_html=True)
        if net is None:
            st.info("Network module not available.")
        else:
            try:
                data = net.get_network_data()
                edges = data.get('publisher_term_edges')
                if edges is not None and not edges.empty:
                    G = net.build_content_graph(edges)
                    node2c = net.community_map_content(G)
                    fig_net = net.create_interactive_network_visualization(
                        G, node2c, edges, title="Content Network: Publishers ↔ Terms"
                    )
                    if fig_net:
                        st.plotly_chart(fig_net, use_container_width=True)
                    else:
                        st.info("Unable to render network figure.")
                else:
                    st.info("No precomputed network data found (publisher_term_edges.csv).")
            except Exception:
                st.info("Network data could not be loaded.")

    render_footer()

def render_footer():
    st.markdown("---")
    st.markdown(
        """
        <div style="font-size: 0.8em; text-align: left; margin-top: 2rem;">
            Built by Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammad Waqas, Mark Saba, Posy Olivetti
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()