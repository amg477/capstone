#!/usr/bin/env python3
"""
Streamlit App for Attribution and PCA Analysis
Refactor: shared DuckDB connection, set-based person search, filter pushdown,
hard caps on UI rendering, and safer caching.
"""

from __future__ import annotations

# -------------------- MUST be first Streamlit call --------------------
import streamlit as st

# -------------------- Imports --------------------
import os
import re
from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Your modules
from data_loaders import (
    load_influencer_table,
    load_final_dataset,
    load_persons_by_row,
)
from data_processors import (
    clean_bin_column,
    extract_clean_names,         # (unused here but you may need elsewhere)
    is_likely_person_name,       # (unused here but kept for compatibility)
)
from charts import (
    create_emotion_chart
)
from network_analysis import (
    get_network_data,                    # (unused in this page)
    build_content_network_edges,
    build_content_graph,
    community_map_content,
    create_interactive_network_visualization
)

# New helpers
from helpers_people import explode_persons
from people_normalize import add_normalized_person

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
    st.warning("CSS file not found. App will run without custom styling.")

# -------------------- Small UI helpers --------------------
def shorten(label: str, max_len: int = 28) -> str:
    s = str(label).strip()
    if len(s) <= max_len:
        return s
    if ' ' in s:
        words = s.split()
        result = ""
        for word in words:
            nxt = (result + " " + word) if result else word
            if len(nxt) <= max_len - 1:
                result = nxt
            else:
                break
        if result:
            return result + "…"
    return s[:max_len-1] + "…"

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
    st.markdown('<div class="main-header">PolicyPath 🏛️</div>', unsafe_allow_html=True)
    st.markdown("Your indispensable guide to healthcare policy influence")

    _load_css()

    # Load influencer table (shared cache; limit protects memory)
    with st.spinner("Loading influencer table..."):
        influencer_df = load_influencer_table(limit=200_000)

    if influencer_df is None or influencer_df.empty:
        st.error("⚠️ Unable to load influencer table. Please ensure the data files are in the correct location.")
        st.info("""
        Expected data file locations:
        - data_storage/final_data/influencer_table.parquet (or .csv)
        - data_storage/final_data/final_dataset_with_attribution.parquet
        - data_storage/final_data/persons_by_row.parquet (or .csv)
        """)
        st.stop()

    # Optional bin cleaning
    try:
        if 'circulation_size_bin' in influencer_df.columns:
            influencer_df = clean_bin_column(influencer_df, 'circulation_size_bin')
        if 'sentiment_score_bin' in influencer_df.columns:
            influencer_df = clean_bin_column(influencer_df, 'sentiment_score_bin')
    except Exception as e:
        st.warning(f"Error cleaning bin columns: {e}")

    tab1, tab2 = st.tabs(["PolicyPath", "People"])

    # --------------------------------------
    # Tab 1: Pulse / Aggregates
    # --------------------------------------
    with tab1:
        # Filters Row 1 (lightweight)
        st.markdown('<div class="section-header">Filters</div>', unsafe_allow_html=True)

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

        # Build people list from persons_by_row (safer than scanning influencer_df text)
        with st.spinner("Preparing people index..."):
            # Load a modest slice of final + pbr to derive person list
            final_df_sample = load_final_dataset(limit=50_000)
            persons_by_row_df = load_persons_by_row(limit=200_000)
            pbr_long = explode_persons(persons_by_row_df)
            pbr_long = add_normalized_person(pbr_long)
            # Top names for dropdown
            top_people = (
                pbr_long['person_norm']
                .value_counts()
                .head(5000)
                .index
                .tolist()
            )

        c1, c2, c3 = st.columns(3)
        with c1:
            selected_persons = st.multiselect("Select Individuals", top_people, default=[])
        with c2:
            selected_clusters = st.multiselect("Select Clusters", clusters, default=[]) if clusters else []
        with c3:
            # Always compute band on the fly (cheap)
            tmp = with_sentiment_band(influencer_df)
            bands = sorted(tmp['sentiment_band'].dropna().unique().tolist()) if 'sentiment_band' in tmp.columns else []
            selected_sentiment_bands = st.multiselect("Sentiment Band", bands, default=[])

        # Filters Row 2 (pushed down to DuckDB for final_df; light args)
        r2c1, r2c2, r2c3 = st.columns(3)
        date_range = None
        selected_publications = []
        selected_source_types = []
        selected_authors = []

        # Peek columns from a small final_df sample to build widgets
        final_cols = final_df_sample.columns.tolist() if final_df_sample is not None else []
        date_col_guess = next((c for c in final_cols if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)

        with r2c1:
            if final_df_sample is not None and not final_df_sample.empty and date_col_guess:
                try:
                    temp = final_df_sample.copy()
                    temp[date_col_guess] = pd.to_datetime(temp[date_col_guess], errors="coerce")
                    min_d, max_d = temp[date_col_guess].min(), temp[date_col_guess].max()
                    if pd.notna(min_d) and pd.notna(max_d):
                        date_range = st.date_input(
                            "Date Range",
                            value=(min_d.date(), max_d.date()),
                            min_value=min_d.date(),
                            max_value=max_d.date(),
                            key="filter_date_range_tab1"
                        )
                except Exception:
                    date_range = None

        with r2c2:
            if 'publication_name' in final_cols:
                pubs = sorted(final_df_sample['publication_name'].dropna().unique().tolist())[:50]
                selected_publications = st.multiselect("Publications", pubs, default=[])

        with r2c3:
            if 'source_type' in final_cols:
                srcs = sorted(final_df_sample['source_type'].dropna().unique().tolist())[:20]
                selected_source_types = st.multiselect("Source Types", srcs, default=[])

        # Build article-level filter via DuckDB (pushdown), then map to people
        with st.spinner("Applying filters..."):
            date_min = str(date_range[0]) if date_range else None
            date_max = str(date_range[1]) if date_range else None
            final_df = load_final_dataset(
                date_min=date_min,
                date_max=date_max,
                publications=selected_publications or None,
                source_types=selected_source_types or None,
                authors=None,
                limit=150_000
            )
            if final_df is not None and not final_df.empty:
                # Ensure row_index
                if 'row_index' not in final_df.columns:
                    final_df = final_df.reset_index().rename(columns={'index': 'row_index'})
                filtered_article_indices = set(final_df['row_index'].tolist())
            else:
                filtered_article_indices = None

            influencer_view = influencer_df

            # Filter by persons if selected (join on exploded people)
            if selected_persons:
                needles = {p.strip().lower() for p in selected_persons if p.strip()}
                pbr_filtered = pbr_long[pbr_long['person_norm'].str.lower().isin(needles)]
                if filtered_article_indices is not None:
                    pbr_filtered = pbr_filtered[pbr_filtered['row_index'].isin(filtered_article_indices)]
                # Keep influencer rows whose person_list contains any selected normalized person (string match simplified)
                # If your influencer_df has a person key, better to join on a key instead of contains:
                keys = set(pbr_filtered['person_norm'].unique().tolist())
                influencer_view = influencer_view[
                    influencer_view['person_list'].astype(str).str.lower().apply(
                        lambda s: any(k.lower() in s for k in keys)
                    )
                ]

            # Cluster filter
            if selected_clusters and cluster_col and cluster_col in influencer_view.columns:
                influencer_view = influencer_view[influencer_view[cluster_col].astype(str).isin(selected_clusters)]

            # Sentiment band filter
            if selected_sentiment_bands:
                influencer_view = with_sentiment_band(influencer_view)
                if 'sentiment_band' in influencer_view.columns:
                    influencer_view = influencer_view[influencer_view['sentiment_band'].isin(selected_sentiment_bands)]

        # Summary metrics
        st.markdown('<div class="section-header">Data Summary</div>', unsafe_allow_html=True)
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
                st.markdown('<div class="section-header">Top Individuals by Emotion</div>', unsafe_allow_html=True)
                n_top = st.slider("Number of top individuals to show", 10, 50, 20, key="emotion_slider_tab1")

                # Final data & persons_by_row for emotions
                # (Reload without low caps if needed — keep a cap to avoid OOM)
                final_df_for_emotions = load_final_dataset(
                    date_min=date_min, date_max=date_max,
                    publications=selected_publications or None,
                    source_types=selected_source_types or None,
                    authors=None,
                    limit=150_000
                )
                persons_by_row_for_emotions = load_persons_by_row(limit=300_000)

                if final_df_for_emotions is None or persons_by_row_for_emotions is None:
                    st.info("Emotion analysis requires final dataset and persons_by_row.")
                elif 'emotion_body' not in final_df_for_emotions.columns:
                    st.info("Emotion data not available in final dataset.")
                else:
                    try:
                        emotion_chart = create_emotion_chart(
                            influencer_view,
                            final_df=final_df_for_emions,
                            persons_by_row_df=persons_by_row_for_emotions,
                            n=n_top,
                            selected_persons=selected_persons if selected_persons else None
                        )
                    except NameError:
                        # fix typo if any
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
        st.markdown('<div class="section-header">People</div>', unsafe_allow_html=True)
        st.markdown("### Search Individual")

        # Reuse pbr_long from above block
        # If you want to isolate caches across tabs, recompute here:
        persons_by_row_df2 = load_persons_by_row(limit=300_000)
        pbr_long2 = add_normalized_person(explode_persons(persons_by_row_df2))

        # Build dropdown from normalized names
        all_people = pbr_long2['person_norm'].value_counts().head(5000).index.tolist()

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
            # Authors (from a small sample for options)
            final_sample2 = load_final_dataset(limit=50_000)
            if final_sample2 is not None and not final_sample2.empty and 'author_name' in final_sample2.columns:
                authors = sorted(final_sample2['author_name'].dropna().unique().tolist())[:50]
                selected_authors_tab2 = st.multiselect("Select Authors", authors, default=[])
            else:
                selected_authors_tab2 = []

        with f2:
            selected_clusters_tab2 = st.multiselect("Select Clusters", clusters2, default=[]) if clusters2 else []

        with f3:
            tmp2 = with_sentiment_band(influencer_df)
            bands2 = sorted(tmp2['sentiment_band'].dropna().unique().tolist()) if 'sentiment_band' in tmp2.columns else []
            selected_sentiment_bands_tab2 = st.multiselect("Sentiment Band", bands2, default=[])

        fr2c1, fr2c2, fr2c3 = st.columns(3)
        date_range_tab2 = None
        selected_publications_tab2 = []
        selected_source_types_tab2 = []

        final_cols2 = final_sample2.columns.tolist() if final_sample2 is not None else []
        date_col_guess2 = next((c for c in final_cols2 if any(k in c.lower() for k in ["date", "time", "ts", "published"])), None)

        with fr2c1:
            if final_sample2 is not None and not final_sample2.empty and date_col_guess2:
                try:
                    tmpd = final_sample2.copy()
                    tmpd[date_col_guess2] = pd.to_datetime(tmpd[date_col_guess2], errors="coerce")
                    min_d2, max_d2 = tmpd[date_col_guess2].min(), tmpd[date_col_guess2].max()
                    if pd.notna(min_d2) and pd.notna(max_d2):
                        date_range_tab2 = st.date_input(
                            "Date Range",
                            value=(min_d2.date(), max_d2.date()),
                            min_value=min_d2.date(),
                            max_value=max_d2.date(),
                            key="tab2_filter_date_range"
                        )
                except Exception:
                    date_range_tab2 = None

        with fr2c2:
            if 'publication_name' in final_cols2:
                pubs2 = sorted(final_sample2['publication_name'].dropna().unique().tolist())[:50]
                selected_publications_tab2 = st.multiselect("Publications", pubs2, default=[])

        with fr2c3:
            if 'source_type' in final_cols2:
                srcs2 = sorted(final_sample2['source_type'].dropna().unique().tolist())[:20]
                selected_source_types_tab2 = st.multiselect("Source Types", srcs2, default=[])

        st.markdown("---")

        # Apply article-level filters via DuckDB
        date_min2 = str(date_range_tab2[0]) if date_range_tab2 else None
        date_max2 = str(date_range_tab2[1]) if date_range_tab2 else None
        final_df2 = load_final_dataset(
            date_min=date_min2, date_max=date_max2,
            publications=selected_publications_tab2 or None,
            source_types=selected_source_types_tab2 or None,
            authors=selected_authors_tab2 or None,
            limit=150_000
        )

        if story_person and story_person.strip():
            with st.spinner("🔍 Searching for articles..."):
                # Find row_index IDs for this person
                needle = story_person.strip().lower()
                rows = pbr_long2.loc[pbr_long2['person_norm'].str.lower() == needle, 'row_index'].unique()

                if final_df2 is None or final_df2.empty:
                    st.info("No article data available for the current filters.")
                else:
                    if 'row_index' not in final_df2.columns:
                        final_df2 = final_df2.reset_index().rename(columns={'index': 'row_index'})
                    person_articles = final_df2[final_df2['row_index'].isin(rows)].copy()

                    # Optional keyword filter
                    if story_keyword and story_keyword.strip():
                        kw = story_keyword.strip().lower()
                        mask = pd.Series(False, index=person_articles.index)
                        for col in ['headline', 'article_body']:
                            if col in person_articles.columns:
                                mask = mask | person_articles[col].astype(str).str.lower().str.contains(kw, regex=False)
                        person_articles = person_articles[mask]

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

                        # Sort by circulation descending if present
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
                        st.caption(f"Exporting {len(person_articles):,} articles with all applied filters.")
                    else:
                        st.warning(f"No article data found for {story_person}" + (f" with keyword '{story_keyword}'" if story_keyword and story_keyword.strip() else "") + ".")
        # else: show nothing for empty selection

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