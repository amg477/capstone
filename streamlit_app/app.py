#!/usr/bin/env python3
"""
Streamlit App for Attribution and PCA Analysis
Compatible with your existing data_loaders.py / charts.py.
Data cleaning is handled offline in clean_people_names.py.
"""

# Import Packages 
from __future__ import annotations
import streamlit as st
import os
from pathlib import Path
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from charts import create_emotion_chart
import networkx as nx
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

@st.cache_data(show_spinner=False, ttl=3600)
def get_all_people_list(pbr_long: pd.DataFrame) -> list:
    """Get cached list of all people"""
    if pbr_long is None or pbr_long.empty:
        return []
    return pbr_long['person'].value_counts().head(5000).index.tolist()


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


def build_person_network_graph(person_articles: pd.DataFrame, person_name: str) -> Optional[go.Figure]:
    """Create an ego network for the selected person across categorical attributes."""
    if person_articles is None or person_articles.empty:
        return None

    center = person_name.strip()
    if not center:
        return None

    G = nx.Graph()
    G.add_node(center, ntype="person", size=55)

    categorical_cols = [
        "publication_name",
        "source_name",
        "source_type",
        "channel_name",
        "author_name",
        "tag_name",
        "sentiment_band",
    ]

    for col in categorical_cols:
        if col not in person_articles.columns:
            continue

        counts = (
            person_articles[col]
            .dropna()
            .astype(str)
            .str.strip()
            .replace("", pd.NA)
            .dropna()
            .value_counts()
        )
        if counts.empty:
            continue
        
        # Filter to top 20 if more than 20 options
        if len(counts) > 20:
            counts = counts.nlargest(20)

        for value, weight in counts.items():
            node_id = f"{col}:{value}"
            G.add_node(node_id, ntype=col, size=18 + min(int(weight), 20))
            G.add_edge(center, node_id, weight=int(weight))

    if G.number_of_edges() == 0:
        return None

    pos = nx.spring_layout(G, k=0.7, seed=42)
    pos[center] = (0.0, 0.0)

    type_palette = {
        "person": "#12715D",
        "publication_name": "#4AB48E",
        "source_name": "#2A9D8F",
        "source_type": "#2A9D8F",
        "channel_name": "#D4A115",
        "author_name": "#D94841",
        "tag_name": "#9467BD",
        "sentiment_band": "#8C564B",
    }

    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(120,120,120,0.4)"),
        hoverinfo="none",
        showlegend=False,
    )

    node_traces = []
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get("ntype", "unknown")
        node_types.setdefault(ntype, []).append(node)

    for ntype, nodes in node_types.items():
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        sizes = [G.nodes[n].get("size", 20) for n in nodes]
        color = type_palette.get(ntype, "#808080")

        hover_text = []
        for n in nodes:
            if n == center:
                hover_text.append(f"<b>{center}</b><br>Type: Person")
            else:
                label = n.split(":", 1)[-1]
                weight = int(G[center][n].get("weight", 0)) if center in G[n] else 0
                hover_text.append(f"<b>{label}</b><br>Type: {ntype.replace('_', ' ').title()}<br>Articles: {weight}")

        display_text = []
        for n in nodes:
            if n == center:
                display_text.append(center)
            else:
                label = n.split(":", 1)[-1]
                display_text.append(label if len(label) <= 14 else f"{label[:11]}…")

        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.85,
                ),
                text=display_text,
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                hovertext=hover_text,
                hoverinfo="text",
                name=ntype.replace("_", " ").title(),
                showlegend=True,
            )
        )

    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(text=f"Network View: {center}", x=0.5, font=dict(size=16)),
        showlegend=True,
        hovermode="closest",
        dragmode="pan",
        margin=dict(b=30, l=10, r=10, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=560,
    )
    return fig


def build_topic_people_network_graph(
    topic_name: str,
    final_df_topic: pd.DataFrame,
    pbr_long_topic: pd.DataFrame,
    max_people: int = 30
) -> Optional[go.Figure]:
    """Create a bipartite network graph showing connections between a topic and people."""
    if final_df_topic is None or final_df_topic.empty:
        return None
    
    if pbr_long_topic is None or pbr_long_topic.empty:
        return None
    
    topic_center = topic_name.strip()
    if not topic_center:
        return None
    
    # Ensure row_index exists
    if 'row_index' not in final_df_topic.columns:
        final_df_topic = final_df_topic.reset_index().rename(columns={'index': 'row_index'})
    
    # Convert row_index to numeric for merging
    final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')
    pbr_long_topic['row_index'] = pd.to_numeric(pbr_long_topic['row_index'], errors='coerce')
    
    # Count articles per person for this topic
    person_counts = pbr_long_topic.groupby('person')['row_index'].nunique().reset_index(name='article_count')
    person_counts = person_counts.sort_values('article_count', ascending=False).head(max_people)
    
    if person_counts.empty:
        return None
    
    # Build the network graph
    G = nx.Graph()
    
    # Add topic as center node
    G.add_node(topic_center, ntype="topic", size=60)
    
    # Add people nodes and edges
    for _, row in person_counts.iterrows():
        person = row['person']
        weight = int(row['article_count'])
        
        # Add person node
        G.add_node(person, ntype="person", size=20 + min(weight, 30))
        
        # Add edge between topic and person
        G.add_edge(topic_center, person, weight=weight)
    
    if G.number_of_edges() == 0:
        return None
    
    # Use bipartite layout for better visualization
    # Position topic in center, people around it
    pos = nx.spring_layout(G, k=1.2, seed=42, iterations=50)
    # Ensure topic is centered
    pos[topic_center] = (0.0, 0.0)
    
    # Color palette
    type_palette = {
        "topic": "#9467BD",  # Purple for topics
        "person": "#12715D",  # Green for people
    }
    
    # Create edge traces - one trace per edge to support variable widths
    edge_traces = []
    edge_weights = []
    for u, v in G.edges():
        weight = G[u][v].get("weight", 1)
        edge_weights.append(weight)
    
    max_weight = max(edge_weights) if edge_weights else 1
    
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        weight = G[u][v].get("weight", 1)
        width = max(1, weight * 3 / max_weight)
        
        edge_traces.append(
            go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode="lines",
                line=dict(
                    width=width,
                    color="rgba(120,120,120,0.3)"
                ),
                hoverinfo="none",
                showlegend=False,
            )
        )
    
    # Create node traces by type
    node_traces = []
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get("ntype", "unknown")
        node_types.setdefault(ntype, []).append(node)
    
    for ntype, nodes in node_types.items():
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        sizes = [G.nodes[n].get("size", 20) for n in nodes]
        color = type_palette.get(ntype, "#808080")
        
        hover_text = []
        display_text = []
        for n in nodes:
            if n == topic_center:
                hover_text.append(f"<b>{topic_center}</b><br>Type: Topic<br>Total Articles: {len(final_df_topic)}")
                display_text.append(topic_center[:20] + "..." if len(topic_center) > 20 else topic_center)
            else:
                weight = int(G[topic_center][n].get("weight", 0)) if topic_center in G[n] else 0
                hover_text.append(f"<b>{n}</b><br>Type: Person<br>Articles: {weight}")
                display_text.append(n[:15] + "..." if len(n) > 15 else n)
        
        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=2, color="white"),
                    opacity=0.9,
                ),
                text=display_text,
                textposition="middle center",
                textfont=dict(size=9 if ntype == "person" else 12, color="white", family="Arial Black"),
                hovertext=hover_text,
                hoverinfo="text",
                name=ntype.title(),
                showlegend=True,
            )
        )
    
    fig = go.Figure(data=edge_traces + node_traces)
    fig.update_layout(
        title=dict(text=f"Topic-People Network: {topic_center}", x=0.5, font=dict(size=16)),
        showlegend=True,
        hovermode="closest",
        dragmode="pan",
        margin=dict(b=30, l=10, r=10, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
    )
    return fig


def build_topic_categorical_network_graph(
    topic_name: str,
    final_df_topic: pd.DataFrame,
    pbr_long_topic: pd.DataFrame
) -> Optional[go.Figure]:
    """Create an ego network for the selected topic across all categorical attributes.
    Filters categorical columns to top 20 if they have more than 20 options, except for person.
    """
    if final_df_topic is None or final_df_topic.empty:
        return None
    
    topic_center = topic_name.strip()
    if not topic_center:
        return None
    
    # Ensure row_index exists
    if 'row_index' not in final_df_topic.columns:
        final_df_topic = final_df_topic.reset_index().rename(columns={'index': 'row_index'})
    
    # Convert row_index to numeric for merging
    final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')
    
    # Get people from pbr_long_topic and add as a column
    if pbr_long_topic is not None and not pbr_long_topic.empty:
        pbr_long_topic['row_index'] = pd.to_numeric(pbr_long_topic['row_index'], errors='coerce')
        # Get unique people per row_index
        people_per_row = pbr_long_topic.groupby('row_index')['person'].apply(
            lambda x: ', '.join(x.dropna().astype(str).unique())
        ).reset_index(name='person')
        # Merge with final_df_topic
        final_df_topic = final_df_topic.merge(people_per_row, on='row_index', how='left')
    
    G = nx.Graph()
    G.add_node(topic_center, ntype="topic", size=60)
    
    categorical_cols = [
        "publication_name",
        "source_name",
        "source_type",
        "channel_name",
        "author_name",
        "tag_name",
        "sentiment_band",
        "person",  # Add person as a categorical column
    ]
    
    for col in categorical_cols:
        if col not in final_df_topic.columns:
            continue
        
        # For person column, handle comma-separated values
        if col == "person":
            # Explode comma-separated persons
            person_series = final_df_topic[col].dropna().astype(str)
            person_list = []
            for persons_str in person_series:
                if ',' in persons_str:
                    person_list.extend([p.strip() for p in persons_str.split(',') if p.strip()])
                else:
                    person_list.append(persons_str.strip())
            
            if not person_list:
                continue
            
            counts = pd.Series(person_list).value_counts()
            # Filter to top 20 if more than 20 options (same as other categorical columns)
            if len(counts) > 20:
                counts = counts.nlargest(20)
        else:
            counts = (
                final_df_topic[col]
                .dropna()
                .astype(str)
                .str.strip()
                .replace("", pd.NA)
                .dropna()
                .value_counts()
            )
            
            # Filter to top 20 if more than 20 options
            if len(counts) > 20:
                counts = counts.nlargest(20)
        
        if counts.empty:
            continue
        
        for value, weight in counts.items():
            node_id = f"{col}:{value}"
            G.add_node(node_id, ntype=col, size=18 + min(int(weight), 20))
            G.add_edge(topic_center, node_id, weight=int(weight))
    
    if G.number_of_edges() == 0:
        return None
    
    pos = nx.spring_layout(G, k=0.7, seed=42, iterations=50)
    pos[topic_center] = (0.0, 0.0)
    
    type_palette = {
        "topic": "#9467BD",
        "person": "#12715D",
        "publication_name": "#4AB48E",
        "source_name": "#2A9D8F",
        "source_type": "#2A9D8F",
        "channel_name": "#D4A115",
        "author_name": "#D94841",
        "tag_name": "#9467BD",
        "sentiment_band": "#8C564B",
    }
    
    edge_x, edge_y = [], []
    for u, v in G.edges():
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        line=dict(width=1, color="rgba(120,120,120,0.4)"),
        hoverinfo="none",
        showlegend=False,
    )
    
    node_traces = []
    node_types = {}
    for node, data in G.nodes(data=True):
        ntype = data.get("ntype", "unknown")
        node_types.setdefault(ntype, []).append(node)
    
    for ntype, nodes in node_types.items():
        xs = [pos[n][0] for n in nodes]
        ys = [pos[n][1] for n in nodes]
        sizes = [G.nodes[n].get("size", 20) for n in nodes]
        color = type_palette.get(ntype, "#808080")
        
        hover_text = []
        for n in nodes:
            if n == topic_center:
                hover_text.append(f"<b>{topic_center}</b><br>Type: Topic<br>Total Articles: {len(final_df_topic)}")
            else:
                label = n.split(":", 1)[-1]
                weight = int(G[topic_center][n].get("weight", 0)) if topic_center in G[n] else 0
                hover_text.append(f"<b>{label}</b><br>Type: {ntype.replace('_', ' ').title()}<br>Articles: {weight}")
        
        display_text = []
        for n in nodes:
            if n == topic_center:
                display_text.append(topic_center[:20] + "..." if len(topic_center) > 20 else topic_center)
            else:
                label = n.split(":", 1)[-1]
                display_text.append(label if len(label) <= 14 else f"{label[:11]}…")
        
        node_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=sizes,
                    color=color,
                    line=dict(width=1, color="white"),
                    opacity=0.85,
                ),
                text=display_text,
                textposition="middle center",
                textfont=dict(size=9, color="white"),
                hovertext=hover_text,
                hoverinfo="text",
                name=ntype.replace("_", " ").title(),
                showlegend=True,
            )
        )
    
    fig = go.Figure(data=[edge_trace] + node_traces)
    fig.update_layout(
        title=dict(text=f"Categorical Network View: {topic_center}", x=0.5, font=dict(size=16)),
        showlegend=True,
        hovermode="closest",
        dragmode="pan",
        margin=dict(b=30, l=10, r=10, t=50),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=600,
    )
    return fig
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

@st.cache_data(show_spinner=False, ttl=3600)
def with_sentiment_band(df: pd.DataFrame) -> pd.DataFrame:
    """Add sentiment band column if not present - simplified"""
    if df is None or df.empty:
        return pd.DataFrame()
    if 'sentiment_score' in df.columns and 'sentiment_band' not in df.columns:
        df = df.copy()
        df['sentiment_band'] = pd.cut(
            df['sentiment_score'],
            bins=[-float('inf'), -0.1, 0.1, float('inf')],
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

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    with col1:
        selected_clusters = st.multiselect("Select Clusters", clusters, default=[], key="tab1_select_clusters") if clusters else []
    with col2:
        sentiment_bands_available = []
        if influencer_df is not None and not influencer_df.empty and 'sentiment_band' in influencer_df.columns:
            try:
                sentiment_bands_available = sorted(influencer_df['sentiment_band'].dropna().unique().tolist())
            except Exception:
                sentiment_bands_available = []
        selected_sentiment_bands_global = st.multiselect("Sentiment Band", sentiment_bands_available, default=[], key="tab1_sentiment_band")
    with col3:
        selected_authors_global = st.multiselect("Authors", author_options, key="tab1_select_authors")
    with col4:
        selected_publications_global = st.multiselect("Publications", publication_options, key="tab1_select_publications")
    with col5:
        selected_source_names_global = st.multiselect("Source Names", source_name_options, key="tab1_select_source_names")
    with col6:
        selected_channels_global = st.multiselect("Channels", channel_options, key="tab1_select_channels")
    with col7:
        selected_topics_global = st.multiselect("Topics", topic_options_global, key="tab1_select_topics")

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
        if selected_topics_global and persons_by_row_filtered is not None and 'tag_name' in persons_by_row_filtered.columns:
            persons_by_row_filtered = persons_by_row_filtered[
                persons_by_row_filtered['tag_name'].astype(str).str.strip().isin(selected_topics_global)
            ]

    if persons_by_row_filtered is None or persons_by_row_filtered is None or (
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
            st.metric("Total Individuals", f"{len(influencer_view):,}", help="Number of unique individuals in the filtered dataset")
        with m2:
            total_mentions = influencer_view.get('mention_count', pd.Series([0]*len(influencer_view))).sum()
            st.metric("Total Mentions", f"{int(total_mentions):,}", help="Total number of mentions across all individuals")
        with m3:
            avg_sentiment = influencer_view.get('sentiment_score', pd.Series(dtype=float)).mean()
            st.metric("Avg Sentiment Score", f"{(avg_sentiment if pd.notna(avg_sentiment) else 0):.2f}", help="Average sentiment score (-1 to 1)")
        with m4:
            avg_circ = influencer_view.get('circulation_size', pd.Series(dtype=float)).mean()
            st.metric("Avg Circulation", f"{(avg_circ if pd.notna(avg_circ) else 0):,.0f}", help="Average circulation size of publications")
    else:
        st.info("No data available. Please check your filters or ensure data files are loaded.")

    show_overview = st.checkbox("See General Overview", value=False)

    if show_overview:
        st.markdown("### General Overview")

        if cluster_col and influencer_view is not None and not influencer_view.empty:
            c1, c2 = st.columns(2)
            with c1:
                sentiment_by_cluster = (
                    influencer_view.groupby(cluster_col)['sentiment_score']
                    .mean()
                    .reset_index()
                    .sort_values('sentiment_score', ascending=False)
                    .head(10)
                )
                if not sentiment_by_cluster.empty:
                    fig_sentiment = go.Figure(data=[
                        go.Bar(x=sentiment_by_cluster[cluster_col], y=sentiment_by_cluster['sentiment_score'])
                    ])
                    fig_sentiment.update_layout(
                        title="Sentiment Distribution by Cluster (Top 10)",
                        xaxis_title="Cluster",
                        yaxis_title="Average Sentiment",
                        height=300,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_sentiment, use_container_width=True)
                else:
                    st.info("No cluster/sentiment data available after filters.")

            with c2:
                circ_by_cluster = (
                    influencer_view.groupby(cluster_col)['circulation_size']
                    .mean()
                    .reset_index()
                    .sort_values('circulation_size', ascending=False)
                    .head(10)
                )
                if not circ_by_cluster.empty:
                    fig_circ = go.Figure(data=[
                        go.Bar(x=circ_by_cluster[cluster_col], y=circ_by_cluster['circulation_size'])
                    ])
                    fig_circ.update_layout(
                        title="Top Clusters by Circulation",
                        xaxis_title="Cluster",
                        yaxis_title="Average Circulation",
                        height=300,
                        showlegend=False,
                    )
                    st.plotly_chart(fig_circ, use_container_width=True)
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
            key="top_people_slider_overview"
        )

        if pbr_long_filtered is None or pbr_long_filtered.empty:
            st.info("Person-by-row data is required to surface the top individuals.")
        else:
            top_people_counts = pbr_long_filtered['person'].value_counts().reset_index()
            top_people_counts.columns = ['person', 'article_count']

            if top_people_counts.empty:
                st.info("No individuals found for the current selection.")
            else:
                top_people_counts = top_people_counts.head(n_top_general)
                fig_top_people = px.bar(
                    top_people_counts.sort_values('article_count', ascending=True),
                    x='article_count',
                    y='person',
                    orientation='h',
                    labels={'article_count': 'Number of Articles', 'person': 'Individual'},
                    title=f"Top {len(top_people_counts)} Individuals by Article Mentions"
                )
                fig_top_people.update_layout(
                    template='simple_white',
                    margin=dict(l=10, r=10, t=60, b=10),
                    height=max(400, len(top_people_counts) * 22),
                    yaxis={'categoryorder': 'array', 'categoryarray': top_people_counts.sort_values('article_count')['person']}
                )
                st.plotly_chart(fig_top_people, use_container_width=True)

    # --------------------------------------
    # Main Tabs: People and Topics
    # --------------------------------------
    st.markdown("<br>", unsafe_allow_html=True)  # Add some spacing
    people_tab, topic_tab = st.tabs(["👥 People", "🏷️ Topics"])

    # ------------------------------- People Tab ------------------------------- #
    with people_tab:
        st.markdown("### Search Individual")

        people_options = all_people_filtered if all_people_filtered else all_people
        col_person, col_keyword = st.columns([2, 1])
        with col_person:
            story_person = st.selectbox(
                "Select an Individual",
                options=[""] + people_options,
                index=0,
                help="Fast, case-insensitive, normalized people search",
                key="tab2_story_person"
            )
        with col_keyword:
            story_keyword = st.text_input(
                "Optional Keyword Filter",
                placeholder="e.g., vaccine, policy...",
                key="tab2_story_keyword"
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

                    needle = story_person.strip().lower()
                    matching_rows = pbr_long_filtered.loc[
                        pbr_long_filtered['person_norm_lc'] == needle, 'row_index'
                    ].unique()
                    person_articles = (
                        final_df_people[final_df_people['row_index'].isin(matching_rows)].copy()
                        if len(matching_rows) > 0 else pd.DataFrame()
                    )

                    if story_keyword and story_keyword.strip() and not person_articles.empty:
                        keyword = story_keyword.strip().lower()
                        keyword_mask = pd.Series(False, index=person_articles.index)
                        for col in ['headline', 'article_body']:
                            if col in person_articles.columns:
                                keyword_mask = keyword_mask | person_articles[col].astype(str).str.lower().str.contains(
                                    keyword, regex=False
                                )
                        person_articles = person_articles[keyword_mask]

                    if person_articles.empty:
                        st.warning(
                            f"No article data found for {story_person}"
                            + (f" with keyword '{story_keyword}'" if story_keyword and story_keyword.strip() else "")
                            + " within the current filters."
                        )
                    else:
                        st.markdown("### Summary")
                        summary_cols = st.columns(4)
                        with summary_cols[0]:
                            st.metric("Total Articles", len(person_articles))
                        with summary_cols[1]:
                            avg_sentiment = person_articles.get('sentiment_score', pd.Series(dtype=float)).mean()
                            st.metric("Avg Sentiment", f"{(avg_sentiment if pd.notna(avg_sentiment) else 0):.2f}")
                        with summary_cols[2]:
                            total_circ = person_articles.get('circulation_size', pd.Series(dtype=float)).sum()
                            st.metric("Total Reach", f"{int(total_circ) if pd.notna(total_circ) else 0:,}")
                        with summary_cols[3]:
                            unique_pubs = person_articles.get('publication_name', pd.Series(dtype=str)).nunique()
                            st.metric("Unique Publishers", int(unique_pubs) if pd.notna(unique_pubs) else 0)

                        extra_metrics = []
                        if 'hit_strength' in person_articles.columns:
                            extra_metrics.append(("Total Hit Strength", person_articles['hit_strength'].sum()))
                        if 'vipr_score' in person_articles.columns:
                            extra_metrics.append(("Total VIPR Score", person_articles['vipr_score'].sum()))
                        if 'vipr_weight' in person_articles.columns:
                            extra_metrics.append(("Total VIPR Weight", person_articles['vipr_weight'].sum()))
                        if extra_metrics:
                            metric_cols = st.columns(len(extra_metrics))
                            for col_obj, (label, value) in zip(metric_cols, extra_metrics):
                                with col_obj:
                                    display_val = f"{value:,.0f}" if pd.notna(value) else "N/A"
                                    st.metric(label, display_val)

                        viz_col1, viz_col2 = st.columns(2)
                        with viz_col1:
                            if 'emotion_body' in person_articles.columns:
                                emotion_counts = person_articles['emotion_body'].value_counts().dropna().head(12)
                                if not emotion_counts.empty:
                                    fig_emotions = px.bar(
                                        x=emotion_counts.values,
                                        y=emotion_counts.index,
                                        orientation='h',
                                        title=f'Emotions in Articles About {story_person}',
                                        labels={'x': 'Number of Articles', 'y': 'Emotion'}
                                    )
                                    fig_emotions.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                                    st.plotly_chart(fig_emotions, use_container_width=True)
                                else:
                                    st.info("No emotion data available.")
                            else:
                                st.info("Emotion column not available.")

                        with viz_col2:
                            if 'sentiment_band' not in person_articles.columns:
                                person_articles = with_sentiment_band(person_articles)
                            sentiment_counts = person_articles['sentiment_band'].value_counts().dropna()
                            if not sentiment_counts.empty:
                                fig_sentiment_person = px.bar(
                                    x=sentiment_counts.values,
                                    y=sentiment_counts.index,
                                    orientation='h',
                                    title=f'Sentiment in Articles About {story_person}',
                                    labels={'x': 'Number of Articles', 'y': 'Sentiment'}
                                )
                                fig_sentiment_person.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
                                st.plotly_chart(fig_sentiment_person, use_container_width=True)
                            else:
                                st.info("No sentiment distribution available.")

                        st.markdown("### Mentions Over Time")
                        if 'published_datetime' in person_articles.columns:
                            mentions_df = person_articles[['published_datetime']].copy()
                            mentions_df['published_datetime'] = pd.to_datetime(
                                mentions_df['published_datetime'], errors='coerce'
                            )
                            mentions_df = mentions_df.dropna(subset=['published_datetime'])
                            if not mentions_df.empty:
                                mentions_df['date'] = mentions_df['published_datetime'].dt.date
                                mentions_series = (
                                    mentions_df.groupby('date')
                                    .size()
                                    .reset_index(name='mentions')
                                    .sort_values('date')
                                )
                                fig_mentions = px.line(
                                    mentions_series,
                                    x='date',
                                    y='mentions',
                                    title=f"Mentions of {story_person} Over Time",
                                    labels={'date': 'Date', 'mentions': 'Number of Articles'}
                                )
                                fig_mentions.update_layout(
                                    template='simple_white',
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=350,
                                )
                                st.plotly_chart(fig_mentions, use_container_width=True)
                            else:
                                st.info("No published dates available to plot mentions over time.")
                        else:
                            st.info("Published datetime column is not available for mentions over time.")

                        st.markdown("### Network Analysis")
                        network_fig = build_person_network_graph(person_articles, story_person)
                        if network_fig:
                            st.plotly_chart(
                                network_fig,
                                use_container_width=True,
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
                            key="export_people_data"
                        )
                        st.caption(f"Exporting {len(person_articles):,} articles with current filters.")
        else:
            st.info("Select a person to see detailed results.")

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
                    key="topic_select"
                )
            with keyword_col:
                topic_keyword = st.text_input(
                    "Keyword Filter",
                    placeholder="e.g., vaccine, policy...",
                    key="topic_keyword"
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
                        key="top_people_slider"
                    )

                    if not pbr_long_topic.empty:
                        top_n_people = pbr_long_topic['person'].value_counts().head(num_people).index.tolist()
                    else:
                        top_n_people = []

                    sentiment_scores = final_df_topic['sentiment_score'].dropna()
                    has_sentiment = len(sentiment_scores) > 0
                    if has_sentiment and 'sentiment_band' not in final_df_topic.columns:
                        final_df_topic['sentiment_band'] = pd.cut(
                            final_df_topic['sentiment_score'],
                            bins=[-float('inf'), -0.1, 0.1, float('inf')],
                            labels=['negative', 'neutral', 'positive']
                        )

                    chart_col1, chart_col2 = st.columns(2)

                    with chart_col1:
                        with st.container(border=True):
                            st.markdown(f"#### Top {num_people} People by Sentiment")

                            if has_sentiment and not pbr_long_topic.empty and len(top_n_people) > 0:
                                pbr_filtered = pbr_long_topic[pbr_long_topic['person'].isin(top_n_people)].copy()
                                pbr_filtered['row_index'] = pd.to_numeric(pbr_filtered['row_index'], errors='coerce')
                                final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')

                                sent_merge = pbr_filtered[['row_index', 'person']].merge(
                                    final_df_topic[['row_index', 'sentiment_band']],
                                    on='row_index',
                                    how='left'
                                )
                                sent_merge = sent_merge[sent_merge['sentiment_band'].notna()]

                                if not sent_merge.empty:
                                    sent_counts = (
                                        sent_merge.groupby(['person', 'sentiment_band'])
                                        .size()
                                        .reset_index(name='count')
                                    )
                                    all_bands = ['negative', 'neutral', 'positive']
                                    sent_pivot = sent_counts.pivot_table(
                                        index='person',
                                        columns='sentiment_band',
                                        values='count',
                                        fill_value=0
                                    ).reset_index()
                                    sent_pivot = sent_pivot[sent_pivot['person'].isin(top_n_people)]
                                    for band in all_bands:
                                        if band not in sent_pivot.columns:
                                            sent_pivot[band] = 0
                                    sent_pivot['total'] = sent_pivot[all_bands].sum(axis=1)
                                    sent_pivot = sent_pivot.sort_values('total', ascending=True).tail(num_people)
                                    sent_chart_df = sent_pivot.melt(
                                        id_vars=['person', 'total'],
                                        value_vars=all_bands,
                                        var_name='sentiment_band',
                                        value_name='count'
                                    )

                                    fig_sent = go.Figure()
                                    band_colors = {'negative': '#D94841', 'neutral': '#D4A115', 'positive': '#4AB48E'}
                                    for band in all_bands:
                                        band_df = sent_chart_df[sent_chart_df['sentiment_band'] == band]
                                        fig_sent.add_trace(go.Bar(
                                            name=band.title(),
                                            y=band_df['person'],
                                            x=band_df['count'],
                                            orientation='h',
                                            marker_color=band_colors.get(band, '#808080')
                                        ))
                                    person_order = sent_pivot.sort_values('total', ascending=True)['person'].tolist()
                                    fig_sent.update_layout(
                                        template='simple_white',
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        height=max(600, len(top_n_people) * 25),
                                        barmode='stack',
                                        xaxis_title='Number of Articles',
                                        yaxis_title='Person',
                                        yaxis={'categoryorder': 'array', 'categoryarray': person_order},
                                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig_sent, use_container_width=True)
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

                            if not pbr_long_topic.empty and len(top_n_people) > 0:
                                pbr_filtered_emotion = pbr_long_topic[pbr_long_topic['person'].isin(top_n_people)].copy()
                                pbr_filtered_emotion['row_index'] = pd.to_numeric(
                                    pbr_filtered_emotion['row_index'], errors='coerce'
                                )
                                final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')

                                emotion_merge = pbr_filtered_emotion[['row_index', 'person']].merge(
                                    final_df_topic[['row_index', 'emotion_body']],
                                    on='row_index',
                                    how='left'
                                )
                                emotion_merge = emotion_merge[emotion_merge['emotion_body'].notna()]
                                emotion_merge['emotion_body'] = emotion_merge['emotion_body'].str.capitalize()

                                if not emotion_merge.empty:
                                    emotion_counts = (
                                        emotion_merge.groupby(['person', 'emotion_body'])
                                        .size()
                                        .reset_index(name='count')
                                    )
                                    all_emotions = sorted(emotion_counts['emotion_body'].unique())
                                    emotion_pivot = emotion_counts.pivot_table(
                                        index='person',
                                        columns='emotion_body',
                                        values='count',
                                        fill_value=0
                                    ).reset_index()
                                    emotion_pivot = emotion_pivot[emotion_pivot['person'].isin(top_n_people)]
                                    emotion_pivot['total'] = emotion_pivot[all_emotions].sum(axis=1)
                                    emotion_pivot = emotion_pivot.sort_values('total', ascending=True).tail(num_people)
                                    emotion_chart_df = emotion_pivot.melt(
                                        id_vars=['person', 'total'],
                                        value_vars=all_emotions,
                                        var_name='emotion',
                                        value_name='count'
                                    )

                                    fig_emotion = go.Figure()
                                    emotion_colors = px.colors.qualitative.Set3[:len(all_emotions)]
                                    if len(all_emotions) > len(emotion_colors):
                                        emotion_colors.extend(
                                            px.colors.qualitative.Pastel[:len(all_emotions) - len(emotion_colors)]
                                        )
                                    for idx, emotion in enumerate(all_emotions):
                                        emotion_df = emotion_chart_df[emotion_chart_df['emotion'] == emotion]
                                        fig_emotion.add_trace(go.Bar(
                                            name=emotion.title() if emotion else 'Unknown',
                                            y=emotion_df['person'],
                                            x=emotion_df['count'],
                                            orientation='h',
                                            marker_color=emotion_colors[idx % len(emotion_colors)]
                                        ))
                                    person_order_emotion = emotion_pivot.sort_values('total', ascending=True)['person'].tolist()
                                    fig_emotion.update_layout(
                                    template='simple_white',
                                    margin=dict(l=10, r=10, t=10, b=10),
                                        height=max(600, len(top_n_people) * 25),
                                        barmode='stack',
                                        xaxis_title='Number of Articles',
                                        yaxis_title='Person',
                                        yaxis={'categoryorder': 'array', 'categoryarray': person_order_emotion},
                                        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                                        showlegend=True
                                    )
                                    st.plotly_chart(fig_emotion, use_container_width=True)
                                else:
                                    st.info("No emotion data available.")
                            else:
                                st.info("No people detected for this topic.")

                    with st.container(border=True):
                        st.markdown(f"#### Mentions Over Time - Top {num_people} People")

                        if not pbr_long_topic.empty and len(top_n_people) > 0:
                            pbr_filtered_time = pbr_long_topic[pbr_long_topic['person'].isin(top_n_people)].copy()
                            pbr_filtered_time['row_index'] = pd.to_numeric(
                                pbr_filtered_time['row_index'], errors='coerce'
                            )
                            final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')

                            time_merge = pbr_filtered_time[['row_index', 'person']].merge(
                                final_df_topic[['row_index', 'published_datetime', 'circulation_size']],
                                on='row_index',
                                how='left'
                            )
                            time_merge = time_merge[time_merge['published_datetime'].notna()]
                            time_merge = time_merge[time_merge['circulation_size'].notna()]

                            if not time_merge.empty:
                                time_merge['published_datetime'] = pd.to_datetime(
                                    time_merge['published_datetime'], errors='coerce'
                                )
                                time_merge = time_merge[time_merge['published_datetime'].notna()]
                                time_merge['date'] = time_merge['published_datetime'].dt.date
                                time_grouped = (
                                    time_merge.groupby(['date', 'person'])['circulation_size']
                                    .sum()
                                    .reset_index()
                                )
                                time_pivot = time_grouped.pivot_table(
                                    index='date',
                                    columns='person',
                                    values='circulation_size',
                                    fill_value=0
                                ).sort_index()

                                fig_time = go.Figure()
                                colors = px.colors.qualitative.Set3[:len(top_n_people)]
                                if len(top_n_people) > len(colors):
                                    colors.extend(px.colors.qualitative.Pastel[:len(top_n_people) - len(colors)])

                                for idx, person in enumerate(top_n_people):
                                    if person in time_pivot.columns:
                                        fig_time.add_trace(go.Scatter(
                                            name=person,
                                            x=time_pivot.index,
                                            y=time_pivot[person],
                                            mode='lines',
                                            stackgroup='one',
                                            fill='tonexty' if idx > 0 else 'tozeroy',
                                            line=dict(width=0.6, color=colors[idx % len(colors)]),
                                            fillcolor=colors[idx % len(colors)]
                                        ))

                                fig_time.update_layout(
                                    template='simple_white',
                                    margin=dict(l=10, r=10, t=10, b=10),
                                    height=500,
                                    xaxis_title='Date',
                                    yaxis_title='Circulation Size',
                                    hovermode='x unified',
                                    legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
                                    showlegend=True
                                )
                                st.plotly_chart(fig_time, use_container_width=True)
                            else:
                                st.info("No data available for mentions over time.")
                        else:
                            st.info("No people detected for this topic.")

                    with st.container(border=True):
                        st.markdown("#### Circulation Quartile Distribution")

                        final_df_topic_circ = final_df_topic.copy()
                        circ_data = final_df_topic_circ['circulation_size'].dropna()

                        if len(circ_data) > 0:
                            try:
                                final_df_topic_circ['circulation_quartile'] = pd.qcut(
                                    circ_data,
                                    q=min(4, len(circ_data.unique())),
                                    labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'][:min(4, len(circ_data.unique()))],
                                    duplicates='drop'
                                )
                                quartile_counts = (
                                    final_df_topic_circ['circulation_quartile']
                                    .value_counts()
                                    .sort_index()
                                )

                                if not quartile_counts.empty:
                                    fig_circ = go.Figure(data=[
                                        go.Bar(
                                            x=quartile_counts.index.astype(str),
                                            y=quartile_counts.values,
                                            marker_color='#12715D',
                                            text=quartile_counts.values,
                                            textposition='outside'
                                        )
                                    ])
                                    fig_circ.update_layout(
                                        template='simple_white',
                                        margin=dict(l=10, r=10, t=10, b=10),
                                        height=300,
                                        xaxis_title='Circulation Quartile',
                                        yaxis_title='Number of Articles',
                                        showlegend=False
                                    )
                                    st.plotly_chart(fig_circ, use_container_width=True)
                                else:
                                    st.info("No circulation quartile data available.")
                            except (ValueError, TypeError):
                                try:
                                    bins = pd.cut(
                                        circ_data,
                                        bins=4,
                                        labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
                                        duplicates='drop'
                                    )
                                    final_df_topic_circ.loc[circ_data.index, 'circulation_quartile'] = bins
                                    quartile_counts = (
                                        final_df_topic_circ['circulation_quartile']
                                        .value_counts()
                                        .sort_index()
                                    )

                                    if not quartile_counts.empty:
                                        fig_circ = go.Figure(data=[
                                            go.Bar(
                                                x=quartile_counts.index.astype(str),
                                                y=quartile_counts.values,
                                                marker_color='#12715D',
                                                text=quartile_counts.values,
                                                textposition='outside'
                                            )
                                        ])
                                        fig_circ.update_layout(
                                            template='simple_white',
                                            margin=dict(l=10, r=10, t=10, b=10),
                                            height=300,
                                            xaxis_title='Circulation Quartile',
                                            yaxis_title='Number of Articles',
                                            showlegend=False
                                        )
                                        st.plotly_chart(fig_circ, use_container_width=True)
                                    else:
                                        st.info("Unable to create circulation quartiles.")
                                except Exception:
                                    st.info("Circulation data available but cannot be divided into quartiles.")
                        else:
                            st.info("No circulation data available.")

                    with st.container(border=True):
                        st.markdown("#### Network Analysis")
                        
                        categorical_network_fig = build_topic_categorical_network_graph(
                            selected_tag,
                            final_df_topic,
                            pbr_long_topic
                        )
                        if categorical_network_fig:
                            st.plotly_chart(
                                categorical_network_fig,
                                use_container_width=True,
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