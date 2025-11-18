"""
Network Analysis Functions
Handles network analysis, graph building, and network visualization
"""

from typing import List, Optional, Tuple, TYPE_CHECKING
from collections import Counter
from itertools import combinations
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

# Try to import streamlit-agraph for interactive network graphs
if TYPE_CHECKING:
    from streamlit_agraph import Node, Edge, Config

try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGraph_AVAILABLE = True
except ImportError:
    AGraph_AVAILABLE = False
    # Define placeholder classes to avoid NameError when module is imported
    class Node:
        pass
    class Edge:
        pass
    class Config:
        pass
    def agraph(*args, **kwargs):
        pass


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

    # Add edges between nodes that co-occur in the same articles
    _add_cooccurrence_edges(G, person_articles, categorical_cols, center, min_cooccurrences=2)

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
    
    # Add edges between nodes that co-occur in the same articles
    _add_cooccurrence_edges(G, final_df_topic, categorical_cols, topic_center, min_cooccurrences=2, handle_person_col=True)

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


def _add_cooccurrence_edges(G: nx.Graph, df: pd.DataFrame, categorical_cols: List[str], center_node: str, min_cooccurrences: int = 2, handle_person_col: bool = False) -> None:
    """Helper function to add edges between nodes that co-occur in the same articles."""
    edge_weights = Counter()
    
    for _, row in df.iterrows():
        nodes_in_row = [center_node]
        
        for col in categorical_cols:
            if col not in row or pd.isna(row[col]):
                continue
            
            if handle_person_col and col == "person":
                # Handle comma-separated persons
                person_str = str(row[col]).strip()
                if person_str:
                    for person in [p.strip() for p in person_str.split(',') if p.strip()]:
                        node_id = f"{col}:{person}"
                        if node_id in G.nodes():
                            nodes_in_row.append(node_id)
            else:
                value = str(row[col]).strip()
                if value:
                    node_id = f"{col}:{value}"
                    if node_id in G.nodes():
                        nodes_in_row.append(node_id)
        
        # Create edges between all pairs of nodes in this article
        if len(nodes_in_row) > 1:
            for node1, node2 in combinations(nodes_in_row, 2):
                edge_weights[tuple(sorted([node1, node2]))] += 1
    
    # Add edges to graph (only if they meet minimum co-occurrence threshold)
    for (node1, node2), weight in edge_weights.items():
        if weight >= min_cooccurrences:
            if G.has_edge(node1, node2):
                G[node1][node2]['weight'] += weight
            else:
                G.add_edge(node1, node2, weight=weight)


def build_person_network_graph_interactive(person_articles: pd.DataFrame, person_name: str) -> Optional[Tuple[List[Node], List[Edge], Config]]:
    """Create an interactive ego network for the selected person with draggable nodes using streamlit-agraph."""
    if not AGraph_AVAILABLE:
        return None
    
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

    # Add edges between nodes that co-occur in the same articles
    _add_cooccurrence_edges(G, person_articles, categorical_cols, center, min_cooccurrences=2)

    if G.number_of_edges() == 0:
        return None

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

    # Create nodes for agraph
    nodes = []
    for node, data in G.nodes(data=True):
        ntype = data.get("ntype", "unknown")
        size = data.get("size", 20)
        color = type_palette.get(ntype, "#808080")
        
        if node == center:
            label = center
            title = f"<b>{center}</b><br>Type: Person"
        else:
            label = node.split(":", 1)[-1]
            if len(label) > 20:
                label = label[:17] + "..."
            weight = int(G[center][node].get("weight", 0)) if center in G[node] else 0
            title = f"<b>{label}</b><br>Type: {ntype.replace('_', ' ').title()}<br>Articles: {weight}"
        
        # Ensure label is not empty
        if not label or label.strip() == "":
            label = str(node)[:20]
        
        # Calculate appropriate font size based on node size
        node_size = max(size, 35)  # Minimum size to ensure label fits
        font_size = max(12, min(16, int(node_size / 2.5)))  # Scale font with node size
        
        nodes.append(
            Node(
                id=node,
                label=str(label),  # Ensure label is a string
                size=node_size,
                color=color,
                title=title,
                font={"color": "white", "size": font_size, "face": "Arial", "align": "center", "strokeWidth": 2, "strokeColor": "#000000"},
                shape="circle"  # Use circle shape to ensure labels render inside
            )
        )

    # Create edges for agraph
    edges = []
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        # Different styling for center edges vs. inter-node edges
        is_center_edge = (u == center or v == center)
        edge_color = "#12715D" if is_center_edge else "#888888"  # Green for center edges, grey for others
        edge_width = min(5, max(1, weight / 5)) if is_center_edge else min(3, max(0.5, weight / 15))
        edges.append(
            Edge(
                source=u,
                target=v,
                weight=weight,
                width=edge_width,
                color=edge_color,
                title=f"Co-occurs in {int(weight)} article(s)"
            )
        )

    # Create configuration with physics enabled for interactive dragging
    config = Config(
        width="100%",
        height=600,
        directed=False,
        physics={
            "enabled": True,
            "stabilization": {"enabled": True, "iterations": 100},
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08,
                "damping": 0.4,
                "avoidOverlap": 1
            }
        },
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={
            "labelProperty": "label",
            "font": {"size": 14, "color": "white", "face": "Arial", "align": "center", "strokeWidth": 2, "strokeColor": "#000000"},
            "scaling": {
                "min": 20,
                "max": 60,
                "label": {
                    "enabled": True,
                    "min": 14,
                    "max": 18,
                    "maxVisible": 18,
                    "drawThreshold": 5
                }
            },
            "chosen": True,
            "labelHighlightBold": True,
            "mass": 1,
            "shape": "circle",
            "borderWidth": 2,
            "borderColor": "#ffffff"
        },
    )

    return nodes, edges, config


def build_topic_categorical_network_graph_interactive(
    topic_name: str,
    final_df_topic: pd.DataFrame,
    pbr_long_topic: pd.DataFrame
) -> Optional[Tuple[List[Node], List[Edge], Config]]:
    """Create an interactive ego network for the selected topic with draggable nodes using streamlit-agraph."""
    if not AGraph_AVAILABLE:
        return None
    
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
        "person",
    ]
    
    for col in categorical_cols:
        if col not in final_df_topic.columns:
            continue
        
        # For person column, handle comma-separated values
        if col == "person":
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
            
            if len(counts) > 20:
                counts = counts.nlargest(20)
        
        if counts.empty:
            continue
        
        for value, weight in counts.items():
            node_id = f"{col}:{value}"
            G.add_node(node_id, ntype=col, size=18 + min(int(weight), 20))
            G.add_edge(topic_center, node_id, weight=int(weight))
    
    # Add edges between nodes that co-occur in the same articles
    _add_cooccurrence_edges(G, final_df_topic, categorical_cols, topic_center, min_cooccurrences=2, handle_person_col=True)

    if G.number_of_edges() == 0:
        return None
    
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

    # Create nodes for agraph
    nodes = []
    for node, data in G.nodes(data=True):
        ntype = data.get("ntype", "unknown")
        size = data.get("size", 20)
        color = type_palette.get(ntype, "#808080")
        
        if node == topic_center:
            label = topic_center[:30] + "..." if len(topic_center) > 30 else topic_center
            title = f"<b>{topic_center}</b><br>Type: Topic<br>Total Articles: {len(final_df_topic)}"
        else:
            label = node.split(":", 1)[-1]
            if len(label) > 20:
                label = label[:17] + "..."
            weight = int(G[topic_center][node].get("weight", 0)) if topic_center in G[node] else 0
            title = f"<b>{label}</b><br>Type: {ntype.replace('_', ' ').title()}<br>Articles: {weight}"
        
        # Ensure label is not empty
        if not label or label.strip() == "":
            label = str(node)[:20]
        
        # Calculate appropriate font size based on node size
        node_size = max(size, 35)  # Minimum size to ensure label fits
        font_size = max(12, min(16, int(node_size / 2.5)))  # Scale font with node size
        
        nodes.append(
            Node(
                id=node,
                label=str(label),  # Ensure label is a string
                size=node_size,
                color=color,
                title=title,
                font={"color": "white", "size": font_size, "face": "Arial", "align": "center", "strokeWidth": 2, "strokeColor": "#000000"},
                shape="circle"  # Use circle shape to ensure labels render inside
            )
        )

    # Create edges for agraph
    edges = []
    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)
        # Different styling for center edges vs. inter-node edges
        is_center_edge = (u == topic_center or v == topic_center)
        edge_color = "#9467BD" if is_center_edge else "#888888"  # Purple for topic center edges, grey for others
        edge_width = min(5, max(1, weight / 5)) if is_center_edge else min(3, max(0.5, weight / 15))
        edges.append(
            Edge(
                source=u,
                target=v,
                weight=weight,
                width=edge_width,
                color=edge_color,
                title=f"Co-occurs in {int(weight)} article(s)"
            )
        )

    # Create configuration with physics enabled for interactive dragging
    config = Config(
        width="100%",
        height=600,
        directed=False,
        physics={
            "enabled": True,
            "stabilization": {"enabled": True, "iterations": 100},
            "forceAtlas2Based": {
                "gravitationalConstant": -50,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.08,
                "damping": 0.4,
                "avoidOverlap": 1
            }
        },
        nodeHighlightBehavior=True,
        highlightColor="#F7A7A6",
        collapsible=False,
        node={
            "labelProperty": "label",
            "font": {"size": 14, "color": "white", "face": "Arial", "align": "center", "strokeWidth": 2, "strokeColor": "#000000"},
            "scaling": {
                "min": 20,
                "max": 60,
                "label": {
                    "enabled": True,
                    "min": 14,
                    "max": 18,
                    "maxVisible": 18,
                    "drawThreshold": 5
                }
            },
            "chosen": True,
            "labelHighlightBold": True,
            "mass": 1,
            "shape": "circle",
            "borderWidth": 2,
            "borderColor": "#ffffff"
        },
    )

    return nodes, edges, config

