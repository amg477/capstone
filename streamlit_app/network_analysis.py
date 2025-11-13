"""
Network Analysis Functions
Handles network analysis, graph building, and network visualization
"""

from typing import Dict, Tuple, List, Set, Iterable, Optional
from collections import Counter
from itertools import combinations
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st

# Try to import streamlit-agraph for interactive network graphs
try:
    from streamlit_agraph import agraph, Node, Edge, Config
    AGraph_AVAILABLE = True
except ImportError:
    AGraph_AVAILABLE = False


def bigram_strings(tokens: List[str]) -> Set[str]:
    """Extract bigrams from token list."""
    if len(tokens) < 2:
        return set()
    return {f"{a} {b}" for a, b in zip(tokens[:-1], tokens[1:])}


def prepare_whitelist_sets(whitelist_terms: Iterable[str], term_weight_tbl: pd.DataFrame) -> Tuple[Set[str], Set[str], Dict[str, float]]:
    """Prepare unigram and bigram whitelist sets with weights."""
    tw_map = {str(t): float(w) for t, w in term_weight_tbl[["term","term_weight"]].itertuples(index=False, name=None)}
    uni, bi = set(), set()
    for t in whitelist_terms:
        t = str(t).strip()
        if not t:
            continue
        if " " in t:
            bi.add(t.lower())
        else:
            uni.add(t.lower())
        if t not in tw_map:
            tw_map[t] = 1.0
    return uni, bi, tw_map


def best_term_from_tokens(tokens: List[str], uni_whitelist: Set[str], bi_whitelist: Set[str],
                          tw_map: Dict[str, float], min_weight: float = 0.0) -> Tuple[str, float]:
    """Find the best matching term from tokens."""
    tok_set = set(tokens)
    bi_set  = bigram_strings(tokens)
    matches = []
    for t in bi_set.intersection(bi_whitelist):
        w = tw_map.get(t, 1.0)
        if w >= min_weight:
            matches.append((t, w))
    for t in tok_set.intersection(uni_whitelist):
        w = tw_map.get(t, 1.0)
        if w >= min_weight:
            matches.append((t, w))
    if not matches:
        return None, 0.0
    matches.sort(key=lambda x: (x[1], len(x[0])), reverse=True)
    return matches[0][0], float(matches[0][1])


def build_content_network_edges(
    df: pd.DataFrame,
    attr_df: pd.DataFrame,
    whitelist_terms: Iterable[str],
    publisher_col: str = "publication_name",
    min_term_weight: float = 0.0,
    use_max_term_credit_first: bool = True
) -> pd.DataFrame:
    """Build edges for content network analysis."""
    # Get term weights from attribution data
    term_w = (attr_df.query("kind=='term'")
              .loc[:, ["value","credit_share"]]
              .rename(columns={"value":"term","credit_share":"term_weight"}))
    term_w["term"] = term_w["term"].astype(str)
    wl = [str(t).strip() for t in whitelist_terms if str(t).strip()]
    if not wl:
        return pd.DataFrame(columns=["publisher","term","weight"])

    # Keep whitelist terms, fill missing with 1.0
    keep = term_w[term_w["term"].isin(wl)]
    if keep.empty:
        keep = pd.DataFrame({"term": wl, "term_weight": 1.0})
    else:
        missing = set(wl) - set(keep["term"])
        if missing:
            keep = pd.concat([keep, pd.DataFrame({"term": list(missing), "term_weight": 1.0})], ignore_index=True)

    uni_wl, bi_wl, TW_MAP = prepare_whitelist_sets(wl, keep)

    # Prepare data
    use = df[[publisher_col, "headline", "body"]].copy()
    use[publisher_col] = use[publisher_col].replace("", "Unknown").fillna("Unknown").astype(str)
    for c in ["headline", "body"]:
        if c in use.columns:
            use[c] = use[c].fillna("").astype(str).str.lower()

    if "max_term_credit" not in df.columns:
        df = df.assign(max_term_credit=0.0)
    if "vipr_weight" not in df.columns:
        df = df.assign(vipr_weight=1.0)

    # Build edges
    edges: Dict[Tuple[str,str], float] = {}
    for idx, r in use.iterrows():
        pub = r[publisher_col]
        tokens = (r.get("headline", "") + " " + r.get("body", "")).split()
        if not tokens:
            continue
        best_t, best_global_w = best_term_from_tokens(tokens, uni_wl, bi_wl, TW_MAP, min_weight=min_term_weight)
        if not best_t:
            continue
        w_article = float(pd.to_numeric(df.loc[idx, "max_term_credit"], errors="coerce") or 0.0) if use_max_term_credit_first else 0.0
        if w_article <= 0:
            w_article = best_global_w * float(pd.to_numeric(df.loc[idx, "vipr_weight"], errors="coerce") or 1.0)
        key = (pub, best_t)
        edges[key] = edges.get(key, 0.0) + w_article

    if not edges:
        return pd.DataFrame(columns=["publisher","term","weight"])
    pub, term, w = zip(*[(p, t, v) for (p, t), v in edges.items()])
    return pd.DataFrame({"publisher": pub, "term": term, "weight": w})


def build_content_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build NetworkX graph from edges."""
    G = nx.Graph()
    for _, r in edges.iterrows():
        p, t, w = r["publisher"], r["term"], float(r["weight"])
        G.add_node(p, ntype="publisher")
        G.add_node(t, ntype="term")
        G.add_edge(p, t, weight=w)
    return G


def community_map_content(G: nx.Graph) -> Dict[str, int]:
    """Map nodes to communities."""
    if G.number_of_edges() == 0:
        return {}
    comms = list(greedy_modularity_communities(G, weight="weight"))
    node2c: Dict[str, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node2c[n] = i
    return node2c


def create_interactive_network_visualization(
    G: nx.Graph,
    node2c: Dict[str, int],
    edges_df: pd.DataFrame,
    title: str = "Content Network: Publishers ↔ High-Impact Terms"
):
    """Create interactive Plotly network visualization."""
    if G.number_of_nodes() == 0:
        return None
    
    # Calculate node metrics
    strength = {n: sum(G[n][nbr].get("weight", 0.0) for nbr in G.neighbors(n)) for n in G.nodes()}
    degree = dict(G.degree())
    
    # Get node positions using spring layout
    pos = nx.spring_layout(G, k=1, seed=42, weight="weight")
    
    # Prepare node data
    node_trace = []
    edge_trace = []
    
    # Community colors
    base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                   "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    
    # Separate publishers and terms
    pubs = [n for n, d in G.nodes(data=True) if d.get("ntype") == "publisher"]
    terms = [n for n, d in G.nodes(data=True) if d.get("ntype") == "term"]
    
    # Create node traces for publishers and terms
    for node_type, nodes, symbol in [("Publisher", pubs, "circle"), ("Term", terms, "triangle-up")]:
        if not nodes:
            continue
            
        x_vals = [pos[node][0] for node in nodes]
        y_vals = [pos[node][1] for node in nodes]
        
        # Node sizes based on strength
        sizes = [max(10, min(50, strength[node] * 20)) for node in nodes]
        
        # Colors based on community
        colors = [base_colors[node2c.get(node, 0) % len(base_colors)] for node in nodes]
        
        # Hover text
        hover_text = []
        for node in nodes:
            community = node2c.get(node, 0)
            node_strength = strength[node]
            node_degree = degree[node]
            
            # Get connected nodes
            neighbors = list(G.neighbors(node))
            
            hover_info = f"""
            <b>{node}</b><br>
            Type: {node_type}<br>
            Community: {community}<br>
            Strength: {node_strength:.2f}<br>
            Degree: {node_degree}<br>
            Connected to: {len(neighbors)} nodes<br>
            """
            hover_text.append(hover_info)
        
        node_trace.append(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode='markers+text',
            marker=dict(
                size=sizes,
                color=colors,
                symbol=symbol,
                line=dict(width=2, color='white'),
                opacity=0.8
            ),
            text=[node[:15] + "..." if len(node) > 15 else node for node in nodes],
            textposition="middle center",
            textfont=dict(size=8, color="white"),
            hovertext=hover_text,
            hoverinfo="text",
            name=node_type,
            showlegend=True
        ))
    
    # Create edge trace
    edge_x = []
    edge_y = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
    
    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1, color='rgba(125,125,125,0.5)'),
        hoverinfo='none',
        mode='lines',
        showlegend=False
    )
    
    # Create the figure
    fig = go.Figure(data=[edge_trace] + node_trace)
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=title,
            x=0.5,
            font=dict(size=16)
        ),
        showlegend=True,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[ dict(
            text="Hover over nodes for details • Drag to pan • Scroll to zoom • Click legend to toggle",
            showarrow=False,
            xref="paper", yref="paper",
            x=0.005, y=-0.002,
            xanchor='left', yanchor='bottom',
            font=dict(color="gray", size=10)
        )],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='white',
        paper_bgcolor='white',
        height=600
    )
    
    return fig


@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_network_data_file(filename: str):
    """Load a single network data file on demand"""
    try:
        # Try multiple possible paths for the data files
        APP_DIR = Path(__file__).parent
        ROOT = APP_DIR.parent
        possible_paths = [
            # Streamlit Cloud paths
            Path("../data/processed"),
            Path("data/processed"),
            # Local development paths
            APP_DIR.parent / "data" / "processed",
            ROOT / "data" / "processed",
            Path.cwd() / "data" / "processed",
        ]
        
        data_dir = next((p for p in possible_paths if p.exists()), None)
        if data_dir is None:
            return pd.DataFrame()
        
        file_path = data_dir / filename
        if file_path.exists():
            try:
                df = pd.read_csv(file_path)
                # Limit to reasonable size for Streamlit Cloud
                if len(df) > 10000:
                    df = df.head(10000)
                return df
            except Exception as file_error:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()


def get_network_data():
    """Get network data with lazy loading"""
    return {
        'influence_nodes': load_network_data_file('influence_nodes.csv'),
        'influence_edges': load_network_data_file('influence_edges.csv'),
        'publisher_term_edges': load_network_data_file('publisher_term_edges.csv'),
        'community_summary': load_network_data_file('community_summary.csv'),
        'term_comparison': load_network_data_file('term_comparison.csv'),
        'top_terms_chunk': load_network_data_file('top_terms_chunk.csv'),
        'top_terms_global': load_network_data_file('top_terms_global.csv')
    }


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

