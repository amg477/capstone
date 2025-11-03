"""
Network Analysis Functions
Handles network analysis, graph building, and network visualization
"""

from typing import Dict, Tuple, List, Set, Iterable
import pandas as pd
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import plotly.graph_objects as go
from pathlib import Path
import streamlit as st


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

