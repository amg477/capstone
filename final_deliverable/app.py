# app.py — PolicyPath (pandas-only; ready for Streamlit Cloud)
from __future__ import annotations

# -------------------- MUST be first Streamlit call --------------------
import streamlit as st

# Add comprehensive startup error handling
try:
    st.set_page_config(page_title="PolicyPath", layout="wide",page_icon="🏛️")
except Exception as e:
    st.error(f"Configuration error: {e}")
    st.stop()

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
import re
from typing import Dict, Optional, Tuple, List, Set, Iterable
import pandas as pd
import plotly.graph_objects as go
import base64
from pathlib import Path
import networkx as nx
from networkx.algorithms.community import greedy_modularity_communities
import numpy as np
# Try to import streamlit_extras, fallback to custom styling if not available
try:
    from streamlit_extras.metric_cards import style_metric_cards
    STREAMLIT_EXTRAS_AVAILABLE = True
except ImportError:
    STREAMLIT_EXTRAS_AVAILABLE = False
    print("streamlit-extras not available, using custom metric styling")
    
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


# -------------------- CSS Injection --------------------
@st.cache_resource
def inject_css(path: str = "style.css"):
    """
    Load external CSS once (replaces all inline <style> blocks).
    Searches the app root, then final_deliverable/.
    """
    try:
        p = Path(path)
        if not p.exists():
            fallback = Path("final_deliverable/style.css")
            p = fallback if fallback.exists() else None
        if p:
            css_content = p.read_text(encoding='utf-8')
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return True
        else:
            st.warning("Custom style.css not found. Using Streamlit defaults.")
            return False
    except Exception as e:
        st.warning(f"Could not load CSS: {e}. Using Streamlit defaults.")
        return False

# ---- Brand defaults for charts (after set_page_config) ----
import plotly.express as px
import plotly.io as pio
import altair as alt

# Penta brand colors (matching CSS variables)
PENTA_COLORS = ["#12715D", "#4AB48E", "#142536", "#D4A115", "#2A9D8F", "#D94841"]
PENTA_PRIMARY = "#12715D"
PENTA_ACCENT = "#4AB48E" 
PENTA_DARK = "#142536"
PENTA_GOLD = "#D4A115"

# Create custom Plotly template
penta_template = {
    "layout": {
        "font": {"family": "Inter, sans-serif", "size": 12, "color": PENTA_DARK},
        "title": {"font": {"family": "Fraunces, serif", "size": 18, "color": PENTA_DARK}},
        "paper_bgcolor": "rgba(0,0,0,0)",  # Transparent background
        "plot_bgcolor": "rgba(0,0,0,0)",   # Transparent background
        "colorway": PENTA_COLORS,
        "xaxis": {
            "gridcolor": "#E5E7EB",
            "linecolor": "#D1D5DB",
            "tickcolor": "#9CA3AF"
        },
        "yaxis": {
            "gridcolor": "#E5E7EB", 
            "linecolor": "#D1D5DB",
            "tickcolor": "#9CA3AF"
        },
        "coloraxis": {"colorbar": {"tickcolor": PENTA_DARK}}
    }
}

# Register and set custom template
pio.templates["penta"] = penta_template
pio.templates.default = "penta"

# Set Plotly Express defaults
px.defaults.color_discrete_sequence = PENTA_COLORS
px.defaults.template = "penta"

# Altair configuration - simplified to avoid startup issues
try:
    alt.theme.enable("default")
    alt.renderers.set_embed_options(actions=False)
    alt.data_transformers.disable_max_rows()
except Exception as e:
    print(f"Altair configuration warning: {e}")

def create_penta_chart(fig, title=None, height=400):
    """Apply Penta branding to Plotly charts."""
    fig.update_layout(
        title={
            "text": title,
            "font": {"family": "Fraunces, serif", "size": 18, "color": PENTA_DARK},
            "x": 0.05,
            "xanchor": "left"
        },
        font={"family": "Inter, sans-serif", "size": 12, "color": PENTA_DARK},
        paper_bgcolor="rgba(0,0,0,0)",  # Transparent
        plot_bgcolor="rgba(0,0,0,0)",    # Transparent
        height=height,
        margin=dict(t=50, b=20, l=20, r=20),
        xaxis={
            "gridcolor": "#E5E7EB",
            "linecolor": "#D1D5DB",
            "tickcolor": "#9CA3AF",
            "title_font": {"family": "Inter, sans-serif", "color": PENTA_DARK}
        },
        yaxis={
            "gridcolor": "#E5E7EB",
            "linecolor": "#D1D5DB", 
            "tickcolor": "#9CA3AF",
            "title_font": {"family": "Inter, sans-serif", "color": PENTA_DARK}
        },
        legend={
            "font": {"family": "Inter, sans-serif", "color": PENTA_DARK},
            "bgcolor": "rgba(255,255,255,0.8)"
        }
    )
    return fig

def apply_penta_style():
    """Return Penta brand colors for consistent theming."""
    return {
        "primaryColor": "#12715D",
        "backgroundColor": "#F6F7F8", 
        "textColor": "#142536"
    }

# -------------------- App paths (define early; used by Debug/Logo) --------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# -------------------- Load Split Dataset Files --------------------
@st.cache_data
def load_combined_dataset() -> pd.DataFrame:
    """Load and combine a subset of split dataset files for memory efficiency."""
    # Try multiple possible paths for the data files
    possible_paths = [
        # Streamlit Cloud paths
        Path("data/split"),  # final_deliverable/data/split
        Path("../data/processed/split"),  # data/processed/split
        Path("data/processed/split"),  # data/processed/split
        # Local development paths
        APP_DIR / "data" / "split",
        APP_DIR.parent / "data" / "processed" / "split",
        Path.cwd() / "data" / "split",
        Path.cwd() / "data" / "processed" / "split",
    ]

    split_dir = next((p for p in possible_paths if p.exists()), None)
    if split_dir is None:
        print("Warning: No split data directory found")
        print(f"Searched paths: {[str(p) for p in possible_paths]}")
        return pd.DataFrame()

    print(f"Loading data from: {split_dir}")
    combined: List[pd.DataFrame] = []
    
    # Try to load files 1-2 first (smaller subset for Streamlit Cloud)
    for i in range(1, 3):  # Load only 2 files to reduce memory
        fp = split_dir / f"final_model_dataset_part_{i:03d}.csv"
        if fp.exists():
            try:
                # Try with pyarrow first, fallback to regular pandas
                try:
                    df = pd.read_csv(fp, dtype_backend="pyarrow")
                except Exception:
                    df = pd.read_csv(fp)
                
                # Keep top 95% by circulation (high-impact) and limit sample size
                if 'circulation_size' in df.columns:
                    df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce')
                    circulation_threshold = df['circulation_size'].quantile(0.05)  # Top 95%
                    df_filtered = df[df['circulation_size'] >= circulation_threshold]
                    # Further limit to 15k rows per file
                    if len(df_filtered) > 15000:
                        df_filtered = df_filtered.nlargest(15000, 'circulation_size')
                    combined.append(df_filtered)
                else:
                    # If no circulation_size, take a smaller sample
                    sample_size = min(5000, len(df))  # Limit to 5k rows max
                    combined.append(df.sample(n=sample_size))
                print(f"Loaded {fp.name}: {len(df)} rows")
            except Exception as e:
                print(f"Warning: Could not load {fp.name}: {e}")
                pass

    if not combined:
        print("Warning: No data files could be loaded")
        return pd.DataFrame()

    final_df = pd.concat(combined, ignore_index=True).drop_duplicates()
    files_count = len(combined)
    combined.clear()

    print(f"Successfully loaded {len(final_df)} rows from {files_count} files")
    st.session_state.dataset_info = {'rows': len(final_df), 'files': files_count}
    return final_df

@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_dataset() -> pd.DataFrame:
    return load_combined_dataset()

# -------------------- Helpers --------------------

def add_to_recent_searches(term: str):
    if term and term not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, term)
        st.session_state.recent_searches = st.session_state.recent_searches[:10]

def shorten(label: str, max_len: int = 28) -> str:
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

def export_data_button(df: pd.DataFrame, filename: str, fmt: str = "csv"):
    if df is None or df.empty:
        st.info("No data to export.")
        return
    if fmt == "csv":
        st.download_button(
            label=f"Download {filename}.csv",
            data=df.to_csv(index=False),
            file_name=f"{filename}.csv",
            mime="text/csv",
        )

# -------------------- Content Network Analysis Functions --------------------
def bigram_strings(tokens: List[str]) -> set[str]:
    """Extract bigrams from token list."""
    if len(tokens) < 2:
        return set()
    return {f"{a} {b}" for a, b in zip(tokens[:-1], tokens[1:])}

def prepare_whitelist_sets(whitelist_terms: Iterable[str], term_weight_tbl: pd.DataFrame) -> Tuple[set[str], set[str], Dict[str, float]]:
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

def best_term_from_tokens(tokens: List[str], uni_whitelist: set[str], bi_whitelist: set[str],
                          tw_map: Dict[str, float], min_weight: float = 0.0) -> Tuple[str|None, float]:
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
        use[c] = use[c].fillna("").astype(str).str.lower()
    
    if "max_term_credit" not in df.columns:
        df = df.assign(max_term_credit=0.0)
    if "vipr_weight" not in df.columns:
        df = df.assign(vipr_weight=1.0)

    # Build edges
    edges: Dict[Tuple[str,str], float] = {}
    for idx, r in use.iterrows():
        pub = r[publisher_col]
        tokens = (r["headline"] + " " + r["body"]).split()
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

def filter_content_edges(
    edges: pd.DataFrame,
    top_publishers: int = 30,
    top_terms: int = 30,
    generic_terms: Iterable[str] = (),
    edge_percentile_cutoff: float = 0.75
) -> pd.DataFrame:
    """Filter content network edges to strongest nodes."""
    if edges.empty:
        return edges
    e = edges.copy()
    if generic_terms:
        e = e[~e["term"].isin(set(map(str.lower, generic_terms)))].reset_index(drop=True)
    pub_strength  = e.groupby("publisher")["weight"].sum().sort_values(ascending=False)
    term_strength = e.groupby("term")["weight"].sum().sort_values(ascending=False)
    keep_pubs  = set(pub_strength.head(top_publishers).index)
    keep_terms = set(term_strength.head(top_terms).index)
    e = e[e["publisher"].isin(keep_pubs) & e["term"].isin(keep_terms)].reset_index(drop=True)
    if e.empty:
        return e
    w_cut = e["weight"].quantile(edge_percentile_cutoff)
    return e[e["weight"] >= w_cut].reset_index(drop=True)

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
            neighbor_types = [G.nodes[n].get("ntype", "unknown") for n in neighbors]
            
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
    edge_info = []
    
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        
        weight = G[edge[0]][edge[1]].get("weight", 0)
        edge_info.append(f"{edge[0]} ↔ {edge[1]}<br>Weight: {weight:.3f}")
    
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

def create_network_statistics_dashboard(G: nx.Graph, edges_df: pd.DataFrame, node2c: Dict[str, int]):
    """Create interactive network statistics dashboard."""
    if G.number_of_nodes() == 0:
        return None, None, None
    
    # Basic network metrics
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    num_communities = len(set(node2c.values()))
    avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0
    
    # Node type distribution
    pubs = [n for n, d in G.nodes(data=True) if d.get("ntype") == "publisher"]
    terms = [n for n, d in G.nodes(data=True) if d.get("ntype") == "term"]
    
    # Top nodes by strength
    strength = {n: sum(G[n][nbr].get("weight", 0.0) for nbr in G.neighbors(n)) for n in G.nodes()}
    top_nodes = sorted(strength.items(), key=lambda x: x[1], reverse=True)[:10]
    
    # Community analysis
    community_stats = []
    for comm_id in sorted(set(node2c.values())):
        comm_nodes = [n for n, c in node2c.items() if c == comm_id]
        comm_pubs = [n for n in comm_nodes if G.nodes[n].get("ntype") == "publisher"]
        comm_terms = [n for n in comm_nodes if G.nodes[n].get("ntype") == "term"]
        
        comm_edges = edges_df[
            edges_df["publisher"].isin(comm_pubs) & 
            edges_df["term"].isin(comm_terms)
        ]
        
        community_stats.append({
            "Community": comm_id,
            "Nodes": len(comm_nodes),
            "Publishers": len(comm_pubs),
            "Terms": len(comm_terms),
            "Edges": len(comm_edges),
            "Total Weight": comm_edges["weight"].sum() if not comm_edges.empty else 0
        })
    
    comm_df = pd.DataFrame(community_stats)
    
    # Create network metrics chart
    metrics_data = {
        "Metric": ["Nodes", "Edges", "Communities", "Avg Degree"],
        "Value": [num_nodes, num_edges, num_communities, round(avg_degree, 2)]
    }
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create node strength distribution chart
    strength_data = pd.DataFrame([
        {"Node": node, "Strength": strength_val, "Type": G.nodes[node].get("ntype", "unknown")}
        for node, strength_val in top_nodes
    ])
    
    return metrics_df, comm_df, strength_data

def summarize_content_communities(
    G: nx.Graph, edges_df: pd.DataFrame, node2c: Dict[str, int],
    top_k: int = 5, top_edges: int = 1
) -> pd.DataFrame:
    """Summarize communities with examples."""
    nodes_in_G = set(G.nodes())
    e = edges_df[
        edges_df["publisher"].isin(nodes_in_G) & edges_df["term"].isin(nodes_in_G)
    ].copy()

    rows = []
    for cid in sorted(set(node2c.values())):
        nodes_c  = {n for n, k in node2c.items() if k == cid}
        pubs_c   = {n for n in nodes_c if G.nodes[n].get("ntype") == "publisher"}
        terms_c  = {n for n in nodes_c if G.nodes[n].get("ntype") == "term"}
        ec = e[e["publisher"].isin(pubs_c) & e["term"].isin(terms_c)]
        if ec.empty: continue
        term_strength = ec.groupby("term")["weight"].sum().sort_values(ascending=False)
        pub_strength  = ec.groupby("publisher")["weight"].sum().sort_values(ascending=False)
        top_edge_rows = (ec.sort_values("weight", ascending=False)
                           .head(top_edges)
                           .assign(example=lambda d: d["publisher"] + " — " + d["term"] +
                                                   " (" + d["weight"].round(1).astype(str) + ")"))
        rows.append({
            "Community": cid,
            "Top Terms": ", ".join(term_strength.head(top_k).index),
            "Top Publishers": ", ".join(pub_strength.head(top_k).index),
            "Representative Edge(s)": "; ".join(top_edge_rows["example"].tolist()),
            "Edges in Community": int(len(ec)),
            "Total Weight": float(ec["weight"].sum().round(1))
        })
    return (pd.DataFrame(rows)
              .sort_values(["Total Weight","Edges in Community"], ascending=False)
              .reset_index(drop=True))

# -------------------- Session State --------------------
def init_session_state():
    if 'dark_mode' not in st.session_state: st.session_state.dark_mode = False
    if 'recent_searches' not in st.session_state: st.session_state.recent_searches = []
    if 'favorites' not in st.session_state: st.session_state.favorites = []

init_session_state()

# -------------------- Data Globals & Loader --------------------
DATA_DIR = (ROOT / "data").resolve()
ATTR_NAME = "attribution_all_scored.csv"
LOGO_FALLBACK = ROOT / "final_deliverable" / "penta_logo.png"  # adjust if needed

df_main: Optional[pd.DataFrame] = None
df_attr: Optional[pd.DataFrame] = None
COLUMNS: Set[str] = set()

def _find_first_existing(*names: str) -> Optional[Path]:
    candidates = [
        # Streamlit Cloud paths
        Path("../data/processed"),  # data/processed
        Path("data/processed"),    # data/processed
        # Local development paths
        ROOT / "data" / "processed",
        ROOT / "data",
        APP_DIR / "data",
        APP_DIR.parent / "data",
        Path.cwd() / "data",
        Path.cwd() / "data" / "processed",
    ]
    for d in candidates:
        for nm in names:
            p = d / nm
            if p.exists():
                return p
    return None

@st.cache_resource(ttl=3600)  # Cache for 1 hour
def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    global df_main, df_attr, COLUMNS
    if df_main is None:
        try:
            # Load main dataset with better error handling
            df_main = get_dataset()
            if df_main is None or df_main.empty:
                print("Warning: No main dataset loaded, using empty DataFrame")
                df_main = pd.DataFrame()
            COLUMNS = set(df_main.columns) if not df_main.empty else set()

            # Load attribution data with better error handling
            attr_csv = _find_first_existing(ATTR_NAME, "attribution_all_scored_sample.csv")
            if attr_csv:
                try:
                    # Try with pyarrow first, fallback to regular pandas
                    try:
                        df_attr = pd.read_csv(attr_csv, dtype_backend="pyarrow")
                    except Exception:
                        df_attr = pd.read_csv(attr_csv)
                    print(f"Successfully loaded attribution data from {attr_csv}")
                except Exception as e:
                    print(f"Warning: Could not load attribution data from {attr_csv}: {e}")
                    df_attr = pd.DataFrame()
            else:
                print("Warning: No attribution data file found")
                print(f"Searched for: {ATTR_NAME}, attribution_all_scored_sample.csv")
                df_attr = pd.DataFrame()
        except Exception as e:
            print(f"Error in get_data(): {e}")
            import traceback
            print(traceback.format_exc())
            df_main, df_attr, COLUMNS = pd.DataFrame(), pd.DataFrame(), set()
    return df_main, df_attr, COLUMNS

# -------------------- Header / Logo --------------------
def render_header():
    logo_path = None
    for p in [
        ROOT / "final_deliverable" / "penta_logo.png",
        ROOT / "data" / "penta_logo.png",  # optional extra fallback
        LOGO_FALLBACK,
    ]:
        if p.exists():
            logo_path = p
            break

    if logo_path and logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <div class="header-bar">
                <img src="data:image/png;base64,{logo_b64}" class="penta-logo" alt="Penta logo"/>
                <div class="header-title">
                    <h1>PolicyPath 🏛️</h1>
                    <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
                </div>
                <div class="header-spacer"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="header-bar">
                <div class="header-title">
                    <h1>PolicyPath 🏛️</h1>
                    <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
                </div>
                <div class="header-spacer"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------- Main App --------------------
def main():
    """Main application function with error handling."""
    # Load CSS after Streamlit is fully initialized
    inject_css()
    
    render_header()

    # Create tabs outside the expander
    tab1, tab2, tab3, tab4 = st.tabs(["PolicyPath", "Pulse", "Paths", "People"])

    return tab1, tab2, tab3, tab4

# Run main app
try:
    tab1, tab2, tab3, tab4 = main()
except Exception as e:
    handle_error(e, "in main() function")

if tab1 is None:
    st.stop()

with tab1:
    _ = apply_penta_style()  # optional; sets Altair defaults

    st.subheader("Welcome to PolicyPath")

    st.markdown("""
    **PolicyPath** helps you trace how healthcare policy narratives move through the media ecosystem — 
    identifying the publications, authors, and topics driving the conversation.

    ### How to Use PolicyPath

    **1. Pulse (Analytics Dashboard)**  
    Monitor the pulse of healthcare policy influence.  
    Use dynamic filters to explore KPIs such as reach, sentiment, and influence across publications, authors, and channels.  
    Includes charts, Sankey flows, and data export options.

    **2. Paths (Attribution Analysis)**  
    Dive into how influence is distributed.  
    Search for specific publications, authors, or terms, and compare their relative credit shares and circulation impact.  
    Ideal for identifying high-impact sources and emerging policy topics.

    **3. People (Network Explorer)**  
    Visualize the relationships between publishers, terms, and influencers.  
    Toggle between *Influence Networks* and *Publisher-Term Networks* to understand how ideas propagate through different communities.

    ---

    *Data:* All results are filtered to high-impact observations (top 90% by circulation).

    *Built by Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammad Waqas, Mark Saba, Posy Olivetti*
""")


with tab2:
    _ = apply_penta_style()  # optional; sets Altair defaults

    # Load data for metrics
    df_main, df_attr, COLUMNS = get_data()
    
    # Debug information
    st.write("### Debug Information")
    st.write(f"Main dataset rows: {len(df_main)}")
    st.write(f"Attribution dataset rows: {len(df_attr)}")
    st.write(f"Available columns: {list(COLUMNS)[:10]}...")  # Show first 10 columns
    
    # Check if we have any data
    if df_main.empty and df_attr.empty:
        st.error("⚠️ No data could be loaded. This might be due to:")
        st.markdown("""
        - Data files not found in expected locations
        - Memory limitations on Streamlit Cloud
        - File format issues
        
        **Troubleshooting steps:**
        1. Check that data files exist in the repository
        2. Verify file paths are correct
        3. Try reducing the dataset size
        """)
        st.stop()

    # PolicyPath Metrics Dashboard
    if not df_main.empty:
        st.markdown("### PolicyPath Pulse")
        st.markdown("Monitor the pulse of healthcare policy influence with interactive KPIs and charts.")

        # Calculate key metrics
        total_pubs = df_main['publication_name'].nunique() if 'publication_name' in df_main.columns else 0
        total_authors = df_main['author_name'].nunique() if 'author_name' in df_main.columns else 0
        total_articles = len(df_main)
        avg_circulation = df_main['circulation_size'].mean() if 'circulation_size' in df_main.columns else 0

        # Attribution metrics if available
        if not df_attr.empty and 'credit_share' in df_attr.columns:
            avg_influence = df_attr['credit_share'].mean()
            top_influence = df_attr['credit_share'].max()
        else:
            avg_influence = 0
            top_influence = 0

        # Row 1: Content Summary Metrics
        metric1, metric2, metric3, metric4, metric5, metric6  = st.columns(6)
        with metric1:
            st.metric(
                label="Total Publications",
                value=f"{total_pubs:,}"
            )
        with metric2:
            st.metric(
                label="Total Authors", 
                value=f"{total_authors:,}"
            )
        with metric3:
            st.metric(
                label="Total Articles",
                value=f"{total_articles:,}"
            )

        with metric4:
            st.metric(
                label="Avg Circulation",
                value=f"{avg_circulation:,.0f}"
            )
        with metric5:
            st.metric(
                label="Avg Influence",
                value=f"{avg_influence:#0.1%}"
            )
        with metric6:
            st.metric(
                label="Peak Influence",
                value=f"{top_influence:#0.1%}"
            )
        
        # Apply metric card styling
        style_metric_cards(
            background_color="#fafafa", 
            border_color="#fafafa", 
            border_left_color="#12715D"
        )

    if df_main.empty:
        st.info("No data loaded.")
    else:
        cols = df_main.columns.tolist()

        # Dynamic Filters - 2 rows, 3 columns
        st.markdown("### Dynamic Filters")
        
        # Row 1: Date, Publications, Channels
        c1, c2, c3 = st.columns(3)

        with c1:
            date_col = next((c for c in cols if any(k in c.lower() for k in ["date", "time", "ts"])), None)
            date_range = None
            if date_col:
                try:
                    df_main[date_col] = pd.to_datetime(df_main[date_col], errors="coerce")
                    min_d, max_d = df_main[date_col].min(), df_main[date_col].max()
                    if pd.notna(min_d) and pd.notna(max_d):
                        date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()),
                                                   min_value=min_d.date(), max_value=max_d.date())
                except Exception:
                    date_range = None

        with c2:
            pub_cols = [c for c in cols if "publication" in c.lower()]
            if pub_cols:
                pub_col = pub_cols[0]
                pubs = pd.Series(df_main[pub_col]).dropna().unique().tolist()[:50]
                sel_pubs = st.multiselect("Publications", pubs)
            else:
                sel_pubs = []

        with c3:
            channel_cols = [c for c in cols if "channel" in c.lower()]
            if channel_cols:
                channel_col = channel_cols[0]
                channels = pd.Series(df_main[channel_col]).dropna().unique().tolist()[:50]
                sel_channels = st.multiselect("Channels", channels)
            else:
                sel_channels = []

        # Row 2: Authors, Source Types, Sentiment Band
        c4, c5, c6 = st.columns(3)
        
        with c4:
            author_cols = [c for c in cols if "author" in c.lower()]
            if author_cols:
                author_col = author_cols[0]
                authors = pd.Series(df_main[author_col]).dropna().unique().tolist()[:50]
                sel_authors = st.multiselect("Authors", authors)
            else:
                sel_authors = []

        with c5:
            source_cols = [c for c in cols if "source" in c.lower()]
            if source_cols:
                source_col = source_cols[0]
                sources = pd.Series(df_main[source_col]).dropna().unique().tolist()[:20]
                sel_sources = st.multiselect("Source Types", sources)
            else:
                sel_sources = []

        with c6:
            sentiment_cols = [c for c in cols if "sentiment_band" in c.lower()]
            if sentiment_cols:
                sentiment_col = sentiment_cols[0]
                sentiments = pd.Series(df_main[sentiment_col]).dropna().unique().tolist()
                sel_sentiments = st.multiselect("Sentiment Band", sentiments)
            else:
                sel_sentiments = []

        # Apply filters (pandas)
        filtered_df = df_main.copy()
        if date_range and date_col and len(date_range) == 2:
            start_d, end_d = date_range
            filtered_df = filtered_df[
                (filtered_df[date_col] >= pd.Timestamp(start_d)) &
                (filtered_df[date_col] <= pd.Timestamp(end_d))
            ]
        if sel_pubs and pub_cols:
            filtered_df = filtered_df[filtered_df[pub_cols[0]].isin(sel_pubs)]
        if sel_channels and channel_cols:
            filtered_df = filtered_df[filtered_df[channel_cols[0]].isin(sel_channels)]
        if sel_authors and author_cols:
            filtered_df = filtered_df[filtered_df[author_cols[0]].isin(sel_authors)]
        if sel_sources and source_cols:
            filtered_df = filtered_df[filtered_df[source_cols[0]].isin(sel_sources)]
        if sel_sentiments and sentiment_cols:
            filtered_df = filtered_df[filtered_df[sentiment_cols[0]].isin(sel_sentiments)]

        # KPIs
        try:
            total_pubs = filtered_df[pub_cols[0]].nunique() if pub_cols else 0
            uniq_sources = filtered_df[source_cols[0]].nunique() if source_cols else 0
            uniq_authors = filtered_df[author_cols[0]].nunique() if author_cols else 0
        except Exception:
            total_pubs = uniq_sources = uniq_authors = 0

        infl_col, avg_infl = None, None
        for c in ["pub_credit_share", "max_term_credit", "credit_share"]:
            if c in filtered_df.columns:
                infl_col = c
                avg_infl = filtered_df[c].mean()
                break
        if avg_infl is None and not df_attr.empty and "credit_share" in df_attr.columns:
            avg_infl = df_attr["credit_share"].mean()
            st.info("Using attribution dataset for influence metrics")

        # Charts
        cat_cols = [c for c in ["publication_name","source_name","channel_name","author_name","topic","sentiment_band"] if c in filtered_df.columns]
        if not cat_cols:
            st.info("No categorical columns to group by.")
        else:
            dim = st.selectbox("Group charts by", cat_cols, index=0)
            circ_col = next((c for c in ["circulation","circulation_size","reach","impressions","audience"] if c in filtered_df.columns), None)

            if not filtered_df.empty and dim in filtered_df.columns:
                # Memory-efficient sampling for Streamlit Cloud
                sample_size = min(10000, len(filtered_df))  # Reduced from 50k to 10k
                if len(filtered_df) > sample_size:
                    # Use stratified sampling for better representation
                    if "circulation_size" in filtered_df:
                        # Sample based on circulation size (high-impact first)
                        filtered_df = filtered_df.nlargest(sample_size, 'circulation_size')
                    else:
                        # Random sampling if no circulation data
                        filtered_df = filtered_df.sample(n=sample_size, random_state=42)

                agg_dict: Dict[str, str] = {filtered_df.columns[0]: "count"}
                if infl_col and infl_col in filtered_df: agg_dict[infl_col] = "mean"
                if circ_col and circ_col in filtered_df: agg_dict[circ_col] = "sum"

                agg = (
                    filtered_df
                    .groupby(dim)
                    .agg(agg_dict)
                    .reset_index()
                    .rename(columns={filtered_df.columns[0]: "n", dim: "dim"})
                )
                if infl_col and infl_col in agg: agg = agg.rename(columns={infl_col: "avg_influence"})
                if circ_col and circ_col in agg: agg = agg.rename(columns={circ_col: "total_metric"})
                agg = agg[agg["dim"].notna()]
            else:
                agg = pd.DataFrame(columns=["dim", "avg_influence", "n", "total_metric"])

            # Top N Slider
            top_n = st.slider("Top N", 5, 50, 20, 1)
            # Simple filtered metrics with centered styling and green outline
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                    <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{total_pubs:,}</div>
                    <div style="font-size: 14px; color: #666;">Publications</div>
                </div>
                """, unsafe_allow_html=True)
            with col2:
                avg_infl_display = f"{avg_infl:.3f}" if avg_infl is not None else "n/a"
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                    <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{avg_infl_display}</div>
                    <div style="font-size: 14px; color: #666;">Avg Influence</div>
                </div>
                """, unsafe_allow_html=True)
            with col3:
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                    <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{uniq_sources:,}</div>
                    <div style="font-size: 14px; color: #666;">Sources</div>
                </div>
                """, unsafe_allow_html=True)
            with col4:
                st.markdown(f"""
                <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                    <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{uniq_authors:,}</div>
                    <div style="font-size: 14px; color: #666;">Authors</div>
                </div>
                """, unsafe_allow_html=True)
            
            
            # 2x2 Grid of Policy Analytics
            st.markdown("### Policy Influence Analytics")
            
            if not filtered_df.empty:
                # Row 1
                col1, col2 = st.columns(2)
                
                with col1:
                    # 1. Average Influence by Category (original bar chart)
                    if not agg.empty and "avg_influence" in agg.columns:
                        avg_chart = alt.Chart(agg.sort_values("avg_influence", ascending=False).head(10)).mark_bar().encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("avg_influence:Q", title="Avg influence"),
                        )
                        avg_chart = avg_chart.properties(height=300, title="Average Influence by Category")
                        st.altair_chart(avg_chart.configure_view(strokeWidth=0).configure_title(
                            fontSize=16,
                            offset=10
                        ), use_container_width=True)
                    else:
                        st.info("No influence data available")
                
                with col2:
                    # 2. Pie Chart (original pie chart)
                    if not agg.empty and "avg_influence" in agg.columns:
                        pie_df = agg.sort_values("avg_influence", ascending=False).head(8)
                        fig_pie = go.Figure(data=[go.Pie(
                            labels=pie_df["dim"],
                            values=pie_df["avg_influence"],
                            textinfo="percent+label"
                        )])
                        fig_pie.update_layout(
                            title="Influence Distribution",
                            height=300,
                            showlegend=True,
                            margin=dict(t=30, b=10, l=10, r=10),
                            title_font_size=16,
                            modebar_remove=['zoom', 'pan', 'select', 'lasso', 'zoomin', 'zoomout', 'autoscale', 'reset', 'toImage']
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
            else:
                st.info("No influence data available")
                
                # Row 2
                if not filtered_df.empty:
                    col3, col4 = st.columns(2)
                    
                    with col3:
                        # 3. Sentiment Analysis
                        if 'sentiment_band' in filtered_df.columns:
                            sentiment_counts = filtered_df['sentiment_band'].value_counts()
                            fig_sentiment = go.Figure(data=[
                                go.Bar(
                                    x=sentiment_counts.index, 
                                    y=sentiment_counts.values,
                                    marker_color=[PENTA_COLORS[3], PENTA_COLORS[4], PENTA_COLORS[1]][:len(sentiment_counts)]  # Penta colors
                                )
                            ])
                            
                            # Apply Penta branding
                            fig_sentiment = create_penta_chart(
                                fig_sentiment, 
                                title="Sentiment Distribution",
                                height=300
                            )
                            
                            fig_sentiment.update_layout(
                                xaxis_title="Sentiment",
                                yaxis_title="Article Count",
                                showlegend=False,
                                margin=dict(t=30, b=10, l=10, r=10),
                                modebar_remove=['zoom', 'pan', 'select', 'lasso', 'zoomin', 'zoomout', 'autoscale', 'reset', 'toImage']
                            )
                            st.plotly_chart(fig_sentiment, use_container_width=True)
                        else:
                            st.info("No sentiment data available")
                
                    with col4:
                        # 4. Publication Reach Analysis
                        if 'publication_name' in filtered_df.columns and 'circulation_size' in filtered_df.columns:
                            pub_reach = filtered_df.groupby('publication_name')['circulation_size'].mean().nlargest(8).reset_index()
                            fig_reach = go.Figure(data=[
                                go.Bar(
                                    x=pub_reach['publication_name'],
                                    y=pub_reach['circulation_size'],
                                    marker_color=PENTA_COLORS[1]  # Penta green
                                )
                            ])
                            
                            # Apply Penta branding
                            fig_reach = create_penta_chart(
                                fig_reach, 
                                title="Top Publications by Reach",
                                height=300
                            )
                            
                            fig_reach.update_layout(
                                xaxis_title="Publication",
                                yaxis_title="Average Circulation",
                                showlegend=False,
                                margin=dict(t=30, b=10, l=10, r=10),
                                modebar_remove=['zoom', 'pan', 'select', 'lasso', 'zoomin', 'zoomout', 'autoscale', 'reset', 'toImage']
                            )
                            st.plotly_chart(fig_reach, use_container_width=True)
                        else:
                            st.info("No publication/reach data available")
                else:
                    st.info("No data for current filters.")

            # Sankey Chart - Third Row
        if cat_cols:
            
            left, right = st.columns(2)
            src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
            tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols)-1), key="sank_tgt")

            c1, c2, c3, c4 = st.columns(4)
            top_sources = c1.slider("Top Sources", 3, 20, 8, 1)
            top_targets = c2.slider("Top Targets", 2, 15, 5, 1)
            max_links = c3.slider("Max Links", 5, 50, 20, 5)
            bucket_other = c4.checkbox("Bucket 'Other'", value=True)

            if src != tgt and not filtered_df.empty:
                src_counts = filtered_df[src].value_counts().head(int(top_sources))
                tgt_counts = filtered_df[tgt].value_counts().head(int(top_targets))
                keep_s = set(src_counts.index.dropna().astype(str))
                keep_t = set(tgt_counts.index.dropna().astype(str))

                nt = filtered_df[[src, tgt]].dropna().copy()
                nt["s"] = nt[src].apply(lambda x: x if str(x) in keep_s else "Other")
                nt["t"] = nt[tgt].apply(lambda x: x if str(x) in keep_t else "Other")
                sdata = (
                    nt.groupby(["s", "t"]).size().reset_index(name="v")
                    .sort_values("v", ascending=False).head(int(max_links))
                )
                if not bucket_other:
                    sdata = sdata[(sdata["s"].isin(keep_s)) & (sdata["t"].isin(keep_t))]

                if not sdata.empty:
                    # Filter out very small flows to reduce clutter
                    min_flow_threshold = sdata["v"].quantile(0.1)  # Remove bottom 10% of flows
                    sdata_clean = sdata[sdata["v"] >= min_flow_threshold].copy()
                    
                    # Remove "Other" if it's too dominant (more than 50% of total flow)
                    total_flow = sdata_clean["v"].sum()
                    other_flow = sdata_clean[sdata_clean["s"] == "Other"]["v"].sum()
                    if other_flow / total_flow > 0.5:
                        sdata_clean = sdata_clean[sdata_clean["s"] != "Other"]
                    
                    if not sdata_clean.empty:
                        nodes = pd.Series(pd.concat([sdata_clean["s"], sdata_clean["t"]])).astype(str).unique().tolist()
                        labels_short = [shorten(x, max_len=20) for x in nodes]  # Shorter labels
                        idx = {n: i for i, n in enumerate(nodes)}

                        # Use Penta brand colors for nodes
                        colors = PENTA_COLORS + ["#7F9EA3", "#A8B5C0", "#C4D1D9", "#E0E7ED"]
                        node_colors = [colors[i % len(colors)] for i in range(len(nodes))]

                        fig = go.Figure(go.Sankey(
                            arrangement="snap",
                            node=dict(
                                label=labels_short,
                                pad=35, 
                                thickness=30,
                                line=dict(width=1, color="#F8F8F8"),
                                color=node_colors
                            ),
                            link=dict(
                                source=[idx[s] for s in sdata_clean["s"]],
                                target=[idx[t] for t in sdata_clean["t"]],
                                value=sdata_clean["v"],
                                color=[node_colors[idx[s]] for s in sdata_clean["s"]],  # Match source node colors
                                hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>Count: %{value:,}<extra></extra>",
                            ),
                        ))
                        
                        # Apply Penta branding
                        fig = create_penta_chart(
                            fig, 
                            title=f"Flow Analysis: {src.replace('_', ' ').title()} → {tgt.replace('_', ' ').title()}",
                            height=600
                        )
                        
                        # Additional Sankey-specific styling
                        fig.update_layout(
                            font=dict(family="Inter, sans-serif", size=12, color=PENTA_DARK),
                            title_font=dict(family="Fraunces, serif", size=18, color=PENTA_DARK),
                            paper_bgcolor="rgba(0,0,0,0)",
                            plot_bgcolor="rgba(0,0,0,0)",
                            margin=dict(t=80, b=40, l=40, r=40)
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("After filtering, not enough significant flows to display.")
                else:
                    st.info("Not enough data for Sankey with these fields.")
            else:
                st.info("Choose different fields for source and target (and ensure data is filtered).")

with tab3:
    st.subheader("Attribution Analysis")
    st.markdown("Discover the influence pathways in healthcare policy and understand impact patterns.")

    df_main, df_attr, COLUMNS = get_data()
    available_cols = df_main.columns.tolist() if not df_main.empty else []
    
    # Debug information for attribution tab
    if df_main.empty:
        st.warning("⚠️ No main dataset loaded. Check the Pulse tab for debug information.")
    
    # Attribution Metrics Dashboard
    if not df_main.empty:
        
        # Calculate key metrics for attribution
        total_pubs = df_main['publication_name'].nunique() if 'publication_name' in df_main.columns else 0
        total_authors = df_main['author_name'].nunique() if 'author_name' in df_main.columns else 0
        total_articles = len(df_main)
        
        # Attribution-specific metrics
        if not df_attr.empty and 'credit_share' in df_attr.columns:
            avg_influence = df_attr['credit_share'].mean()
            top_influence = df_attr['credit_share'].max()
            total_attributions = len(df_attr)
        else:
            avg_influence = 0
            top_influence = 0
            total_attributions = 0
        
        # Selector for attribution analysis
        # Get available item columns
        item_cols = [c for c in df_main.columns if any(k in c.lower() for k in ["publication", "author", "channel", "publisher", "source"])]
        if item_cols:
            selected_item_col = st.selectbox(
                "Select Item Type to Analyze",
                item_cols,
                help="Choose which type of items to analyze for attribution"
            )
        else:
            selected_item_col = None
        
        # Create the two bar charts
        col1, col2 = st.columns(2)
        
        with col1:
            # Top Attribution Items by Credit Share
            if selected_item_col and selected_item_col in df_main.columns:
                item_attribution = df_main.groupby(selected_item_col).agg({
                    'pub_credit_share': 'mean',
                    'circulation_size': 'mean'
                }).reset_index()
                item_attribution = item_attribution.dropna()
                
                if not item_attribution.empty:
                    top_items_credit = item_attribution.nlargest(10, 'pub_credit_share').sort_values('pub_credit_share', ascending=True)
                    
                    fig_credit = go.Figure(data=[
                        go.Bar(
                            x=top_items_credit['pub_credit_share'],
                            y=top_items_credit[selected_item_col],
                            orientation='h',
                            marker_color=PENTA_COLORS[0],  # Penta primary
                            text=top_items_credit['pub_credit_share'].round(3),
                            textposition='auto'
                        )
                    ])
                    
                    # Apply Penta branding
                    fig_credit = create_penta_chart(
                        fig_credit, 
                        title=f"Top {selected_item_col.title()} by Credit Share",
                        height=400
                    )
                    
                    fig_credit.update_layout(
                        xaxis_title="Average Credit Share",
                        yaxis_title=selected_item_col.title(),
                        margin=dict(t=30, b=10, l=10, r=10),
                        modebar_remove=['zoom', 'pan', 'select', 'lasso', 'zoomin', 'zoomout', 'autoscale', 'reset', 'toImage']
                    )
                    st.plotly_chart(fig_credit, use_container_width=True)
                else:
                    st.info(f"No {selected_item_col} attribution data available")
            else:
                st.info("Select an item type to see attribution analysis")
        
        with col2:
            # Top Attribution Items by Importance (Circulation)
            if selected_item_col and selected_item_col in df_main.columns:
                item_attribution = df_main.groupby(selected_item_col).agg({
                    'pub_credit_share': 'mean',
                    'circulation_size': 'mean'
                }).reset_index()
                item_attribution = item_attribution.dropna()
                
                if not item_attribution.empty:
                    top_items_importance = item_attribution.nlargest(10, 'circulation_size').sort_values('circulation_size', ascending=True)
                    
                    fig_importance = go.Figure(data=[
                        go.Bar(
                            x=top_items_importance['circulation_size'],
                            y=top_items_importance[selected_item_col],
                            orientation='h',
                            marker_color=PENTA_COLORS[1],  # Penta green
                            text=top_items_importance['circulation_size'].round(0),
                            textposition='auto'
                        )
                    ])
                    # Apply Penta branding
                    fig_importance = create_penta_chart(
                        fig_importance, 
                        title=f"Top {selected_item_col.title()} by Circulation",
                        height=400
                    )
                    
                    fig_importance.update_layout(
                        xaxis_title="Average Circulation Size",
                        yaxis_title=selected_item_col.title(),
                        margin=dict(t=30, b=10, l=10, r=10),
                        modebar_remove=['zoom', 'pan', 'select', 'lasso', 'zoomin', 'zoomout', 'autoscale', 'reset', 'toImage']
                    )
                    st.plotly_chart(fig_importance, use_container_width=True)
                else:
                    st.info(f"No {selected_item_col} attribution data available")
            else:
                st.info("Select an item type to see attribution analysis")
        
        st.markdown("---")
    
    st.markdown("### Search for Specific Attribution Items or Terms")
    col1, col2 = st.columns([2, 1])
    with col1:
        lookup_type = st.radio("Search Type", ["Item Attribution", "Term Attribution"], horizontal=True)
    with col2:
        st.empty()  # Space for future elements

    # Recent searches
    if st.session_state.recent_searches:
        with st.expander("Recent Searches"):
            for s in st.session_state.recent_searches[:5]:
                if st.button(f"{s}", key=f"recent_{s}"):
                    st.session_state.current_search = s
                    # Don't use st.rerun() here - let the search happen naturally

    if lookup_type == "Item Attribution":
        if not available_cols:
            st.warning("No searchable columns found.")
        else:
            search_cols = [c for c in available_cols if any(k in c.lower() for k in ["publication", "author", "channel", "publisher"])]
            if not search_cols:
                st.warning("No item columns found.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    sel_col = st.selectbox("Search by", search_cols)
                with c2:
                    # Get unique values for searchable selectbox
                    if sel_col in df_main:
                        unique_values = df_main[sel_col].dropna().unique().tolist()
                        # Sort for better user experience
                        unique_values = sorted(unique_values)[:200]  # Limit to 200 for performance
                        search_term = st.selectbox(
                            f"Search {sel_col}",
                            options=[""] + unique_values,
                            key=f"search_{sel_col}",
                            help=f"Type to search and select from {sel_col} options"
                        )
                    else:
                        search_term = ""

                if f"selected_{sel_col}" in st.session_state:
                    search_term = st.session_state[f"selected_{sel_col}"]
                    st.success(f"Selected: {search_term}")
                    if st.button("Clear Selection", key=f"clear_{sel_col}"):
                        del st.session_state[f"selected_{sel_col}"]
                        # Don't use st.rerun() here - let the clearing happen naturally

                if search_term and sel_col in df_main:
                    add_to_recent_searches(f"{sel_col}: {search_term}")
                    try:
                        s = df_main[sel_col].astype("string[pyarrow]", errors="ignore")
                        matches = s.fillna("").str.contains(search_term, case=False, na=False)
                        options = s[matches].dropna().drop_duplicates().head(20).tolist()
                        if options:
                            st.success(f"Found {len(options)} matches for '{search_term}'")
                            selected_item = st.selectbox("Select item", options, key=f"select_{sel_col}")
                            if selected_item:
                                item_rows = df_main[s.fillna("") == str(selected_item)].head(100)
                                if not item_rows.empty:
                                    st.markdown(f"### Data for: {selected_item}")
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.markdown(f"""
                                        <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                            <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{len(item_rows):,}</div>
                                            <div style="font-size: 14px; color: #666;">Records</div>
                                        </div>
                                        """, unsafe_allow_html=True)
                                    with c2:
                                        if "circulation_size" in item_rows:
                                            st.markdown(f"""
                                            <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                                <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{item_rows['circulation_size'].mean():,.0f}</div>
                                                <div style="font-size: 14px; color: #666;">Avg Circulation</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    with c3:
                                        if "body_token_count" in item_rows:
                                            st.markdown(f"""
                                            <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                                <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{item_rows['body_token_count'].mean():,.0f}</div>
                                                <div style="font-size: 14px; color: #666;">Avg Tokens</div>
                                            </div>
                                            """, unsafe_allow_html=True)
                                    st.dataframe(item_rows, use_container_width=True, height=400)
                                    export_data_button(item_rows, f"{sel_col}_{str(selected_item)[:40]}", "csv")
                                else:
                                    st.warning("No rows for the selected item.")
                        else:
                            st.warning(f"No matches for '{search_term}' in {sel_col}.")
                    except Exception as e:
                        st.error(f"Error searching: {e}")
    else:
        # Term Attribution - Use a simple text input instead of heavy selectbox
        st.markdown("**Search for terms in article content:**")
        term = st.text_input(
            "Enter a term to search for",
            placeholder="e.g., healthcare, policy, reform...",
            key="term_search",
            help="Search for specific terms in headlines and article content"
        )

        if "selected_term" in st.session_state:
            term = st.session_state["selected_term"]
            st.success(f"Selected term: {term}")
            if st.button("Clear Term Selection", key="clear_term"):
                del st.session_state["selected_term"]
                # Don't use st.rerun() here - let the clearing happen naturally

        if term and not df_main.empty:
            add_to_recent_searches(f"Term: {term}")
            try:
                # Limit search to prevent memory issues
                search_df = df_main.head(5000)  # Limit to first 5k rows for performance
                
                text_cols = [c for c in available_cols if any(k in c.lower() for k in ["headline", "body", "content", "text"])]
                if not text_cols:
                    st.warning("No text columns found for term search.")
                else:
                    # Create search mask more efficiently
                    mask = pd.Series(False, index=search_df.index)
                    for c in text_cols:
                        try:
                            s = search_df[c].astype(str, errors="ignore")
                            mask |= s.fillna("").str.contains(term, case=False, na=False, regex=False)
                        except Exception as col_error:
                            st.warning(f"Could not search column '{c}': {col_error}")
                            continue
                    
                    hits = search_df[mask].head(100)
                    
                    if not hits.empty:
                        st.success(f"Found {len(hits)} articles containing '{term}' (searched {len(search_df):,} articles)")
                        
                        # Display metrics
                        c1, c2, c3 = st.columns(3)
                        with c1:
                            st.markdown(f"""
                            <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{len(hits):,}</div>
                                <div style="font-size: 14px; color: #666;">Total Matches</div>
                            </div>
                            """, unsafe_allow_html=True)
                        with c2:
                            if "circulation_size" in hits.columns:
                                try:
                                    total_reach = hits['circulation_size'].sum()
                                    st.markdown(f"""
                                    <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                        <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{total_reach:,.0f}</div>
                                        <div style="font-size: 14px; color: #666;">Total Reach</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                except Exception:
                                    st.markdown(f"""
                                    <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                        <div style="font-size: 24px; font-weight: bold; color: #0A473B;">N/A</div>
                                        <div style="font-size: 14px; color: #666;">Total Reach</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        with c3:
                            date_cols = [c for c in hits.columns if ("date" in c.lower() or "time" in c.lower())]
                            if date_cols:
                                try:
                                    dc = date_cols[0]
                                    dates = pd.to_datetime(hits[dc], errors="coerce").dropna()
                                    unique_dates = dates.dt.date.nunique()
                                    st.markdown(f"""
                                    <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                        <div style="font-size: 24px; font-weight: bold; color: #0A473B;">{unique_dates}</div>
                                        <div style="font-size: 14px; color: #666;">Date Span (days)</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                except Exception:
                                    st.markdown(f"""
                                    <div style="text-align: center; border: 2px solid #0A473B; padding: 15px; border-radius: 8px; margin: 5px;">
                                        <div style="font-size: 24px; font-weight: bold; color: #0A473B;">N/A</div>
                                        <div style="font-size: 14px; color: #666;">Date Span</div>
                                    </div>
                                    """, unsafe_allow_html=True)
                        
                        st.markdown("### Sample Results")
                        st.dataframe(hits, use_container_width=True, height=400)
                        export_data_button(hits, f"term_search_{term[:40]}", "csv")
                    else:
                        st.warning(f"No articles found containing '{term}' in the searched {len(search_df):,} articles.")
                    
            except Exception as e:
                st.error(f"Error searching for term: {e}")
                st.info("Try using a shorter or simpler search term.")

# -------------------- Network Data Loading --------------------
@st.cache_data(ttl=1800)  # Cache for 30 minutes
def load_network_data_file(filename: str):
    """Load a single network data file on demand"""
    try:
        # Try multiple possible paths for the data files
        possible_paths = [
            # Streamlit Cloud paths
            Path("../data/processed"),  # data/processed
            Path("data/processed"),    # data/processed
            # Local development paths
            APP_DIR.parent / "data" / "processed",
            ROOT / "data" / "processed",
            Path.cwd() / "data" / "processed",
        ]
        
        data_dir = next((p for p in possible_paths if p.exists()), None)
        if data_dir is None:
            print(f"Warning: No network data directory found for {filename}")
            return pd.DataFrame()
        
        file_path = data_dir / filename
        if file_path.exists():
            try:
                # Limit file size for memory efficiency
                df = pd.read_csv(file_path)
                # Limit to reasonable size for Streamlit Cloud
                if len(df) > 10000:
                    df = df.head(10000)
                print(f"Loaded {filename}: {len(df)} rows")
                return df
            except Exception as file_error:
                print(f"Warning: Could not load {filename}: {file_error}")
                return pd.DataFrame()
        else:
            print(f"Warning: {filename} not found at {file_path}")
            return pd.DataFrame()
    except Exception as e:
        print(f"Error loading {filename}: {e}")
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

with tab4:
    st.subheader("People - Network Analysis")
    st.markdown("Explore influence networks and publisher-term relationships using pre-computed data.")
    
    # Load network data with lazy loading
    network_data = get_network_data()
    
    if network_data is None:
        st.error("Failed to load network data. Please ensure the data files exist.")
    else:
        # Convert keys to match expected format
        network_data = {
            'influence_nodes': network_data.get('influence_nodes', pd.DataFrame()),
            'influence_edges': network_data.get('influence_edges', pd.DataFrame()),
            'publisher_term_edges': network_data.get('publisher_term_edges', pd.DataFrame()),
            'community_summary': network_data.get('community_summary', pd.DataFrame()),
            'term_comparison': network_data.get('term_comparison', pd.DataFrame()),
            'top_terms_chunk': network_data.get('top_terms_chunk', pd.DataFrame()),
            'top_terms_global': network_data.get('top_terms_global', pd.DataFrame())
        }
        # Network type selection (similar to attribution tab)
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            network_type = st.selectbox(
                "Select Network Type:",
                ["Influence Network", "Publisher-Term Network", "Data Tables"]
            )
        
        with col2:
            if network_type in ["Influence Network", "Publisher-Term Network"]:
                top_n = st.slider("Top Items", 10, 100, 50, 10)
        
        with col3:
            if network_type == "Publisher-Term Network":
                top_pubs = st.slider("Top Publishers", 5, 50, 20, 5)
        
        # Display selected network type
        if network_type == "Influence Network":
            st.markdown("### Influence Network Analysis")
            
            # Analysis controls
            col1, col2, col3 = st.columns(3)
            with col1:
                top_n = st.slider("Top Nodes to Display", 10, 100, 30, 10)
            with col2:
                min_credit = st.slider("Min Credit Share", 0.0, 0.1, 0.001, 0.001)
            with col3:
                show_edges = st.checkbox("Show Connections", value=True)
            
            # Filter and analyze data
            filtered_nodes = network_data['influence_nodes'][
                network_data['influence_nodes']['credit_share'] >= min_credit
            ].nlargest(top_n, 'credit_share')
            
            # Create summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Nodes", len(filtered_nodes))
            with col2:
                st.metric("Avg Credit Share", f"{filtered_nodes['credit_share'].mean():.3f}")
            with col3:
                st.metric("Top Dimension", filtered_nodes['dimension'].mode().iloc[0] if not filtered_nodes.empty else "N/A")
            with col4:
                st.metric("Terms vs Items", f"{len(filtered_nodes[filtered_nodes['kind']=='term'])}/{len(filtered_nodes[filtered_nodes['kind']=='item'])}")
            
            # Visual at the top - Top Influence Chart
            st.markdown("#### Top Influence Performers")
            
            # Create a horizontal bar chart of top performers
            top_20 = filtered_nodes.head(20)
            fig = go.Figure()
            
            # Add bars for terms and items with different colors
            terms = top_20[top_20['kind'] == 'term']
            items = top_20[top_20['kind'] == 'item']
            
            if not terms.empty:
                fig.add_trace(go.Bar(
                    y=terms['value'],
                    x=terms['credit_share'],
                    orientation='h',
                    name='Terms',
                    marker_color=PENTA_COLORS[1],  # Penta green
                    text=terms['credit_share'].round(3),
                    textposition='auto',
                ))
            
            if not items.empty:
                fig.add_trace(go.Bar(
                    y=items['value'],
                    x=items['credit_share'],
                    orientation='h',
                    name='Items',
                    marker_color=PENTA_COLORS[0],  # Penta primary
                    text=items['credit_share'].round(3),
                    textposition='auto',
                ))
            
            # Apply Penta branding
            fig = create_penta_chart(
                fig, 
                title="Top Influence Performers by Credit Share",
                height=600
            )
            
            fig.update_layout(
                xaxis_title="Credit Share",
                yaxis_title="Terms/Items",
                barmode='group',
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Note: Network visualization disabled for memory optimization
            st.info("💡 Network visualization is disabled to optimize memory usage on Streamlit Cloud. Use the data tables below for detailed analysis.")
            
            # Top performers table
            st.markdown("#### Top Influence Performers")
            display_cols = ['value', 'kind', 'dimension', 'credit_share', 'credit', 'rating']
            st.dataframe(
                filtered_nodes[display_cols].head(20),
                use_container_width=True,
                hide_index=True
            )
            
            # Dimension breakdown
            st.markdown("#### Influence by Dimension")
            dim_analysis = filtered_nodes.groupby('dimension').agg({
                'credit_share': ['sum', 'mean', 'count'],
                'credit': 'sum'
            }).round(3)
            dim_analysis.columns = ['Total Credit Share', 'Avg Credit Share', 'Node Count', 'Total Credit']
            st.dataframe(dim_analysis, use_container_width=True)
            

        elif network_type == "Publisher-Term Network":
            st.markdown("### Publisher-Term Association Analysis")
            
            # Analysis controls
            col1, col2, col3 = st.columns(3)
            with col1:
                top_pubs = st.slider("Top Publishers", 5, 50, 15, 5)
            with col2:
                top_terms = st.slider("Top Terms", 5, 50, 15, 5)
            with col3:
                min_weight = st.slider("Min Association Weight", 0.0, 1.0, 0.1, 0.01)
            
            # Filter data
            top_pubs_list = network_data['publisher_term_edges'].groupby('publisher')['weight'].sum().nlargest(top_pubs).index
            top_terms_list = network_data['publisher_term_edges'].groupby('term')['weight'].sum().nlargest(top_terms).index
            
            filtered_edges = network_data['publisher_term_edges'][
                (network_data['publisher_term_edges']['publisher'].isin(top_pubs_list)) & 
                (network_data['publisher_term_edges']['term'].isin(top_terms_list)) &
                (network_data['publisher_term_edges']['weight'] >= min_weight)
            ]
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Associations", len(filtered_edges))
            with col2:
                st.metric("Unique Publishers", filtered_edges['publisher'].nunique())
            with col3:
                st.metric("Unique Terms", filtered_edges['term'].nunique())
            with col4:
                st.metric("Avg Weight", f"{filtered_edges['weight'].mean():.3f}")
            
            # Visual at the top - Top Associations Chart
            st.markdown("#### Top Publisher-Term Associations")
            
            # Create a horizontal bar chart of top associations
            top_associations = filtered_edges.nlargest(15, 'weight')
            fig = go.Figure()
            
            fig.add_trace(go.Bar(
                y=[f"{row['publisher'][:20]}... - {row['term'][:20]}..." for _, row in top_associations.iterrows()],
                x=top_associations['weight'],
                orientation='h',
                name='Associations',
                marker_color=PENTA_COLORS[2],  # Penta dark
                text=top_associations['weight'].round(3),
                textposition='auto',
            ))
            
            # Apply Penta branding
            fig = create_penta_chart(
                fig, 
                title="Strongest Publisher-Term Associations",
                height=600
            )
            
            fig.update_layout(
                xaxis_title="Association Weight",
                yaxis_title="Publisher - Term",
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Note: Network visualization disabled for memory optimization
            st.info("💡 Network visualization is disabled to optimize memory usage on Streamlit Cloud. Use the data tables below for detailed analysis.")
            
            # Top associations table
            st.markdown("#### Strongest Publisher-Term Associations")
            st.dataframe(
                filtered_edges.nlargest(20, 'weight')[['publisher', 'term', 'weight']],
                use_container_width=True,
                hide_index=True
            )
            
            # Publisher analysis
            st.markdown("#### Publisher Performance")
            pub_analysis = filtered_edges.groupby('publisher').agg({
                'weight': ['sum', 'mean', 'count'],
                'term': 'nunique'
            }).round(3)
            pub_analysis.columns = ['Total Weight', 'Avg Weight', 'Association Count', 'Unique Terms']
            st.dataframe(pub_analysis.sort_values('Total Weight', ascending=False), use_container_width=True)
            
            # Term analysis
            st.markdown("#### Term Performance")
            term_analysis = filtered_edges.groupby('term').agg({
                'weight': ['sum', 'mean', 'count'],
                'publisher': 'nunique'
            }).round(3)
            term_analysis.columns = ['Total Weight', 'Avg Weight', 'Association Count', 'Unique Publishers']
            st.dataframe(term_analysis.sort_values('Total Weight', ascending=False), use_container_width=True)
            
            # Community summary
            if not network_data['community_summary'].empty:
                st.markdown("#### Network Communities")
                st.dataframe(network_data['community_summary'], use_container_width=True)
            
        
        elif network_type == "Data Tables":
            st.markdown("### Network Data Tables")
            
            # Network metrics dashboard
            st.markdown("#### Network Metrics Dashboard")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Total Influence Nodes",
                    value=f"{len(network_data['influence_nodes']):,}"
                )
            
            with col2:
                st.metric(
                    label="Publisher-Term Associations",
                    value=f"{len(network_data['publisher_term_edges']):,}"
                )
            
            with col3:
                st.metric(
                    label="Network Communities",
                    value=f"{len(network_data['community_summary'])}"
                )
            
            with col4:
                st.metric(
                    label="Unique Publishers",
                    value=f"{network_data['publisher_term_edges']['publisher'].nunique():,}"
                )
            
            # Data tables
            st.markdown("#### Raw Data Tables")
            
            table_option = st.selectbox(
                "Select table to view:",
                ["Influence Nodes", "Influence Edges", "Publisher-Term Edges", 
                 "Community Summary", "Term Comparison", "Top Terms (Chunk)", "Top Terms (Global)"]
            )
            
            table_mapping = {
                "Influence Nodes": network_data['influence_nodes'],
                "Influence Edges": network_data['influence_edges'],
                "Publisher-Term Edges": network_data['publisher_term_edges'],
                "Community Summary": network_data['community_summary'],
                "Term Comparison": network_data['term_comparison'],
                "Top Terms (Chunk)": network_data['top_terms_chunk'],
                "Top Terms (Global)": network_data['top_terms_global']
            }
            
            selected_table = table_mapping[table_option]
            st.dataframe(selected_table, use_container_width=True)
            
            # Download button
            csv = selected_table.to_csv(index=False)
            st.download_button(
                label=f"Download {table_option} as CSV",
                data=csv,
                file_name=f"{table_option.lower().replace(' ', '_')}.csv",
                mime="text/csv"
            )
 
# -------------------- Dataset Footnote --------------------
if 'dataset_info' in st.session_state:
    st.markdown("---")
    st.markdown(
        f"""
        <div style="font-size: 0.8em; text-align: center; margin-top: 2rem;">
            Dataset: {st.session_state.dataset_info['rows']:,} observations from {st.session_state.dataset_info['files']} files.
            Filtered to show top 90% by circulation size for high-impact analysis.
        </div>
        """,
        unsafe_allow_html=True
    )