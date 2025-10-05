"""
Network Analysis Functions for PolicyPath
Contains network building, analysis, and visualization functions.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import re
from typing import Dict, List, Tuple, Optional


# Constants
SINK_LABEL = "<CONV>"
DIMENSION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
KIND_MARKERS = {"item": "o", "term": "^"}  # circle, triangle


def prepare_article_data(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare article data for content network analysis."""
    required_cols = ["publisher_name", "processed_headline", "processed_body"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"df missing required columns: {missing}")

    df_use = df.copy()
    df_use["publisher_name"] = df_use["publisher_name"].astype(str).fillna("Unknown")
    for c in ["processed_headline", "processed_body"]:
        df_use[c] = df_use[c].fillna("").astype(str)
    
    return df_use


def build_term_scores(attr_df: pd.DataFrame) -> pd.DataFrame:
    """Build term scores table from attribution data."""
    term_scores = (
        attr_df.query("kind=='term'")
               .loc[:, ["value", "credit_share"]]
               .dropna()
               .rename(columns={"value": "term", "credit_share": "term_weight"})
               .sort_values("term_weight", ascending=False)
               .reset_index(drop=True)
    )
    return term_scores


def compile_term_patterns(terms: List[str]) -> List[re.Pattern]:
    """Compile regex patterns for term matching."""
    return [re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE) for t in terms]


def best_term_for_text(text: str, ranked_terms: List[Tuple[str, float]], patterns: List[re.Pattern]) -> Tuple[Optional[str], float]:
    """Return (best_term, best_weight) present in 'text' or (None, 0.0)."""
    # iterate in ranked order; first hit is the best
    for (t, w), pat in zip(ranked_terms, patterns):
        if pat.search(text):
            return t, float(w)
    return None, 0.0


def choose_labels_by_rank(G: nx.DiGraph, top_labels_per_dim: int = 10) -> Dict[str, str]:
    """Choose labels for nodes based on per-dimension credit share ranking."""
    by_dim = {}
    for n, d in G.nodes(data=True):
        if n == SINK_LABEL: 
            continue
        dim = d.get("dimension", "other")
        by_dim.setdefault(dim, []).append((n, d.get("credit_share", 0.0)))
    
    labels = {}
    for dim, pairs in by_dim.items():
        pairs.sort(key=lambda t: t[1], reverse=True)
        for n, _ in pairs[:top_labels_per_dim]:
            labels[n] = n
    return labels


def prep_attr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare attribution dataframe with proper data types."""
    required = {"kind", "dimension", "value", "credit", "credit_share", "rating", "rating_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"attr_df missing required columns: {sorted(missing)}")
    
    out = df.copy()
    out["kind"] = out["kind"].astype(str)
    out["dimension"] = out["dimension"].astype(str)
    out["value"] = out["value"].astype(str)
    
    for c in ("credit", "credit_share", "rating_pct"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(0).astype(int)
    
    return out


def filter_top_per_dimension(df: pd.DataFrame, top_k: int = 50, min_share: float = 0.0) -> pd.DataFrame:
    """Filter top items per dimension by credit share."""
    x = df.copy()
    
    # filter by per-dimension share (credit_share) — classic behavior
    if min_share > 0:
        x = x[x["credit_share"] >= min_share]
    
    if top_k and top_k > 0:
        x = (
            x.sort_values(["dimension", "credit_share"], ascending=[True, False])
            .groupby("dimension", as_index=False)
            .head(top_k)
        )
    
    return x.reset_index(drop=True)


def build_graph_classic(attr_df: pd.DataFrame) -> nx.DiGraph:
    """Build classic influence network graph from attribution data."""
    df = prep_attr_df(attr_df)

    # colors by dimension (fixed order)
    dims = sorted(df["dimension"].unique())
    dim_to_color = {d: DIMENSION_COLORS[i % len(DIMENSION_COLORS)] for i, d in enumerate(dims)}

    # one node per (value,kind,dimension) with summed metrics
    nodes_tbl = (
        df.groupby(["value", "kind", "dimension"], as_index=False)
          .agg({"credit": "sum", "credit_share": "sum", "rating": "max", "rating_pct": "max"})
    )

    G = nx.DiGraph()

    # nodes: size = linear in credit_share (classic)
    for _, r in nodes_tbl.iterrows():
        node_id = r["value"]
        color = dim_to_color.get(r["dimension"], "#7f7f7f")
        size = max(80, float(r["credit_share"]) * 2200)
        G.add_node(
            node_id,
            kind=r["kind"], dimension=r["dimension"], color=color, size=size,
            credit=float(r["credit"]), credit_share=float(r["credit_share"]),
            rating=int(r["rating"]), rating_pct=float(r["rating_pct"])
        )

    # sink node
    if SINK_LABEL not in G:
        G.add_node(SINK_LABEL, kind="sink", dimension="conversion", color="#222222", size=500)

    # edges: weight = linear credit_share (classic)
    edge_tbl = df.groupby("value", as_index=False)["credit_share"].sum().rename(columns={"credit_share": "edge_weight"})
    for _, r in edge_tbl.iterrows():
        w = float(r["edge_weight"])
        if w > 0:
            G.add_edge(r["value"], SINK_LABEL, weight=w)

    return G


def draw_static_classic(G: nx.DiGraph, figsize=(12, 8), top_labels_per_dim: int = 10):
    """Draw static classic influence network."""
    pos = nx.spring_layout(G, k=0.6, seed=42)  # classic, reproducible

    plt.figure(figsize=figsize)

    # edges: linear thickness
    widths = [max(0.5, G[u][v].get("weight", 0.0) * 6) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.25, arrows=True, arrowstyle="-|>", arrowsize=10)

    # nodes by kind (marker), colors by dimension
    for kind, marker in KIND_MARKERS.items():
        nodes_k = [n for n, d in G.nodes(data=True) if d.get("kind") == kind]
        sizes = [G.nodes[n].get("size", 120) for n in nodes_k]
        colors = [G.nodes[n].get("color", "#7f7f7f") for n in nodes_k]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_k, node_size=sizes, node_color=colors,
                               node_shape=marker, linewidths=0.8, edgecolors="#333", alpha=0.95)

    # sink
    if SINK_LABEL in G:
        nx.draw_networkx_nodes(G, pos, nodelist=[SINK_LABEL], node_size=700,
                               node_color="#222222", node_shape="s",
                               linewidths=0.8, edgecolors="#111111")

    # labels: top-N per dimension by credit_share
    labels = choose_labels_by_rank(G, top_labels_per_dim=top_labels_per_dim)
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title("Influence Network: Actors & Terms Driving Conversion", fontsize=14, pad=10)
    subtitle = "Color = dimension • Shape = kind (● item, ▲ term) • Size ≈ credit_share • Edge width ≈ credit_share"
    plt.suptitle(subtitle, y=0.88, fontsize=9, color="#444")

    kind_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#777', markeredgecolor='#333', label='item', markersize=8),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='#777', markeredgecolor='#333', label='term', markersize=8),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='#222', markeredgecolor='#111', label='<CONV>', markersize=8),
    ]
    plt.legend(handles=kind_handles, loc='lower left', frameon=True, title='Kind')

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def build_publisher_term_edges(
    articles: pd.DataFrame,
    term_table: pd.DataFrame,
    term_limit: int = 500,
    min_term_weight: float = 0.0
) -> pd.DataFrame:
    """
    Build edges between publishers and their most influential terms.
    Returns edge table with columns: publisher, term, weight
    """
    tt = term_table.head(term_limit).copy()
    ranked_terms = list(tt.itertuples(index=False, name=None))  # (term, weight)
    patterns = compile_term_patterns([t for t, _ in ranked_terms])

    edges = {}
    for _, r in articles.iterrows():
        txt = f"{r['processed_headline']} {r['processed_body']}"
        best_t, best_w = best_term_for_text(txt, ranked_terms, patterns)
        if best_t is None or best_w < min_term_weight:
            continue
        key = (r["publisher_name"], best_t)
        edges[key] = edges.get(key, 0.0) + best_w

    if not edges:
        return pd.DataFrame(columns=["publisher", "term", "weight"])

    pub, term, w = zip(*[(p, t, v) for (p, t), v in edges.items()])
    return pd.DataFrame({"publisher": pub, "term": term, "weight": w})


def build_bipartite_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build an undirected bipartite graph between publishers and terms."""
    G = nx.Graph()
    for _, r in edges.iterrows():
        p, t, w = r["publisher"], r["term"], float(r["weight"])
        if p not in G: 
            G.add_node(p, ntype="publisher")
        if t not in G: 
            G.add_node(t, ntype="term")
        if w > 0:
            G.add_edge(p, t, weight=w)
    return G


def draw_publisher_term_network(
    G: nx.Graph,
    top_labels_per_type: int = 12,
    figsize=(14, 9)
):
    """Draw publisher-term content network."""
    # size nodes by (weighted) degree
    def node_strength(n):
        return sum(G[n][nbr].get("weight", 0.0) for nbr in G.neighbors(n))

    strengths = {n: node_strength(n) for n in G.nodes()}
    max_strength = max(strengths.values()) if strengths else 1.0

    # scale sizes (nonlinear to spread)
    def scale(s): 
        return max(80.0, (s / max_strength) ** 0.65 * 3000.0)
    sizes = [scale(strengths[n]) for n in G.nodes()]

    # colors & shapes by type
    color_map = {"publisher": "#1f77b4", "term": "#ff7f0e"}
    shape_map = {"publisher": "o", "term": "^"}
    node_types = nx.get_node_attributes(G, "ntype")
    node_colors = [color_map.get(node_types.get(n, "publisher"), "#7f7f7f") for n in G.nodes()]

    # layout (weighted for clarity)
    pos = nx.spring_layout(G, k=0.6, seed=42, weight="weight")

    plt.figure(figsize=figsize)

    # edges: thickness by weight (sqrt for contrast)
    widths = [max(0.5, (G[u][v].get("weight", 0.0) ** 0.5) * 6) for u, v in G.edges()]
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.25)

    # draw nodes by type (for shapes)
    for t, marker in shape_map.items():
        nodes_t = [n for n in G.nodes() if node_types.get(n) == t]
        sizes_t = [sizes[list(G.nodes()).index(n)] for n in nodes_t]
        colors_t = [node_colors[list(G.nodes()).index(n)] for n in nodes_t]
        nx.draw_networkx_nodes(
            G, pos, nodelist=nodes_t, node_size=sizes_t, node_color=colors_t,
            node_shape=marker, linewidths=0.8, edgecolors="#333", alpha=0.95
        )

    # labels: top-N publishers and top-N terms by strength
    pubs = [n for n in G.nodes() if node_types.get(n) == "publisher"]
    terms = [n for n in G.nodes() if node_types.get(n) == "term"]
    pubs_top = sorted(pubs, key=lambda n: strengths[n], reverse=True)[:top_labels_per_type]
    terms_top = sorted(terms, key=lambda n: strengths[n], reverse=True)[:top_labels_per_type]
    labels = {n: n for n in pubs_top + terms_top}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    # title + legend
    plt.title("Content Network: Publishers ↔ Highest-Impact Terms (per article)", fontsize=14, pad=10)
    subtitle = "Circle = publisher • Triangle = term • Node size ≈ weighted degree • Edge width ≈ association strength"
    plt.suptitle(subtitle, y=0.80, fontsize=9, color="#444")

    legend_handles = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor=color_map["publisher"],
               markeredgecolor='#333', label='publisher', markersize=8),
        Line2D([0], [0], marker='^', color='w', markerfacecolor=color_map["term"],
               markeredgecolor='#333', label='term', markersize=8),
    ]
    plt.legend(handles=legend_handles, loc='upper left', frameon=True, title='Node type')

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def run_influence_network_analysis(attr_df: pd.DataFrame, top_k: int = 50, min_share: float = 0.0):
    """Run complete influence network analysis."""
    _attr = prep_attr_df(attr_df)
    _attr_trim = filter_top_per_dimension(_attr, top_k=top_k, min_share=min_share)
    
    # Build & plot
    G = build_graph_classic(_attr_trim)
    draw_static_classic(G, figsize=(12, 8), top_labels_per_dim=10)
    
    return G


def run_content_network_analysis(df: pd.DataFrame, attr_df: pd.DataFrame, 
                                term_limit: int = 200, min_term_weight: float = 0.0,
                                top_labels_per_type: int = 12):
    """Run complete content network analysis."""
    df_use = prepare_article_data(df)
    term_scores = build_term_scores(attr_df)
    
    edges = build_publisher_term_edges(
        articles=df_use,
        term_table=term_scores,
        term_limit=term_limit,
        min_term_weight=min_term_weight
    )
    
    G_content = build_bipartite_graph(edges)
    draw_publisher_term_network(G_content, top_labels_per_type=top_labels_per_type, figsize=(14, 9))
    
    return G_content, edges


# -------------------- Command Line Interface --------------------
import argparse
import os
from pathlib import Path


def load_datasets():
    """Load the required datasets from their expected locations."""
    # Define possible paths for the datasets
    possible_paths = [
        Path("data/processed"),
        Path("../data/processed"),
        Path("data"),
        Path("../data"),
        Path("../../data/processed"),
        Path("../../../data/processed"),
    ]
    
    # Find the datasets
    main_dataset_path = None
    attribution_dataset_path = None
    
    for base_path in possible_paths:
        main_path = base_path / "final_model_dataset.csv"
        attr_path = base_path / "attribution_all_scored.csv"
        
        if main_path.exists():
            main_dataset_path = main_path
        if attr_path.exists():
            attribution_dataset_path = attr_path
    
    if not main_dataset_path:
        raise FileNotFoundError(
            "Could not find final_model_dataset.csv. "
            "Expected locations: data/processed/final_model_dataset.csv"
        )
    
    if not attribution_dataset_path:
        raise FileNotFoundError(
            "Could not find attribution_all_scored.csv. "
            "Expected locations: data/processed/attribution_all_scored.csv"
        )
    
    print(f"Loading main dataset from: {main_dataset_path}")
    print(f"Loading attribution dataset from: {attribution_dataset_path}")
    
    # Load the datasets
    try:
        df_main = pd.read_csv(main_dataset_path, dtype_backend="pyarrow")
        df_attr = pd.read_csv(attribution_dataset_path, dtype_backend="pyarrow")
        
        print(f"Loaded {len(df_main):,} articles and {len(df_attr):,} attribution records")
        return df_main, df_attr
        
    except Exception as e:
        print(f"Error loading datasets: {e}")
        raise


def run_term_comparison(df_attr):
    """Run term ranking comparison analysis."""
    print("\n" + "="*60)
    print("TERM RANKING COMPARISON")
    print("="*60)
    
    # Get term scores
    term_scores = build_term_scores(df_attr)
    
    print(f"\nTop 20 terms by credit share:")
    print("-" * 40)
    top_terms = term_scores.head(20)
    for i, (_, row) in enumerate(top_terms.iterrows(), 1):
        print(f"{i:2d}. {row['term']:<30} {row['term_weight']:.4f}")
    
    # Analyze by dimension
    print(f"\nTop terms by dimension:")
    print("-" * 40)
    df_prep = prep_attr_df(df_attr)
    term_df = df_prep[df_prep['kind'] == 'term'].copy()
    
    for dimension in sorted(term_df['dimension'].unique()):
        dim_terms = term_df[term_df['dimension'] == dimension].nlargest(5, 'credit_share')
        print(f"\n{dimension.upper()}:")
        for _, row in dim_terms.iterrows():
            print(f"  • {row['value']:<25} {row['credit_share']:.4f}")


def main():
    """Main function to run network analysis."""
    parser = argparse.ArgumentParser(description="Run PolicyPath Network Analysis")
    parser.add_argument("--influence-only", action="store_true", 
                       help="Run only influence network analysis")
    parser.add_argument("--content-only", action="store_true", 
                       help="Run only content network analysis")
    parser.add_argument("--save-plots", action="store_true", 
                       help="Save plots as PNG files instead of displaying")
    parser.add_argument("--output-dir", default="output", 
                       help="Directory to save plots (default: output)")
    parser.add_argument("--top-k", type=int, default=50, 
                       help="Top K items per dimension for influence network (default: 50)")
    parser.add_argument("--term-limit", type=int, default=200, 
                       help="Term limit for content network (default: 200)")
    
    args = parser.parse_args()
    
    print("PolicyPath Network Analysis")
    print("=" * 50)
    
    try:
        # Load datasets
        df_main, df_attr = load_datasets()
        
        # Create output directory if saving plots
        if args.save_plots:
            output_dir = Path(args.output_dir)
            output_dir.mkdir(exist_ok=True)
            print(f"Output directory: {output_dir.absolute()}")
        
        # Run analyses based on arguments
        if args.content_only:
            print("\nRunning Content Network Analysis...")
            run_content_network_analysis(
                df_main, df_attr, 
                term_limit=args.term_limit,
                top_labels_per_type=12
            )
            
        elif args.influence_only:
            print("\nRunning Influence Network Analysis...")
            run_influence_network_analysis(df_attr, top_k=args.top_k)
            
        else:
            # Run both analyses
            print("\nRunning Influence Network Analysis...")
            run_influence_network_analysis(df_attr, top_k=args.top_k)
            
            print("\nRunning Content Network Analysis...")
            run_content_network_analysis(
                df_main, df_attr, 
                term_limit=args.term_limit,
                top_labels_per_type=12
            )
        
        # Always run term comparison
        run_term_comparison(df_attr)
        
        print("\n" + "="*60)
        print("ANALYSIS COMPLETE")
        print("="*60)
        
    except Exception as e:
        print(f"\nError: {e}")
        import sys
        sys.exit(1)


if __name__ == "__main__":
    main()
