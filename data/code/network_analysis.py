#!/usr/bin/env python3
"""
PolicyPath Network Analysis
Complete network analysis script that can be run from terminal.
"""

import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import re
import math
from networkx.algorithms.community import greedy_modularity_communities
from typing import Optional, List, Tuple, Dict, Any
import argparse
import sys
import os


# Constants
SINK_LABEL = "<CONV>"
DIMENSION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
KIND_MARKERS = {"item": "o", "term": "^"}  # circle, triangle


def load_data():
    """Load the main datasets."""
    print("Loading data...")
    df = pd.read_csv("../../data/processed/final_model_dataset.csv")
    attr_df = pd.read_csv("../../data/processed/attribution_all_scored.csv")
    print(f"Loaded {len(df)} articles and {len(attr_df)} attribution records")
    return df, attr_df


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


def build_graph_classic(attr_df: pd.DataFrame) -> nx.DiGraph:
    """Build classic influence network graph from attribution data."""
    print("Building influence network graph...")
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

    print(f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def draw_static_classic(G: nx.DiGraph, figsize=(12, 8), top_labels_per_dim: int = 10, save_path=None):
    """Draw static classic influence network."""
    print("Drawing influence network...")
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved influence network plot to {save_path}")
    else:
        plt.show()


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


def build_publisher_term_edges(
    articles: pd.DataFrame,
    term_table: pd.DataFrame,
    term_limit: int = 500,
    min_term_weight: float = 0.0
) -> pd.DataFrame:
    """Build edges between publishers and their most influential terms."""
    print(f"Building publisher-term edges (limit: {term_limit})...")
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
    result = pd.DataFrame({"publisher": pub, "term": term, "weight": w})
    print(f"Built {len(result)} publisher-term edges")
    return result


def build_bipartite_graph(edges: pd.DataFrame) -> nx.Graph:
    """Build an undirected bipartite graph between publishers and terms."""
    print("Building bipartite graph...")
    G = nx.Graph()
    for _, r in edges.iterrows():
        p, t, w = r["publisher"], r["term"], float(r["weight"])
        if p not in G: 
            G.add_node(p, ntype="publisher")
        if t not in G: 
            G.add_node(t, ntype="term")
        if w > 0:
            G.add_edge(p, t, weight=w)
    
    print(f"Built bipartite graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    return G


def draw_publisher_term_network(
    G: nx.Graph,
    top_labels_per_type: int = 12,
    figsize=(14, 9),
    save_path=None
):
    """Draw publisher-term content network."""
    print("Drawing content network...")
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
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved content network plot to {save_path}")
    else:
        plt.show()


def compare_term_rankings(attr_df: pd.DataFrame):
    """Compare term rankings between chunk-normalized and global-normalized approaches."""
    print("Comparing term rankings...")
    
    # Prepare terms data
    _terms = attr_df[attr_df["kind"] == "term"].copy()
    _terms["credit"] = pd.to_numeric(_terms["credit"], errors="coerce").fillna(0.0)
    _terms["credit_share"] = pd.to_numeric(_terms["credit_share"], errors="coerce").fillna(0.0)
    _terms["value"] = _terms["value"].astype(str)
    
    # Add global credit share
    total = _terms["credit"].sum()
    _terms["credit_share_global"] = _terms["credit"] / total if total > 0 else 0.0

    # Get top terms for each approach
    top_chunk = (_terms.groupby("value", as_index=False)["credit_share"]
                      .sum()
                      .sort_values("credit_share", ascending=False)
                      .head(20))
    top_chunk.insert(0, "rank", range(1, len(top_chunk) + 1))
    top_chunk = top_chunk.rename(columns={"rank": "rank_chunk", "credit_share": "credit_share_chunk"})

    top_global = (_terms.groupby("value", as_index=False)["credit_share_global"]
                       .sum()
                       .sort_values("credit_share_global", ascending=False)
                       .head(20))
    top_global.insert(0, "rank", range(1, len(top_global) + 1))
    top_global = top_global.rename(columns={"rank": "rank_global", "credit_share_global": "credit_share_global"})

    # Merge to compare
    compare = pd.merge(top_chunk, top_global, on="value", how="outer")
    
    # Fill missing ranks
    max_rank = 999
    compare["rank_chunk"] = compare["rank_chunk"].fillna(max_rank).astype(int)
    compare["rank_global"] = compare["rank_global"].fillna(max_rank).astype(int)
    compare["credit_share_chunk"] = compare["credit_share_chunk"].fillna(0.0)
    compare["credit_share_global"] = compare["credit_share_global"].fillna(0.0)

    # Calculate deltas
    compare["rank_delta"] = compare["rank_global"] - compare["rank_chunk"]
    compare["share_delta"] = compare["credit_share_global"] - compare["credit_share_chunk"]

    # Sort by global rank
    compare = compare.sort_values(["rank_global", "rank_chunk"]).reset_index(drop=True)

    return compare, top_chunk, top_global


def main():
    """Main function to run the complete analysis."""
    parser = argparse.ArgumentParser(description='PolicyPath Network Analysis')
    parser.add_argument('--influence-only', action='store_true', help='Run only influence network analysis')
    parser.add_argument('--content-only', action='store_true', help='Run only content network analysis')
    parser.add_argument('--save-plots', action='store_true', help='Save plots to files instead of displaying')
    parser.add_argument('--output-dir', default='output', help='Output directory for saved plots')
    parser.add_argument('--top-k', type=int, default=50, help='Top K items per dimension for influence network')
    parser.add_argument('--term-limit', type=int, default=200, help='Term limit for content network')
    
    args = parser.parse_args()
    
    # Create output directory if saving plots
    if args.save_plots:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df, attr_df = load_data()
    
    # Run influence network analysis
    if not args.content_only:
        print("\n" + "="*60)
        print("INFLUENCE NETWORK ANALYSIS")
        print("="*60)
        
        _attr = prep_attr_df(attr_df)
        _attr_trim = filter_top_per_dimension(_attr, top_k=args.top_k, min_share=0.0)
        
        G_influence = build_graph_classic(_attr_trim)
        
        influence_save_path = os.path.join(args.output_dir, "influence_network.png") if args.save_plots else None
        draw_static_classic(G_influence, figsize=(12, 8), top_labels_per_dim=10, save_path=influence_save_path)
    
    # Run content network analysis
    if not args.influence_only:
        print("\n" + "="*60)
        print("CONTENT NETWORK ANALYSIS")
        print("="*60)
        
        df_use = prepare_article_data(df)
        term_scores = build_term_scores(attr_df)
        
        edges = build_publisher_term_edges(
            articles=df_use,
            term_table=term_scores,
            term_limit=args.term_limit,
            min_term_weight=0.0
        )
        
        G_content = build_bipartite_graph(edges)
        
        content_save_path = os.path.join(args.output_dir, "content_network.png") if args.save_plots else None
        draw_publisher_term_network(G_content, top_labels_per_type=12, figsize=(14, 9), save_path=content_save_path)
        
        # Show top associations
        print("\nTop Publisher-Term Associations:")
        print(edges.sort_values("weight", ascending=False)
                   .head(20)
                   .rename(columns={"publisher": "Publisher", "term": "Term", "weight": "AssocWeight"})
                   .to_string(index=False))
    
    # Run term ranking comparison
    print("\n" + "="*60)
    print("TERM RANKING COMPARISON")
    print("="*60)
    
    compare, top_chunk, top_global = compare_term_rankings(attr_df)
    
    print("\nTop 20 Terms — CHUNK-normalized:")
    print(top_chunk.to_string(index=False))
    
    print("\nTop 20 Terms — GLOBAL-normalized:")
    print(top_global.to_string(index=False))
    
    print("\nComparison (union of Top 20s) — rank/weight changes:")
    print(compare[["value", "rank_chunk", "rank_global", "rank_delta",
                   "credit_share_chunk", "credit_share_global", "share_delta"]].to_string(index=False))
    
    print("\n" + "="*60)
    print("ANALYSIS COMPLETE")
    print("="*60)


if __name__ == "__main__":
    main()
