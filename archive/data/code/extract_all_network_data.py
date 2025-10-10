#!/usr/bin/env python3
"""
Complete Network Data Extraction for Streamlit App
Extracts all network tables from attribution analysis and saves as CSV files
"""

import pandas as pd
import networkx as nx
import numpy as np
import re
import os
from typing import Tuple, Optional

# Constants from the notebook
SINK_LABEL = "<CONV>"
DIMENSION_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]
KIND_MARKERS = {"item": "o", "term": "^"}

def prep_attr_df(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare attribution DataFrame with proper data types"""
    required = {"kind","dimension","value","credit","credit_share","rating","rating_pct"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"attr_df missing required columns: {sorted(missing)}")
    
    out = df.copy()
    out["kind"] = out["kind"].astype(str)
    out["dimension"] = out["dimension"].astype(str)
    out["value"] = out["value"].astype(str)
    for c in ("credit","credit_share","rating_pct"):
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(0).astype(int)
    return out

def filter_top_per_dimension(df: pd.DataFrame, top_k: Optional[int] = 50, min_share: float = 0.0) -> pd.DataFrame:
    """Filter to top K items per dimension by credit share"""
    x = df.copy()
    if min_share > 0:
        x = x[x["credit_share"] >= min_share]
    if top_k and top_k > 0:
        x = (
            x.sort_values(["dimension","credit_share"], ascending=[True, False])
             .groupby("dimension", as_index=False)
             .head(top_k)
        )
    return x.reset_index(drop=True)

def extract_influence_nodes(attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract influence network nodes DataFrame
    
    Returns:
        DataFrame with columns: value, kind, dimension, credit, credit_share, rating, rating_pct, color, marker, node_size
    """
    df = prep_attr_df(attr_df)
    
    # Create nodes table - one row per (value, kind, dimension) combination
    nodes_df = (
        df.groupby(["value","kind","dimension"], as_index=False)
          .agg({
              "credit": "sum",
              "credit_share": "sum", 
              "rating": "max",
              "rating_pct": "max"
          })
    )
    
    # Add additional metadata
    dims = sorted(df["dimension"].unique())
    dim_to_color = {d: DIMENSION_COLORS[i % len(DIMENSION_COLORS)] for i, d in enumerate(dims)}
    
    nodes_df["color"] = nodes_df["dimension"].map(dim_to_color).fillna("#7f7f7f")
    nodes_df["marker"] = nodes_df["kind"].map(KIND_MARKERS).fillna("o")
    
    # Calculate node size based on credit_share
    nodes_df["node_size"] = nodes_df["credit_share"].apply(lambda x: max(80, float(x) * 2200))
    
    return nodes_df

def extract_influence_edges(attr_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract influence network edges DataFrame
    
    Returns:
        DataFrame with columns: source, target, weight
    """
    df = prep_attr_df(attr_df)
    
    # Create edges table - connections from values to CONV sink
    edges_df = (
        df.groupby("value", as_index=False)["credit_share"]
          .sum()
          .rename(columns={"credit_share": "weight"})
    )
    
    # Add target as CONV sink
    edges_df["target"] = SINK_LABEL
    edges_df = edges_df.rename(columns={"value": "source"})
    
    # Filter out zero-weight edges
    edges_df = edges_df[edges_df["weight"] > 0].reset_index(drop=True)
    
    return edges_df

def extract_term_comparison_tables(attr_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract term comparison tables (chunk vs global normalization)
    
    Returns:
        Tuple of (top_chunk_df, top_global_df, comparison_df)
    """
    # Prepare terms data
    terms_df = attr_df[attr_df["kind"] == "term"].copy()
    terms_df["credit"] = pd.to_numeric(terms_df["credit"], errors="coerce").fillna(0.0)
    terms_df["credit_share"] = pd.to_numeric(terms_df["credit_share"], errors="coerce").fillna(0.0)
    terms_df["value"] = terms_df["value"].astype(str)
    
    # Add global term share
    total = terms_df["credit"].sum()
    terms_df["credit_share_global"] = terms_df["credit"] / total if total > 0 else 0.0
    
    # Top terms by chunk credit share
    top_chunk = (
        terms_df.groupby("value", as_index=False)["credit_share"]
                .sum()
                .sort_values("credit_share", ascending=False)
                .head(20)
    )
    top_chunk.insert(0, "rank", range(1, len(top_chunk) + 1))
    top_chunk = top_chunk.rename(columns={"rank": "rank_chunk", "credit_share": "credit_share_chunk"})
    
    # Top terms by global credit share
    top_global = (
        terms_df.groupby("value", as_index=False)["credit_share_global"]
                .sum()
                .sort_values("credit_share_global", ascending=False)
                .head(20)
    )
    top_global.insert(0, "rank", range(1, len(top_global) + 1))
    top_global = top_global.rename(columns={"rank": "rank_global", "credit_share_global": "credit_share_global"})
    
    # Merge to compare ranks & values
    compare = pd.merge(top_chunk, top_global, on="value", how="outer")
    
    # Fill missing ranks with large number
    max_rank = 999
    compare["rank_chunk"] = compare["rank_chunk"].fillna(max_rank).astype(int)
    compare["rank_global"] = compare["rank_global"].fillna(max_rank).astype(int)
    compare["credit_share_chunk"] = compare["credit_share_chunk"].fillna(0.0)
    compare["credit_share_global"] = compare["credit_share_global"].fillna(0.0)
    
    # Calculate rank and share deltas
    compare["rank_delta"] = compare["rank_global"] - compare["rank_chunk"]
    compare["share_delta"] = compare["credit_share_global"] - compare["credit_share_chunk"]
    
    # Sort by global rank first, then chunk rank
    compare = compare.sort_values(["rank_global", "rank_chunk"]).reset_index(drop=True)
    
    return top_chunk, top_global, compare

def extract_publisher_term_edges(articles_df: pd.DataFrame, attr_df: pd.DataFrame, 
                                term_limit: int = 200, min_term_weight: float = 0.001) -> pd.DataFrame:
    """
    Extract publisher-term association edges
    
    Returns:
        DataFrame with columns: publisher, term, weight
    """
    # Get term scores from attribution data
    term_scores = (
        attr_df.query("kind=='term'")
               .loc[:, ["value","credit_share"]]
               .dropna()
               .rename(columns={"value":"term","credit_share":"term_weight"})
               .sort_values("term_weight", ascending=False)
               .head(term_limit)
               .reset_index(drop=True)
    )
    
    if term_scores.empty:
        return pd.DataFrame(columns=["publisher","term","weight"])
    
    # Create term patterns for matching
    def compile_term_patterns(terms):
        return [re.compile(rf"\b{re.escape(t)}\b", flags=re.IGNORECASE) for t in terms]
    
    def best_term_for_text(text, ranked_terms, patterns):
        for (t, w), pat in zip(ranked_terms, patterns):
            if pat.search(text):
                return t, float(w)
        return None, 0.0
    
    ranked_terms = list(term_scores.itertuples(index=False, name=None))
    patterns = compile_term_patterns([t for t,_ in ranked_terms])
    
    # Build edges
    edges = {}
    for _, row in articles_df.iterrows():
        text = f"{row['processed_headline']} {row['processed_body']}"
        best_term, best_weight = best_term_for_text(text, ranked_terms, patterns)
        
        if best_term is None or best_weight < min_term_weight:
            continue
            
        publisher = row["publisher_name"]
        key = (publisher, best_term)
        edges[key] = edges.get(key, 0.0) + best_weight
    
    if not edges:
        return pd.DataFrame(columns=["publisher","term","weight"])
    
    pub, term, weight = zip(*[(p,t,v) for (p,t),v in edges.items()])
    return pd.DataFrame({"publisher": pub, "term": term, "weight": weight})

def extract_community_summary(edges_df: pd.DataFrame, top_publishers: int = 30, 
                             top_terms: int = 30, edge_quantile: float = 0.75) -> pd.DataFrame:
    """
    Extract community summary from publisher-term edges
    
    Returns:
        DataFrame with community analysis
    """
    # Filter edges
    ef = edges_df.copy()
    
    # Remove generic terms
    generic_terms = {"online", "video", "link"}
    ef = ef[~ef["term"].isin(generic_terms)].reset_index(drop=True)
    
    # Keep top publishers and terms
    pub_strength = ef.groupby("publisher")["weight"].sum().sort_values(ascending=False)
    term_strength = ef.groupby("term")["weight"].sum().sort_values(ascending=False)
    keep_pubs = set(pub_strength.head(top_publishers).index)
    keep_terms = set(term_strength.head(top_terms).index)
    ef = ef[ef["publisher"].isin(keep_pubs) & ef["term"].isin(keep_terms)].reset_index(drop=True)
    
    # Filter by edge weight quantile
    if not ef.empty:
        w_cut = ef["weight"].quantile(edge_quantile)
        ef = ef[ef["weight"] >= w_cut].reset_index(drop=True)
    
    if ef.empty:
        return pd.DataFrame(columns=["Community", "Top_Terms", "Top_Publishers", 
                                    "Representative_Edge", "Edges_in_Community", "Total_Weight"])
    
    # Build graph for community detection
    G = nx.Graph()
    for _, row in ef.iterrows():
        G.add_edge(row["publisher"], row["term"], weight=row["weight"])
    
    if G.number_of_edges() == 0:
        return pd.DataFrame(columns=["Community", "Top_Terms", "Top_Publishers", 
                                    "Representative_Edge", "Edges_in_Community", "Total_Weight"])
    
    # Community detection
    from networkx.algorithms.community import greedy_modularity_communities
    comms = list(greedy_modularity_communities(G, weight="weight"))
    
    # Create community summary
    rows = []
    for i, community in enumerate(comms):
        pubs = [n for n in community if n in ef["publisher"].values]
        terms = [n for n in community if n in ef["term"].values]
        
        if not pubs or not terms:
            continue
            
        # Get edges within this community
        ec = ef[ef["publisher"].isin(pubs) & ef["term"].isin(terms)]
        
        if ec.empty:
            continue
            
        # Top terms and publishers by strength
        term_strength = ec.groupby("term")["weight"].sum().sort_values(ascending=False)
        pub_strength = ec.groupby("publisher")["weight"].sum().sort_values(ascending=False)
        
        # Representative edge
        top_edge = ec.sort_values("weight", ascending=False).iloc[0]
        example = f"{top_edge['publisher']} — {top_edge['term']} ({top_edge['weight']:.1f})"
        
        rows.append({
            "Community": i,
            "Top_Terms": ", ".join(term_strength.head(5).index),
            "Top_Publishers": ", ".join(pub_strength.head(5).index),
            "Representative_Edge": example,
            "Edges_in_Community": len(ec),
            "Total_Weight": ec["weight"].sum()
        })
    
    return pd.DataFrame(rows).sort_values(["Total_Weight", "Edges_in_Community"], ascending=False)

def extract_all_network_data(attr_df: pd.DataFrame, articles_df: pd.DataFrame, 
                           output_dir: str = "/Users/annaglass/capstone/capstone/data/processed") -> dict:
    """
    Extract all network data and save to CSV files
    
    Returns:
        Dictionary containing all extracted DataFrames
    """
    print("Extracting network data...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract all data
    influence_nodes = extract_influence_nodes(attr_df)
    influence_edges = extract_influence_edges(attr_df)
    top_chunk, top_global, term_comparison = extract_term_comparison_tables(attr_df)
    pub_term_edges = extract_publisher_term_edges(articles_df, attr_df)
    community_summary = extract_community_summary(pub_term_edges)
    
    # Save all DataFrames
    influence_nodes.to_csv(f"{output_dir}/influence_nodes.csv", index=False)
    influence_edges.to_csv(f"{output_dir}/influence_edges.csv", index=False)
    top_chunk.to_csv(f"{output_dir}/top_terms_chunk.csv", index=False)
    top_global.to_csv(f"{output_dir}/top_terms_global.csv", index=False)
    term_comparison.to_csv(f"{output_dir}/term_comparison.csv", index=False)
    pub_term_edges.to_csv(f"{output_dir}/publisher_term_edges.csv", index=False)
    community_summary.to_csv(f"{output_dir}/community_summary.csv", index=False)
    
    # Print summary
    print(f"\nSaved network data to {output_dir}:")
    print(f"  - influence_nodes.csv: {len(influence_nodes)} nodes")
    print(f"  - influence_edges.csv: {len(influence_edges)} edges")
    print(f"  - top_terms_chunk.csv: {len(top_chunk)} terms")
    print(f"  - top_terms_global.csv: {len(top_global)} terms")
    print(f"  - term_comparison.csv: {len(term_comparison)} comparisons")
    print(f"  - publisher_term_edges.csv: {len(pub_term_edges)} associations")
    print(f"  - community_summary.csv: {len(community_summary)} communities")
    
    return {
        "influence_nodes": influence_nodes,
        "influence_edges": influence_edges,
        "top_terms_chunk": top_chunk,
        "top_terms_global": top_global,
        "term_comparison": term_comparison,
        "publisher_term_edges": pub_term_edges,
        "community_summary": community_summary
    }

def main():
    """Main function to run the extraction"""
    print("Loading data...")
    
    # Load the data
    attr_df = pd.read_csv("/Users/annaglass/capstone/capstone/data/processed/attribution_all_scored.csv")
    articles_df = pd.read_csv("/Users/annaglass/capstone/capstone/data/processed/final_model_dataset.csv")
    
    print(f"Loaded attribution data: {len(attr_df)} rows")
    print(f"Loaded articles data: {len(articles_df)} rows")
    
    # Extract and save all network data
    network_data = extract_all_network_data(attr_df, articles_df)
    
    # Display sample data
    print("\nSample influence nodes:")
    print(network_data["influence_nodes"].head())
    
    print("\nSample publisher-term edges:")
    print(network_data["publisher_term_edges"].head())
    
    print("\nSample term comparison:")
    print(network_data["term_comparison"].head())
    
    print("\nSample community summary:")
    print(network_data["community_summary"].head())
    
    print("\nData extraction complete! All files saved to data/processed/")

if __name__ == "__main__":
    main()
