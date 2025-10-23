#!/usr/bin/env python3
"""
Run Spontaneous Forest Fire Sampling (SFFS) on a bipartite graph (Source <-> Tag)
built from your text dataset, and output ONE final sampled CSV that preserves the
exact schema of the original input.

Default input: text_processed_data.csv (in the same folder).
Default mode: depth-limited SFFS with max_depth=3 (no seeds required).

Outputs (in --out_dir, default "."):
- bip_edges.csv                     : bipartite edges (src, dst=tag:<value>, weight)
- sffs_nodes.csv                    : sampled nodes
- sffs_edges.csv                    : sampled edges (src,dst)
- sffs_sample.gpickle               : pickled NetworkX subgraph (via Python 'pickle')
- text_processed_data_SFFS_sampled.csv : FINAL sampled dataset with all original columns

Example (just run; assumes this file and CSV are in the same folder):
    python sampling.py
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any, List, Set
import argparse
import os
import random
import pickle

import pandas as pd
import networkx as nx

"""
RUN INSTRUCTIONS (IN TERMINAL): 
-------
cd data/processed
python sampling.py
"""

# -------------------------
# Config defaults for your file/columns
# -------------------------

DEFAULT_DATA_CSV = "data/text_processed_data.csv"
DEFAULT_SOURCE_COL = "source_unique_id"
DEFAULT_TAG_COL = "tag_name"

# -------------------------
# Build bipartite edges
# -------------------------

def build_bipartite_edges(
    df: pd.DataFrame,
    source_col: str,
    tag_col: str,
    weight: bool = True,
    min_tag_freq: int = 1,
    max_sources_per_tag: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return DataFrame with columns: src, dst, weight
      - src: source id (string)
      - dst: 'tag:<tag_value>'
      - weight: frequency of (source, tag) across rows (1 if weight=False)

    Filters:
      - drop tags with total frequency < min_tag_freq
      - cap very popular tags at max_sources_per_tag sources (deterministic order)
    """
    if source_col not in df.columns or tag_col not in df.columns:
        raise ValueError(f"CSV must include '{source_col}' and '{tag_col}' columns.")

    d = df[[source_col, tag_col]].dropna()
    if d.empty:
        return pd.DataFrame(columns=["src", "dst", "weight"])

    d[source_col] = d[source_col].astype(str)
    d[tag_col] = d[tag_col].astype(str)

    if weight:
        pairs = (
            d.groupby([source_col, tag_col], as_index=False)
             .size()
             .rename(columns={"size": "weight"})
        )
    else:
        pairs = d.drop_duplicates([source_col, tag_col]).copy()
        pairs["weight"] = 1

    # Tag frequency across all sources (to filter rare tags)
    tag_freq = pairs.groupby(tag_col)["weight"].sum().rename("tag_total_w").reset_index()
    pairs = pairs.merge(tag_freq, on=tag_col, how="left")
    pairs = pairs[pairs["tag_total_w"] >= min_tag_freq].drop(columns=["tag_total_w"])

    # Cap overly popular tags if requested
    if max_sources_per_tag is not None:
        pairs = (
            pairs.sort_values([tag_col, source_col])  # deterministic
                 .groupby(tag_col, as_index=False)
                 .head(max_sources_per_tag)
        )

    pairs = pairs.rename(columns={source_col: "src", tag_col: "dst"})
    pairs["dst"] = "tag:" + pairs["dst"]
    return pairs[["src", "dst", "weight"]].reset_index(drop=True)


# -------------------------
# Graph + SFFS utilities
# -------------------------

def build_graph_from_bip_edges(df_edges: pd.DataFrame, directed: bool = False) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for u, v, w in df_edges[["src", "dst", "weight"]].itertuples(index=False, name=None):
        G.add_edge(u, v, weight=float(w))
    return G


def _sffs_core(
    G: nx.Graph,
    seeds: Optional[Iterable[Any]],
    p_burn: float,
    jump_every: int,
    max_nodes: Optional[int],
    max_steps: Optional[int],
    restore_edges: bool,
    depth_limited: bool = True,
    max_depth: int = 3,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
    """Core SFFS traversal; depth-limited by default to enforce 3 levels."""
    if rng is None:
        rng = random.Random()
    neigh = G.successors if G.is_directed() else G.neighbors

    seed_pool: List[Any] = []
    if seeds:
        seed_pool = [s for s in seeds if s in G]
        rng.shuffle(seed_pool)

    sampled: Set[Any] = set()
    traversed: List[Tuple[Any, Any]] = []
    fires = 0
    steps = 0

    def next_start() -> Optional[Any]:
        while seed_pool:
            s = seed_pool.pop()
            if s not in sampled:
                return s
        if len(sampled) < G.number_of_nodes():
            nodes_tuple = tuple(G.nodes)
            for _ in range(50):
                c = rng.choice(nodes_tuple)
                if c not in sampled:
                    return c
            for n in G.nodes:
                if n not in sampled:
                    return n
        return None

    while True:
        if (max_nodes is not None and len(sampled) >= max_nodes) or \
           (max_steps is not None and steps >= max_steps):
            break

        start = next_start()
        if start is None:
            break
        fires += 1

        steps_this_fire = 0
        frontier: List[Tuple[Any, int]] = [(start, 0)] if depth_limited else [start]

        while frontier:
            if (max_nodes is not None and len(sampled) >= max_nodes) or \
               (max_steps is not None and steps >= max_steps):
                break

            if depth_limited:
                cur, depth = frontier.pop(0)
            else:
                cur = frontier.pop(0)

            if cur in sampled:
                continue
            sampled.add(cur)

            if (not depth_limited) or (depth < max_depth):
                for nb in (n for n in neigh(cur) if n not in sampled):
                    if rng.random() < p_burn:
                        traversed.append((cur, nb))
                        if depth_limited:
                            frontier.append((nb, depth + 1))
                        else:
                            frontier.append(nb)

            steps += 1
            steps_this_fire += 1
            if steps_this_fire >= jump_every:
                break

    # Build output graph
    if restore_edges:
        Gs = G.subgraph(sampled).copy()     # induce edges among sampled nodes
        edges_out = list(Gs.edges())
    else:
        Gs = nx.DiGraph() if G.is_directed() else nx.Graph()
        Gs.add_nodes_from(sampled)
        Gs.add_edges_from(traversed)
        edges_out = traversed

    stats = {
        "fires_started": fires,
        "steps": steps,
        "p_burn": p_burn,
        "jump_every": jump_every,
        "restore_edges": restore_edges,
        "directed": G.is_directed(),
        "sampled_nodes": len(sampled),
        "sampled_edges": len(edges_out),
        "max_depth": max_depth if depth_limited else None,
        "method": "depth" if depth_limited else "pretrim",
    }
    return {"nodes": sampled, "edges": edges_out, "G_sample": Gs, "stats": stats}


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SFFS sampling and one-shot final dataset filter.")
    # paths / columns
    ap.add_argument("--data_csv", default=DEFAULT_DATA_CSV, help="Input CSV (default: text_processed_data.csv)")
    ap.add_argument("--source_col", default=DEFAULT_SOURCE_COL, help="Column for source node IDs")
    ap.add_argument("--tag_col", default=DEFAULT_TAG_COL, help="Column for tag values")
    ap.add_argument("--out_dir", default=".", help="Output directory (default: current folder)")
    # graph construction controls
    ap.add_argument("--weight", action="store_true", help="Weight edges by (source,tag) frequency")
    ap.add_argument("--min_tag_freq", type=int, default=1, help="Drop tags with total freq < this value")
    ap.add_argument("--max_sources_per_tag", type=int, default=None, help="Cap # of sources per tag")
    # sffs controls (depth-limited default)
    ap.add_argument("--p_burn", type=float, default=0.4)
    ap.add_argument("--jump_every", type=int, default=3)
    ap.add_argument("--max_nodes", type=int, default=50000)
    ap.add_argument("--max_steps", type=int, default=2000000)
    ap.add_argument("--restore_edges", action="store_true")
    ap.add_argument("--directed", action="store_true", help="Treat graph as directed (usually False)")
    ap.add_argument("--random_seed", type=int, default=42)
    return ap.parse_args()


# -------------------------
# Main
# -------------------------

def main():
    args = parse_args()
    rng = random.Random(args.random_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load your text dataset
    data_path = args.data_csv if os.path.isabs(args.data_csv) else os.path.join(os.getcwd(), args.data_csv)
    df = pd.read_csv(data_path)

    # 2) Build bipartite edges with filters
    bip_edges = build_bipartite_edges(
        df,
        source_col=args.source_col,
        tag_col=args.tag_col,
        weight=args.weight,
        min_tag_freq=args.min_tag_freq,
        max_sources_per_tag=args.max_sources_per_tag,
    )
    bip_edges_path = os.path.join(args.out_dir, "bip_edges.csv")
    bip_edges.to_csv(bip_edges_path, index=False)
    print(f"[Info] Built {len(bip_edges)} bipartite edges -> {bip_edges_path}")

    if bip_edges.empty:
        raise SystemExit("No bipartite edges built. Check your columns or relax the filters.")

    # 3) Graph
    G = build_graph_from_bip_edges(bip_edges, directed=args.directed)

    # 4) Run SFFS (depth-limited to 3 levels; no seeds needed)
    res = _sffs_core(
        G, seeds=None, p_burn=args.p_burn, jump_every=args.jump_every,
        max_nodes=args.max_nodes, max_steps=args.max_steps,
        restore_edges=args.restore_edges, depth_limited=True, max_depth=3, rng=rng,
    )

    # 5) Save SFFS artifacts
    nodes_path = os.path.join(args.out_dir, "sffs_nodes.csv")
    sffs_edges_path = os.path.join(args.out_dir, "sffs_edges.csv")
    g_path = os.path.join(args.out_dir, "sffs_sample.gpickle")

    pd.Series(sorted(res["nodes"])).to_csv(nodes_path, index=False)
    pd.DataFrame(res["edges"], columns=["src", "dst"]).to_csv(sffs_edges_path, index=False)
    with open(g_path, "wb") as f:
        pickle.dump(res["G_sample"], f)

    # 6) Build ONE final sampled dataset with same columns as input
    #    Keep a row iff (source_col, tag_col) appears as an edge in the sampled graph.
    edges = pd.read_csv(sffs_edges_path)

    def edge_to_pair(u, v):
        su, sv = str(u), str(v)
        if su.startswith("tag:") and not sv.startswith("tag:"):
            return (sv, su.removeprefix("tag:"))
        if sv.startswith("tag:") and not su.startswith("tag:"):
            return (su, sv.removeprefix("tag:"))
        return None

    pairs = []
    for u, v in edges[["src", "dst"]].itertuples(index=False, name=None):
        p = edge_to_pair(u, v)
        if p:
            pairs.append(p)

    pairs_df = pd.DataFrame(pairs, columns=[args.source_col, args.tag_col]).drop_duplicates()
    sampled = df.merge(pairs_df, on=[args.source_col, args.tag_col], how="inner")

    final_name = os.path.splitext(os.path.basename(args.data_csv))[0] + "_SFFS_sampled.csv"
    final_path = os.path.join(args.out_dir, final_name)
    sampled.to_csv(final_path, index=False)

    # 7) Stats
    print("\n=== SFFS RUN STATS ===")
    for k, v in res["stats"].items():
        print(f"{k}: {v}")
    print(f"\nSaved:\n- {bip_edges_path}\n- {nodes_path}\n- {sffs_edges_path}\n- {g_path}\n- {final_path}  <-- FINAL DATASET")


if __name__ == "__main__":
    main()