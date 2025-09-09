#!/usr/bin/env python3
"""
Spontaneous Forest Fire Sampling (SFFS) on a bipartite graph (Source <-> Tag).

Why bipartite?
- Avoids the O(k^2) blow-up of projecting tags to all source-source pairs.
- Scales better on large datasets.

3-level enforcement:
- --method pretrim : restrict to k=3 hops around seeds, then SFFS (requires seeds)
- --method depth   : depth-limited SFFS with max_depth=3 (no seeds required)

Outputs (in --out_dir):
- bip_edges.csv          : bipartite edges (src, dst=tag:<value>, weight)
- sffs_nodes.csv         : sampled node IDs (tag nodes are prefixed with 'tag:')
- sffs_edges.csv         : edges in the sampled graph
- sffs_sample.gpickle    : pickled NetworkX subgraph (saved with Python 'pickle')

Example (no seeds, depth-limited):
    python sampling.py \
      --data_csv processed_data.csv \
      --source_col source_unique_id \
      --tag_col tag_name \
      --weight \
      --min_tag_freq 2 \
      --max_sources_per_tag 2000 \
      --method depth \
      --p_burn 0.4 \
      --jump_every 3 \
      --max_nodes 50000 \
      --restore_edges \
      --out_dir . \
      --random_seed 13
"""

from __future__ import annotations
from typing import Iterable, Optional, Tuple, Dict, Any, List, Set
import argparse
import os
import random
import pickle

import pandas as pd
import networkx as nx


# -------------------------
# Build bipartite edges
# -------------------------

def build_bipartite_edges(
    df: pd.DataFrame,
    source_col: str = "source_unique_id",
    tag_col: str = "tag_name",
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

    tag_freq = pairs.groupby(tag_col)["weight"].sum().rename("tag_total_w").reset_index()
    pairs = pairs.merge(tag_freq, on=tag_col, how="left")
    pairs = pairs[pairs["tag_total_w"] >= min_tag_freq].drop(columns=["tag_total_w"])

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

def build_graph_from_bip_edges(
    df_edges: pd.DataFrame,
    directed: bool = False,
) -> nx.Graph:
    G = nx.DiGraph() if directed else nx.Graph()
    for u, v, w in df_edges[["src", "dst", "weight"]].itertuples(index=False, name=None):
        G.add_edge(u, v, weight=float(w))
    return G


def restrict_to_k_hop_subgraph(
    G: nx.Graph,
    seeds: Iterable[Any],
    k: int = 3,
    undirected_view_for_distance: bool = True,
) -> nx.Graph:
    seeds = [s for s in seeds if s in G]
    if not seeds:
        raise ValueError("None of the provided seeds are in the graph.")
    H = G.to_undirected() if (undirected_view_for_distance and G.is_directed()) else G
    keep: Set[Any] = set()
    for s in seeds:
        lengths = nx.single_source_shortest_path_length(H, s, cutoff=k)
        keep.update(lengths.keys())
    return G.subgraph(keep).copy()


def _sffs_core(
    G: nx.Graph,
    seeds: Optional[Iterable[Any]],
    p_burn: float,
    jump_every: int,
    max_nodes: Optional[int],
    max_steps: Optional[int],
    restore_edges: bool,
    depth_limited: bool = False,
    max_depth: int = 3,
    rng: Optional[random.Random] = None,
) -> Dict[str, Any]:
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
        if depth_limited:
            frontier: List[Tuple[Any, int]] = [(start, 0)]
        else:
            frontier: List[Any] = [start]

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

    if restore_edges:
        Gs = G.subgraph(sampled).copy()
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
    }
    if depth_limited:
        stats["max_depth"] = max_depth
    return {"nodes": sampled, "edges": edges_out, "G_sample": Gs, "stats": stats}


# -------------------------
# CLI
# -------------------------

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run SFFS on a bipartite (Source <-> Tag) graph with a 3-level limit.")
    ap.add_argument("--data_csv", required=True, help="Path to processed_data.csv")
    ap.add_argument("--source_col", default="source_unique_id")
    ap.add_argument("--tag_col", default="tag_name")
    ap.add_argument("--weight", action="store_true", help="Weight edges by frequency of (source,tag).")
    ap.add_argument("--min_tag_freq", type=int, default=1, help="Drop tags with total freq < this value.")
    ap.add_argument("--max_sources_per_tag", type=int, default=None, help="Cap # of distinct sources per tag.")
    ap.add_argument("--method", choices=["pretrim", "depth"], default="pretrim",
                    help="pretrim: k=3 hops around seeds; depth: depth-limited SFFS (max_depth=3).")
    ap.add_argument("--seeds", default="", help="Comma-separated seeds (source ids or tag values).")
    ap.add_argument("--seeds_file", default=None, help="File with one seed per line.")
    ap.add_argument("--seed_type", choices=["source", "tag", "auto"], default="auto",
                    help="Interpret seeds as source ids, tag values, or auto (try both).")
    ap.add_argument("--p_burn", type=float, default=0.4)
    ap.add_argument("--jump_every", type=int, default=3)
    ap.add_argument("--max_nodes", type=int, default=50000)
    ap.add_argument("--max_steps", type=int, default=2000000)
    ap.add_argument("--restore_edges", action="store_true")
    ap.add_argument("--directed", action="store_true", help="Usually False for bipartite.")
    ap.add_argument("--out_dir", default=".")
    ap.add_argument("--random_seed", type=int, default=42)
    return ap.parse_args()


def _load_seeds(args: argparse.Namespace) -> List[str]:
    seeds: List[str] = []
    if args.seeds:
        seeds += [s.strip() for s in args.seeds.split(",") if s.strip()]
    if args.seeds_file and os.path.exists(args.seeds_file):
        try:
            df = pd.read_csv(args.seeds_file, header=None)
            seeds += df.iloc[:, 0].astype(str).tolist()
        except Exception:
            with open(args.seeds_file, "r", encoding="utf-8") as f:
                seeds += [line.strip() for line in f if line.strip()]
    # unique, preserve order
    seen, uniq = set(), []
    for s in seeds:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def _derive_bipartite_seeds(
    seeds_raw: List[str],
    source_ids: Set[str],
    tag_values: Set[str],
    seed_type: str,
) -> List[str]:
    """
    Map user-provided seeds to actual node ids in the bipartite graph.
    - source seeds stay as-is
    - tag seeds become 'tag:<value>'
    - auto tries both
    """
    out: List[str] = []
    for s in seeds_raw:
        if seed_type == "source":
            if s in source_ids: out.append(s)
        elif seed_type == "tag":
            if s in tag_values: out.append("tag:" + s)
        else:  # auto
            added = False
            if s in source_ids:
                out.append(s); added = True
            if s in tag_values:
                out.append("tag:" + s); added = True
            if not added:
                # Keep as-is; may exist already in graph (e.g., user pre-added "tag:" prefix)
                out.append(s)
    seen, uniq = set(), []
    for s in out:
        if s not in seen:
            seen.add(s); uniq.append(s)
    return uniq


def main():
    args = parse_args()
    rng = random.Random(args.random_seed)
    os.makedirs(args.out_dir, exist_ok=True)

    # 1) Load data
    df = pd.read_csv(args.data_csv)

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
        raise SystemExit("No bipartite edges built. Check columns/filters.")

    # 3) Graph
    G = build_graph_from_bip_edges(bip_edges, directed=args.directed)

    # 4) Seeds -> bipartite node ids
    seeds_raw = _load_seeds(args)
    sources_set = set(bip_edges["src"].unique())
    tags_set = set(bip_edges["dst"].str.removeprefix("tag:").unique())
    seeds = _derive_bipartite_seeds(seeds_raw, sources_set, tags_set, args.seed_type)
    print(f"[Info] Loaded {len(seeds)} seed(s). Example: {seeds[:5]}")

    # 5) Run SFFS with 3-level enforcement
    if args.method == "pretrim":
        if not seeds:
            raise ValueError("pretrim requires at least one valid seed (use --seeds/--seeds_file + --seed_type).")
        print("[Info] Pre-restricting to k=3 hops around seeds.")
        G_run = restrict_to_k_hop_subgraph(G, seeds, k=3, undirected_view_for_distance=True)
        res = _sffs_core(
            G_run, seeds=seeds, p_burn=args.p_burn, jump_every=args.jump_every,
            max_nodes=args.max_nodes, max_steps=args.max_steps,
            restore_edges=args.restore_edges, depth_limited=False, rng=rng,
        )
        res["stats"]["method"] = "pretrim"
        res["stats"]["k"] = 3
    else:
        print("[Info] Using depth-limited SFFS with max_depth=3.")
        res = _sffs_core(
            G, seeds=seeds if seeds else None, p_burn=args.p_burn, jump_every=args.jump_every,
            max_nodes=args.max_nodes, max_steps=args.max_steps,
            restore_edges=args.restore_edges, depth_limited=True, max_depth=3, rng=rng,
        )
        res["stats"]["method"] = "depth"
        res["stats"]["max_depth"] = 3

    # 6) Save outputs
    nodes_path = os.path.join(args.out_dir, "sffs_nodes.csv")
    sffs_edges_path = os.path.join(args.out_dir, "sffs_edges.csv")
    g_path = os.path.join(args.out_dir, "sffs_sample.gpickle")

    pd.Series(sorted(res["nodes"])).to_csv(nodes_path, index=False)
    pd.DataFrame(res["edges"], columns=["src", "dst"]).to_csv(sffs_edges_path, index=False)

    # Save graph with Python pickle (works on NetworkX 3.x)
    with open(g_path, "wb") as f:
        pickle.dump(res["G_sample"], f)

    # 7) Stats
    print("\n=== SFFS RUN STATS ===")
    for k, v in res["stats"].items():
        print(f"{k}: {v}")
    print(f"\nSaved:\n- {bip_edges_path}\n- {nodes_path}\n- {sffs_edges_path}\n- {g_path}")


if __name__ == "__main__":
    main()