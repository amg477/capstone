#!/usr/bin/env python3
import argparse, pickle
import pandas as pd, networkx as nx

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graph_path", default="sampling/sffs_sample.gpickle")
    ap.add_argument("--nodes_csv", default="sampling/sffs_nodes.csv")
    ap.add_argument("--data_csv", default="data/processed_data.csv")
    ap.add_argument("--out_csv", default="node_dataset.csv")
    args = ap.parse_args()

    with open(args.graph_path, "rb") as f:
        G = pickle.load(f)

    nodes = pd.read_csv(args.nodes_csv, header=None, names=["node"])
    source_nodes = nodes[~nodes["node"].astype(str).str.startswith("tag:")]["node"].tolist()
    tag_nodes = [n for n in G.nodes if str(n).startswith("tag:")]

    # Project to source-only graph S (edge weight=#shared tags)
    S = nx.Graph()
    S.add_nodes_from(source_nodes)
    for t in tag_nodes:
        nbrs = [u for u in G.neighbors(t) if u in source_nodes]
        for i in range(len(nbrs)):
            u = nbrs[i]
            for j in range(i+1, len(nbrs)):
                v = nbrs[j]
                w = S[u][v]["weight"] + 1 if S.has_edge(u,v) else 1
                S.add_edge(u, v, weight=w)

    # Features
    deg = dict(S.degree())
    wdeg = dict(S.degree(weight="weight"))
    pr = nx.pagerank(S, weight="weight") if S.number_of_edges() else {n:0 for n in S}
    btw = nx.betweenness_centrality(S, weight=lambda u,v,d: 1.0/max(d.get("weight",1),1)) if S.number_of_edges() else {n:0 for n in S}
    kcore = nx.core_number(S) if S.number_of_edges() else {n:0 for n in S}
    clust = nx.clustering(S, weight="weight") if S.number_of_edges() else {n:0 for n in S}
    tag_degree = {s: sum(1 for nb in G.neighbors(s) if str(nb).startswith("tag:")) for s in source_nodes}

    X_net = pd.DataFrame({
        "source_unique_id": source_nodes,
        "deg": [deg.get(n,0) for n in source_nodes],
        "wdeg": [wdeg.get(n,0) for n in source_nodes],
        "pagerank": [pr.get(n,0) for n in source_nodes],
        "betweenness": [btw.get(n,0) for n in source_nodes],
        "kcore": [kcore.get(n,0) for n in source_nodes],
        "clustering": [clust.get(n,0) for n in source_nodes],
        "tag_degree": [tag_degree.get(n,0) for n in source_nodes],
    })

    df = pd.read_csv(args.data_csv, usecols=["source_unique_id","tag_name"])
    agg = (df.groupby("source_unique_id", as_index=False)
             .agg(n_articles=("tag_name","size"),
                  n_unique_tags=("tag_name", pd.Series.nunique)))

    dataset = (X_net.merge(agg, on="source_unique_id", how="left")
                    .fillna({"n_articles":0, "n_unique_tags":0}))
    dataset.to_csv(args.out_csv, index=False)
    print(f"[OK] Wrote {args.out_csv} with {len(dataset)} rows, {dataset.shape[1]} columns.")

if __name__ == "__main__":
    main()