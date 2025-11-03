from __future__ import annotations
import re
import numpy as np
import pandas as pd
import networkx as nx
from typing import Iterable, Tuple, Dict, List
from networkx.algorithms.community import greedy_modularity_communities
import matplotlib.pyplot as plt

# ---------- fast matching utilities (token set + bigrams) ----------
def bigram_strings(tokens: List[str]) -> set[str]:
    if len(tokens) < 2:
        return set()
    return {f"{a} {b}" for a, b in zip(tokens[:-1], tokens[1:])}

def prepare_whitelist_sets(whitelist_terms: Iterable[str], term_weight_tbl: pd.DataFrame
) -> Tuple[set[str], set[str], Dict[str, float]]:
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

# ---------- main pipeline ----------
def build_edges_fast(
    df: pd.DataFrame,
    attr_df: pd.DataFrame,
    whitelist_terms: Iterable[str],
    publisher_col: str = "publisher_name",
    min_term_weight: float = 0.0,
    use_max_term_credit_first: bool = True
) -> pd.DataFrame:
    """
    Returns edges DataFrame: [publisher, term, weight]
    Edge weight = max_term_credit (if available & >0) else (global_term_weight * vipr_weight).
    """
    # global term weights from attr_df
    term_w = (attr_df.query("kind=='term'")
              .loc[:, ["value","credit_share"]]
              .rename(columns={"value":"term","credit_share":"term_weight"}))
    term_w["term"] = term_w["term"].astype(str)
    wl = [str(t).strip() for t in whitelist_terms if str(t).strip()]
    if not wl:
        return pd.DataFrame(columns=["publisher","term","weight"])

    # keep whitelist terms, fill missing with 1.0
    keep = term_w[term_w["term"].isin(wl)]
    if keep.empty:
        keep = pd.DataFrame({"term": wl, "term_weight": 1.0})
    else:
        missing = set(wl) - set(keep["term"])
        if missing:
            keep = pd.concat([keep, pd.DataFrame({"term": list(missing), "term_weight": 1.0})], ignore_index=True)

    uni_wl, bi_wl, TW_MAP = prepare_whitelist_sets(wl, keep)

    # ensure fields
    use = df[[publisher_col, "processed_headline", "processed_body"]].copy()
    use[publisher_col] = use[publisher_col].replace("", "Unknown").fillna("Unknown").astype(str)
    for c in ["processed_headline", "processed_body"]:
        use[c] = use[c].fillna("").astype(str).str.lower()
    if "max_term_credit" not in df.columns:
        df = df.assign(max_term_credit=0.0)
    if "vipr_weight" not in df.columns:
        df = df.assign(vipr_weight=1.0)

    # build edges
    edges: Dict[Tuple[str,str], float] = {}
    for idx, r in use.iterrows():
        pub = r[publisher_col]
        tokens = (r["processed_headline"] + " " + r["processed_body"]).split()
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

def filter_edges(
    edges: pd.DataFrame,
    top_publishers: int = 30,
    top_terms: int = 30,
    generic_terms: Iterable[str] = (),
    edge_percentile_cutoff: float = 0.75
) -> pd.DataFrame:
    """Filter to strongest nodes and drop the weakest edges by percentile within subset."""
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

def build_graph(edges: pd.DataFrame) -> nx.Graph:
    G = nx.Graph()
    for _, r in edges.iterrows():
        p, t, w = r["publisher"], r["term"], float(r["weight"])
        G.add_node(p, ntype="publisher")
        G.add_node(t, ntype="term")
        G.add_edge(p, t, weight=w)
    return G

def community_map(G: nx.Graph) -> Dict[str, int]:
    if G.number_of_edges() == 0:
        return {}
    comms = list(greedy_modularity_communities(G, weight="weight"))
    node2c: Dict[str, int] = {}
    for i, cset in enumerate(comms):
        for n in cset:
            node2c[n] = i
    return node2c

def draw_network(
    G: nx.Graph,
    node2c: Dict[str, int],
    labels_per_type: int = 10,
    figsize: Tuple[int,int] = (12, 8),
    title: str = "Content Network: Publishers ↔ High-Impact Terms"
):
    import matplotlib.pyplot as plt
    # node strength
    strength = {n: sum(G[n][nbr].get("weight", 0.0) for nbr in G.neighbors(n)) for n in G.nodes()}
    svals = np.array(list(strength.values())) if strength else np.array([1.0])
    s_log = np.log1p(svals)
    smin, smax = float(s_log.min()), float(s_log.max())
    def nsize(n, base=110, span=2200):
        if smax - smin < 1e-9: return base + span * 0.5
        x = (np.log1p(strength[n]) - smin) / (smax - smin)
        return base + span * x
    # edges -> percentile widths
    ws = np.array([float(G[u][v].get("weight", 0.0)) for u,v in G.edges()])
    if len(ws) == 0:
        p10, p90 = 0, 1
    else:
        p10, p90 = np.quantile(ws, [0.10, 0.90])
    denom = max(p90 - p10, 1e-9)
    widths = [0.8 + 4.0 * max(0, min(1, (G[u][v]["weight"] - p10)/denom)) for u,v in G.edges()]

    # colors by community
    base_colors = ["#1f77b4","#ff7f0e","#2ca02c","#d62728","#9467bd",
                   "#8c564b","#e377c2","#7f7f7f","#bcbd22","#17becf"]
    node_colors = [base_colors[node2c.get(n, 0) % len(base_colors)] for n in G.nodes()]

    # layout
    pos = nx.spring_layout(G, k=0.6, seed=42, weight="weight")

    pubs  = [n for n,d in G.nodes(data=True) if d.get("ntype") == "publisher"]
    terms = [n for n,d in G.nodes(data=True) if d.get("ntype") == "term"]

    plt.figure(figsize=figsize)
    nx.draw_networkx_edges(G, pos, width=widths, alpha=0.20, edge_color="#666")
    nx.draw_networkx_nodes(G, pos, nodelist=pubs,
                           node_size=[nsize(n) for n in pubs],
                           node_color=[node_colors[list(G.nodes()).index(n)] for n in pubs],
                           node_shape="o", linewidths=0.8, edgecolors="#333", alpha=0.95)
    nx.draw_networkx_nodes(G, pos, nodelist=terms,
                           node_size=[nsize(n) for n in terms],
                           node_color=[node_colors[list(G.nodes()).index(n)] for n in terms],
                           node_shape="^", linewidths=0.8, edgecolors="#333", alpha=0.95)

    pubs_top  = sorted(pubs,  key=lambda n: strength[n], reverse=True)[:labels_per_type]
    terms_top = sorted(terms, key=lambda n: strength[n], reverse=True)[:labels_per_type]
    labels = {n: n for n in pubs_top + terms_top}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=9)

    plt.title(title, fontsize=14, pad=10)
    plt.figtext(0.5, 0.02,
        "Communities colored • Circle=publisher • Triangle=term • Size≈log(weighted degree) • Edge≈pct-normalized",
        ha="center", fontsize=9, color="#444")
    plt.axis("off"); plt.tight_layout()
    return plt.gcf()

def summarize_communities_with_examples(
    G: nx.Graph, edges_df: pd.DataFrame, node2c: Dict[str, int],
    top_k: int = 5, top_edges: int = 1
) -> pd.DataFrame:
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

def render_content_network(
    df: pd.DataFrame,
    attr_df: pd.DataFrame,
    whitelist_terms: Iterable[str],
    publisher_col: str = "publisher_name",
    min_term_weight: float = 0.0,
    top_publishers: int = 30,
    top_terms: int = 30,
    generic_terms: Iterable[str] = (),
    edge_percentile_cutoff: float = 0.75,
    labels_per_type: int = 10,
    figsize: Tuple[int,int] = (12,8),
    title: str = "Content Network: Publishers ↔ High-Impact Terms",
    use_max_term_credit_first: bool = True,
) -> Tuple[plt.Figure, pd.DataFrame, pd.DataFrame]:
    """
    Returns:
      fig: matplotlib Figure
      edges_filt: filtered edges used to draw
      comm_summary: community summary table
    """
    edges = build_edges_fast(
        df=df, attr_df=attr_df, whitelist_terms=whitelist_terms,
        publisher_col=publisher_col, min_term_weight=min_term_weight,
        use_max_term_credit_first=use_max_term_credit_first
    )
    edges_filt = filter_edges(
        edges, top_publishers=top_publishers, top_terms=top_terms,
        generic_terms=generic_terms, edge_percentile_cutoff=edge_percentile_cutoff
    )
    G = build_graph(edges_filt)
    node2c = community_map(G)
    fig = draw_network(G, node2c, labels_per_type=labels_per_type, figsize=figsize, title=title)
    comm_tbl = summarize_communities_with_examples(G, edges_filt, node2c, top_k=5, top_edges=1)
    return fig, edges_filt, comm_tbl
    


