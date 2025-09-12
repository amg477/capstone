"""
---------------------------------
FILE INFORMATION: 
1. Adds 'is_conversion' as bool value 
2. Builds Markov paths using 'tag_name' as the journey key
3. Computes removal-effect for chosen dimensions. 
4. Saves a csv file of credit/credit-share/rating (1-5)

---------------------------------
CONVERSION DEFINITION: 
-- source_name_type is in {National News, Government, Wires, General News, Regional News, Trade News}
-- circulation_size is in the top quartile (>= 75th percentile)
-- vipr_weight is in top quartile 
-- sentiment_score absolute value >= |75| (strongly positive or negative )

---------------------------------
RUN: 
python attribution.py
"""

# Import packages 
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, identity
from scipy.sparse.linalg import spsolve
from pathlib import Path


#### Load data & clean 
df = pd.read_csv('..data/final_dataset_sampled.csv')

df["load_date"] = pd.to_datetime(df["load_date"], errors = 'coerce')

#### Define conversion
reliable_types = {"National News", "Government", "Wires", "General News", "Regional News", "Trade News"}

circ_cutoff = df["circulation_size"].quantile(0.75)
vipr_cutoff = df["vipr_weight"].quantile(0.75)

# Conversion flag
df["conversion"] = (
    df["source_type_name"].isin(reliable_types) 
    | (df["circulation_size"] >= circ_cutoff)
    | (df["vipr_weight"] >= vipr_cutoff)
    | (df["sentiment_score"].abs() >= 75)
)

#### Markov Attribution (Removal Effect)
def markov_attribution(
        df: pd.DataFrame,
        path_key: str = "tag_name", 
        time_col: str = "load_date", 
        state_col: str = "publication_name", 
        weight_col: str = "vipr_weight", 
        conv_flag: str = "conversion",
) -> pd.DataFrame: 
    
    """
    Computes removal-effect credit for 'state_col' given a conversion flag. 
    Returns: state, credit_, credit_share
    """

    # keep only needed columns
    keep = [path_key, time_col, state_col, weight_col, conv_flag]
    dff = df[keep].dropna(subset=[path_key, time_col, state_col])
    dff[time_col] = pd.to_datetime(dff[time_col], errors="coerce")
    dff = dff.dropna(subset=[time_col])
    dff = dff.sort_values([path_key, time_col])

    # weight 
    w = pd.to_numeric(dff[weight_col], errors='coerce').fillna(1.0)
    dff["_w"] = w

    START, CONV = "<START>", "<CONV>"

    # group into sequences
    paths = []
    for k, g in dff.groupby(path_key, sort=False):
        seq = []
        last = None
        for _, row in g.iterrows():
            if row[state_col] != last:  
                seq.append(row[state_col])
                last = row[state_col]
            if row[conv_flag]:
                seq.append(CONV)
                break
        if seq:
            paths.append((seq, g[weight_col].sum()))

    if not paths:
        return pd.DataFrame(columns=["state","credit","credit_share"])

    # encode states
    all_states = sorted({s for seq,_ in paths for s in seq if s != CONV})
    states = all_states + [CONV]
    sid = {s:i for i,s in enumerate(states)}

    # build transition counts
    src, dst, wt = [], [], []
    for seq, w in paths:
        src.append(len(states)) 
        dst.append(sid[seq[0]])
        wt.append(w)
        for a,b in zip(seq[:-1], seq[1:]):
            src.append(sid[a]); dst.append(sid[b]); wt.append(w)

    n_all = len(states)+1
    T = coo_matrix((wt,(src,dst)), shape=(n_all,n_all)).tocsr()
    rs = np.asarray(T.sum(axis=1)).ravel()
    rs[rs==0] = 1
    P = T.multiply(1/rs[:,None]).tocsr()

    transient = np.array([i for i,s in enumerate(states) if s != CONV], int)
    absorbing = np.array([sid[CONV]], int)
    order = np.r_[ [len(states)], transient, absorbing ]
    P = P[order][:, order]

    t_slice = slice(1, 1+len(transient))
    a_slice = slice(1+len(transient), None)
    Q = P[t_slice, t_slice]
    R = P[t_slice, a_slice]
    P0 = P[0, t_slice]

    I = identity(Q.shape[0], format="csr")
    N = spsolve(I - Q, identity(Q.shape[0]))
    baseline = float((P0 @ N @ R).A.ravel()[0])

    credits = []
    for i in range(Q.shape[0]):
        Q2 = Q.copy().tocsr(); R2 = R.copy().tocsr()
        Q2.data[Q2.indptr[i]:Q2.indptr[i+1]] = 0.0
        R2.data[R2.indptr[i]:R2.indptr[i+1]] = 0.0
        N2 = spsolve(I - Q2, identity(Q2.shape[0]))
        conv2 = float((P0 @ N2 @ R2).A.ravel()[0])
        credits.append(max(0.0, baseline - conv2))

    credits = np.array(credits)
    share = credits / credits.sum() if credits.sum() else np.zeros_like(credits)

    return pd.DataFrame({
        "state": [states[i] for i in transient],
        "credit": credits,
        "credit_share": share
    }).sort_values("credit", ascending=False, ignore_index=True)

#### Run Attribution for Key Dimensions 
dimensions = [
    "tag_name","source_feed_name","feed_name","author_name",
    "source_type_name","channel_name","genre","publisher_name",
    "publication_name","circulation_size","sentiment_score",
    "source_type","sentiment_band"
]

results = []
for dim in dimensions:
    print(f"[run] {dim}")
    try:
        out = markov_attribution(df, state_col=dim)
        out["dimension"] = dim
        results.append(out)
    except Exception as e:
        print(f"   skipped {dim}: {e}")

all_attr = pd.concat(results, ignore_index=True)
all_attr.to_csv("data/attribution_data.csv", index = False)

print("\n[OK] Saved -> data/attribution_data.csv")