#!/usr/bin/env python3
"""
pca.py — One-command PCA → KMeans → Influencer table

Run:
    python pca.py

Automatically uses:
  final_df:     /Users/annaglass/capstone/capstone/data_storage/final_data/final_dataset_with_attribution.parquet
  person_rows:  /Users/annaglass/capstone/capstone/data_storage/final_data/persons_by_row.csv
  outdir:       /Users/annaglass/capstone/capstone/data_storage/final_data
"""

from pathlib import Path
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# ---------------------------------------------------------------------------
# Default paths (change here if your layout moves)
# ---------------------------------------------------------------------------
ROOT = Path("/Users/annaglass/capstone/capstone")
FINAL_PARQUET = ROOT / "data_storage/final_data/final_dataset_with_attribution.parquet"
PERSONS_CSV   = ROOT / "data_storage/final_data/persons_by_row.csv"
OUTDIR        = ROOT / "data_storage/final_data"

# ---------------------------------------------------------------------------
@dataclass
class PCAConfig:
    k_min: int = 2
    k_max: int = 10
    random_state: int = 42
    sil_sample: int = 5000
    cap_components_at: Optional[int] = None   # e.g., 10; None means use variance target
    variance_target: float = 0.90             # keep ~90% variance

# ---------------------------------------------------------------------------
def _read_any(path: str | Path) -> pd.DataFrame:
    path = str(path)
    if path.lower().endswith((".parquet", ".pq")):
        return pd.read_parquet(path)
    return pd.read_csv(path)

def _ensure_row_index(df: pd.DataFrame) -> pd.DataFrame:
    if "row_index" not in df.columns:
        df = df.reset_index(drop=True).reset_index().rename(columns={"index": "row_index"})
    return df

def _feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
    candidate_numeric = [
        "vipr_score", "vipr_weight", "hit_strength", "sentiment_score",
        "circulation_size", "headline_token_count", "body_token_count", "token_count",
    ]
    binary_flags = [c for c in ["has_person", "is_conversion"] if c in df.columns]
    features = [c for c in candidate_numeric if c in df.columns] + binary_flags
    for c in binary_flags:
        df[c] = df[c].astype(int)
    if not features:
        raise ValueError(
            "No usable numeric columns found. Expected at least one of: "
            "vipr_score, vipr_weight, hit_strength, sentiment_score, "
            "circulation_size, headline_token_count, body_token_count, token_count, "
            "has_person, is_conversion."
        )
    X = df[features].to_numpy()
    return X, features

def _choose_components(X_scaled: np.ndarray, cfg: PCAConfig):
    """Fit PCA and decide #components using variance target (and optional cap)."""
    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum_var, cfg.variance_target) + 1)
    n_keep = min(n_90, cfg.cap_components_at) if cfg.cap_components_at else n_90
    pca = PCA(n_components=n_keep, random_state=cfg.random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, n_keep, X_pca

def _pick_k_by_silhouette(X_pca: np.ndarray, cfg: PCAConfig):
    """Grid-search K by silhouette on a sample (for speed)."""
    ks, sils = [], []
    n = X_pca.shape[0]
    idx = None
    if n > cfg.sil_sample:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(n, size=cfg.sil_sample, replace=False)
    for k in range(cfg.k_min, cfg.k_max + 1):
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
        labels = km.fit_predict(X_pca)
        sil = silhouette_score(X_pca if idx is None else X_pca[idx],
                               labels if idx is None else labels[idx])
        ks.append(k)
        sils.append(sil)
    df = pd.DataFrame({"k": ks, "silhouette": sils})
    best_k = int(df.loc[df["silhouette"].idxmax(), "k"])
    return best_k, df

def _split_persons(s):
    if pd.isna(s): 
        return []
    return [p.strip() for p in str(s).split(",") if p.strip()]

# ---------------------------------------------------------------------------
def run_pipeline(final_path=FINAL_PARQUET, person_path=PERSONS_CSV, outdir=OUTDIR):
    print(f"\n[LOAD] {final_path}")
    df = _read_any(final_path)
    df = _ensure_row_index(df)
    print(f"Rows loaded: {len(df):,}")

    persons_df = None
    if Path(person_path).exists():
        persons_df = _read_any(person_path)
        print(f"Loaded persons file ({len(persons_df):,} rows)")
    else:
        print("No persons file found — influencer table will be skipped.")

    cfg = PCAConfig()
    X, features = _feature_matrix(df)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca, n_components, X_pca = _choose_components(X_scaled, cfg)
    print(f"PCA components kept: {n_components}")

    best_k, _ = _pick_k_by_silhouette(X_pca, cfg)
    print(f"Optimal K: {best_k}")

    km = KMeans(n_clusters=best_k, random_state=cfg.random_state, n_init=10)
    df["cluster"] = km.fit_predict(X_pca)

    # --- Influencer table ---
    if persons_df is not None and {"row_index", "persons"}.issubset(persons_df.columns):
        merged = persons_df.merge(df[["row_index", "cluster"]], on="row_index", how="left")
        merged["person_list"] = merged["persons"].apply(_split_persons)
        exploded = merged.explode("person_list").dropna(subset=["person_list"])
        influencer = (
            exploded.groupby(["person_list", "cluster"])
            .size()
            .reset_index(name="mention_count")
            .sort_values("mention_count", ascending=False)
            .reset_index(drop=True)
        )
        out_path = Path(outdir) / "influencer_table.csv"
        Path(outdir).mkdir(parents=True, exist_ok=True)
        influencer.to_csv(out_path, index=False)
        print(f"[SAVED] Influencer table → {out_path}")
    else:
        print("Skipped influencer table: persons file missing or malformed (needs columns: row_index, persons).")

    print("\n=== PCA pipeline complete ===")
    print(f"Features used: {', '.join(features)}")
    try:
        print(df["cluster"].value_counts().sort_index())
    except Exception:
        pass

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    run_pipeline()