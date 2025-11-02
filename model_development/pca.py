#!/usr/bin/env python3
"""
pca.py — Self-contained PCA → KMeans → Influencer table

- Loads final dataset (CSV or Parquet)
- Selects numeric + binary features
- Scales, runs PCA (retain ~90% variance by default, or cap)
- Chooses K via silhouette over a configurable range
- Fits final KMeans
- Builds influencer_table from persons_by_row (if provided)
- Writes ONLY influencer_table.csv

Usage examples:
  python pca.py \
    --final /Users/annaglass/capstone/capstone/data_storage/final_data/final_dataset_with_attribution.parquet \
    --person-rows /Users/annaglass/capstone/capstone/data_storage/final_data/persons_by_row.csv \
    --outdir /Users/annaglass/capstone/capstone/data_storage/final_data

  # CSV inputs also supported:
  python pca.py --final final_df.csv --person-rows persons_by_row.csv --outdir outputs
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


# ----------------------------- Config & Core ---------------------------------
@dataclass
class PCAConfig:
    k_min: int = 2
    k_max: int = 10
    random_state: int = 42
    sil_sample: int = 5000              # sample size for silhouette scoring
    cap_components_at: Optional[int] = None  # hard cap on number of PCs
    variance_target: float = 0.90       # retain ~90% variance if no cap


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
    # Candidate numeric columns from your notebook
    candidate_numeric = [
        "vipr_score", "vipr_weight", "hit_strength", "sentiment_score",
        "circulation_size", "headline_token_count", "body_token_count", "token_count"
    ]
    binary_flags = [c for c in ["has_person", "is_conversion"] if c in df.columns]
    features = [c for c in candidate_numeric if c in df.columns] + binary_flags

    # Cast flags to int (0/1) if present
    for c in binary_flags:
        df[c] = df[c].astype(int)

    if not features:
        raise ValueError("No usable feature columns found. "
                         "Expected at least one of: "
                         "vipr_score, vipr_weight, hit_strength, sentiment_score, "
                         "circulation_size, headline_token_count, body_token_count, token_count, "
                         "has_person, is_conversion.")

    X = df[features].to_numpy()
    return X, features


def _choose_components(X_scaled: np.ndarray, cfg: PCAConfig) -> Tuple[PCA, int, np.ndarray]:
    """Fit full PCA to compute cumulative variance, then refit to target/cap."""
    pca_full = PCA().fit(X_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_90 = int(np.searchsorted(cum_var, cfg.variance_target) + 1)
    if cfg.cap_components_at is not None:
        n_keep = min(n_90, int(cfg.cap_components_at))
    else:
        n_keep = n_90

    pca = PCA(n_components=n_keep, random_state=cfg.random_state)
    X_pca = pca.fit_transform(X_scaled)
    return pca, n_keep, X_pca


def _pick_k_by_silhouette(X_pca: np.ndarray, cfg: PCAConfig) -> Tuple[int, pd.DataFrame]:
    k_values = list(range(cfg.k_min, cfg.k_max + 1))
    inertias, sil_scores = [], []

    # Sample for silhouette if very large
    n = X_pca.shape[0]
    if n > cfg.sil_sample:
        rng = np.random.default_rng(cfg.random_state)
        idx = rng.choice(n, size=cfg.sil_sample, replace=False)
        X_sil = X_pca[idx]
        sample_idx = idx
    else:
        X_sil = X_pca
        sample_idx = None

    for k in k_values:
        km = KMeans(n_clusters=k, random_state=cfg.random_state, n_init=10)
        labels_full = km.fit_predict(X_pca)
        inertias.append(km.inertia_)

        if sample_idx is not None:
            labels_sil = labels_full[sample_idx]
        else:
            labels_sil = labels_full

        # Silhouette needs at least 2 clusters and less than n_samples
        if k >= 2 and k < X_sil.shape[0]:
            sil = silhouette_score(X_sil, labels_sil)
        else:
            sil = np.nan
        sil_scores.append(sil)

    df = pd.DataFrame({"k": k_values, "inertia": inertias, "silhouette": sil_scores})
    # Choose k by max silhouette (ties → smallest k)
    df_valid = df.dropna(subset=["silhouette"])
    if df_valid.empty:
        # Fallback: smallest k
        optimal_k = k_values[0]
    else:
        optimal_k = int(df_valid.loc[df_valid["silhouette"].idxmax(), "k"])
    return optimal_k, df


def _split_persons(s: object) -> List[str]:
    if pd.isna(s):
        return []
    return [p.strip() for p in str(s).split(",") if str(p).strip()]


def run_pca_analysis(
    final_df: pd.DataFrame,
    person_row_df: Optional[pd.DataFrame],
    config: PCAConfig,
) -> Dict[str, object]:
    # Ensure stable key and keep a few identifier columns if present
    df = _ensure_row_index(final_df.copy())
    keep_id_cols = [c for c in ["row_index", "author_name", "tag_name"] if c in df.columns]

    # Build features
    X, features_used = _feature_matrix(df)

    # Scale & PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca, n_components, X_pca = _choose_components(X_scaled, config)
    pc_cols = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    df_pca = pd.DataFrame(X_pca, columns=pc_cols)
    for c in keep_id_cols:
        df_pca[c] = df[c].values

    # Choose K
    optimal_k, k_diag = _pick_k_by_silhouette(X_pca, config)

    # Final KMeans
    km = KMeans(n_clusters=optimal_k, random_state=config.random_state, n_init=10)
    df_pca["cluster"] = km.fit_predict(X_pca)

    # Interpret clusters with original metrics (subset)
    metrics_cols = [c for c in ["vipr_score", "vipr_weight", "sentiment_score", "circulation_size"] if c in df.columns]
    interp_df = df[["row_index"] + metrics_cols].merge(df_pca[["row_index", "cluster"]], on="row_index", how="left")
    cluster_summary = interp_df.groupby("cluster")[metrics_cols].mean().reset_index()

    # Influencer table (optional)
    influencer_table = None
    if person_row_df is not None and isinstance(person_row_df, pd.DataFrame) and {"row_index", "persons"}.issubset(person_row_df.columns):
        merged = (
            person_row_df[["row_index", "persons"]]
            .merge(df_pca[["row_index", "cluster"]], on="row_index", how="left")
            .merge(df[["row_index"] + metrics_cols], on="row_index", how="left")
        )
        merged["person_list"] = merged["persons"].apply(_split_persons)
        exploded = merged.explode("person_list").dropna(subset=["person_list"])

        agg_dict = {m: "mean" for m in metrics_cols}
        agg_dict.update({"row_index": "count"})  # mention_count
        influencer_table = (
            exploded.groupby(["person_list", "cluster"])
            .agg(agg_dict)
            .rename(columns={"row_index": "mention_count"})
            .reset_index()
        )

        # Simple, interpretable cluster labels
        lab = cluster_summary.copy()

        def _label_row(r: pd.Series) -> str:
            vis = r.get("vipr_score", np.nan)
            tone = r.get("sentiment_score", np.nan)
            if pd.notna(vis) and pd.notna(tone):
                vis_med = np.nanmedian(cluster_summary.get("vipr_score", pd.Series([vis])))
                if vis >= vis_med and tone >= 0:
                    return "High Visibility / Positive Tone"
                if vis >= vis_med and tone < 0:
                    return "High Visibility / Negative Tone"
                if vis < vis_med and tone >= 0:
                    return "Lower Visibility / Positive Tone"
                return "Lower Visibility / Negative Tone"
            return "Unlabeled"

        lab["cluster_label"] = lab.apply(_label_row, axis=1)
        influencer_table = influencer_table.merge(lab[["cluster", "cluster_label"]], on="cluster", how="left")

        # Rank by mention_count, then vipr_score if present
        sort_cols, sort_asc = ["mention_count"], [False]
        if "vipr_score" in influencer_table.columns:
            sort_cols.append("vipr_score"); sort_asc.append(False)
        influencer_table = influencer_table.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)

    return {
        "df_pca": df_pca,
        "cluster_summary": cluster_summary,
        "k_diagnostics": k_diag,
        "optimal_k": optimal_k,
        "n_components": n_components,
        "features_used": features_used,
        "influencer_table": influencer_table,
    }


# ------------------------------ CLI Runner -----------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute PCA → KMeans → Influencer pipeline (save ONLY influencer_table.csv).")
    ap.add_argument("--final", required=True, help="Path to final_df (CSV or Parquet).")
    ap.add_argument("--person-rows", dest="person_rows", default=None, help="Path to persons_by_row CSV (optional, builds influencer_table).")
    ap.add_argument("--outdir", default=".", help="Output directory (default: current).")
    ap.add_argument("--k-min", type=int, default=2, help="Minimum K (default: 2).")
    ap.add_argument("--k-max", type=int, default=10, help="Maximum K (default: 10).")
    ap.add_argument("--sil-sample", type=int, default=5000, help="Silhouette sample size (default: 5000).")
    ap.add_argument("--cap-components", type=int, default=0, help="Cap PCA components (0 = off).")
    ap.add_argument("--variance-target", type=float, default=0.90, help="Variance target for PCA if no cap (default: 0.90).")
    ap.add_argument("--random-state", type=int, default=42, help="Random seed (default: 42).")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    final_df = _read_any(args.final)
    person_row_df = None
    if args.person_rows:
        # persons_by_row is expected to be CSV; Parquet supported too if provided
        person_row_df = _read_any(args.person_rows)

    # Configure pipeline
    cfg = PCAConfig(
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=args.random_state,
        sil_sample=args.sil_sample,
        cap_components_at=(None if args.cap_components == 0 else int(args.cap_components)),
        variance_target=float(args.variance_target),
    )

    # Run analysis
    out = run_pca_analysis(final_df, person_row_df=person_row_df, config=cfg)

    # Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)

    # Save ONLY the influencer table
    if out["influencer_table"] is not None and not out["influencer_table"].empty:
        out_path = os.path.join(args.outdir, "influencer_table.csv")
        out["influencer_table"].to_csv(out_path, index=False)
        print(f"\nSaved influencer_table.csv to: {os.path.abspath(out_path)}")
    else:
        print(
            "\nNo influencer_table produced. Ensure person_row_df is provided and includes "
            "{'row_index','persons'} with at least one non-empty 'persons' value.",
            file=sys.stderr,
        )

    # Minimal console summary
    print("\n=== PCA Influencer Pipeline Complete (no other CSVs written) ===")
    print(f"Optimal K*: {out['optimal_k']}")
    print(f"PCA components kept: {out['n_components']}")
    print("Features used:", ", ".join(out["features_used"]))
    print("Cluster sizes:")
    try:
        print(out["df_pca"]["cluster"].value_counts().sort_index())
    except Exception:
        print("(cluster counts unavailable)")


if __name__ == "__main__":
    main()