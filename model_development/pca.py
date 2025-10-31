"""
pca.py

Thin runner that imports the PCA pipeline (Option A) and executes it.
- Finds optimal K via elbow + silhouette (sampled).
- Clusters PCA scores and builds the influencer table.
- Writes ONLY:
    - influencer_table.csv  (requires person_row_df with {'row_index','persons'})

Usage:
    python pca.py --final final_df.csv --person-rows person_row_df.csv --outdir outputs
"""

# 
from run_pca_influencer import run_pca_analysis, PCAConfig

import argparse
import os
import sys
import pandas as pd


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute PCA → KMeans → Influencer pipeline (save ONLY influencer_table.csv).")
    ap.add_argument("--final", required=True, help="Path to final_df.csv")
    ap.add_argument("--person-rows", default=None, help="Path to person_row_df.csv (required to produce influencer_table.csv)")
    ap.add_argument("--outdir", default=".", help="Output directory (default: current)")
    ap.add_argument("--k-min", type=int, default=2, help="Minimum K (default: 2)")
    ap.add_argument("--k-max", type=int, default=10, help="Maximum K (default: 10)")
    ap.add_argument("--sil-sample", type=int, default=5000, help="Silhouette sample size (default: 5000)")
    ap.add_argument("--cap-components", type=int, default=0, help="Cap PCA components (0 = off)")
    ap.add_argument("--use-minibatch", action="store_true", help="Use MiniBatchKMeans for elbow curve (speed on very large data)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    # Load data
    final_df = pd.read_csv(args.final)
    person_row_df = pd.read_csv(args.person_rows) if args.person_rows else None

    # Configure pipeline
    cfg = PCAConfig(
        k_min=args.k_min,
        k_max=args.k_max,
        random_state=42,
        sil_sample=args.sil_sample,
        cap_components_at=(None if args.cap_components == 0 else args.cap_components),
        use_minibatch_for_elbow=args.use_minibatch,
    )

    # Run analysis
    out = run_pca_analysis(final_df, person_row_df=person_row_df, config=cfg)

    # Prepare output directory
    os.makedirs(args.outdir, exist_ok=True)

    # === Save ONLY the influencer table ===
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

    # Minimal console summary (no other files written)
    print("\n=== PCA Influencer Pipeline Complete (no other CSVs written) ===")
    print(f"Optimal K*: {out['optimal_k']}")
    print(f"PCA components kept: {out['n_components']}")
    print("Features used:", ", ".join(out["features_used"]))
    print("Cluster sizes:")
    print(out["df_pca"]["cluster"].value_counts().sort_index())


if __name__ == "__main__":
    main()