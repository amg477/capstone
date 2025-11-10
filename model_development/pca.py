#!/usr/bin/env python3
"""
pca.py — Self-contained PCA → KMeans → Influencer table

- Loads final dataset (Parquet)
- Selects numeric + binary features
- Scales, runs PCA (retain ~90% variance by default, or cap)
- Chooses K via silhouette over a configurable range
- Fits final KMeans
- Builds influencer_table from persons_by_row (if provided)
- Writes ONLY influencer_table.parquet

Usage examples:
  # Run with default paths (no arguments needed):
  python pca.py

  # Override specific paths if needed:
  python pca.py \
    --final /path/to/final_dataset_with_attribution.parquet \
    --person-rows /path/to/persons_by_row.parquet \
    --outdir /path/to/output

  # CSV inputs also supported:
  python pca.py --final final_df.csv --person-rows persons_by_row.csv --outdir outputs
"""

from __future__ import annotations

import argparse
import os
import re
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
    """Read CSV file (or Parquet if CSV fails)."""
    path = str(path)
    # Try CSV first
    try:
        return pd.read_csv(path)
    except Exception:
        # If CSV fails, try Parquet as fallback
        return pd.read_parquet(path)


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


def _normalize_middle_initial(name: str) -> str:
    """
    Normalize middle initials to include periods.
    Example: "Robert F Kennedy" -> "Robert F. Kennedy"
    """
    if not name or pd.isna(name):
        return name
    
    name_str = str(name).strip()
    parts = name_str.split()
    
    if len(parts) < 2:
        return name_str
    
    normalized_parts = []
    for i, part in enumerate(parts):
        # Check if this is a single letter (likely a middle initial)
        # and it's not the first or last part
        if i > 0 and i < len(parts) - 1 and len(part) == 1 and part.isalpha():
            # Single letter in middle position - add period if not present
            if not part.endswith('.'):
                normalized_parts.append(part.upper() + '.')
            else:
                normalized_parts.append(part.upper())
        else:
            normalized_parts.append(part)
    
    return ' '.join(normalized_parts)


def _get_name_base(name: str) -> tuple[str, str]:
    """
    Extract base name components for comparison.
    Returns (first_name, last_name, middle_parts) tuple.
    Example: "Robert F. Kennedy" -> ("robert", "kennedy", ["f"])
    """
    if not name or pd.isna(name):
        return ("", "", [])
    
    parts = str(name).strip().split()
    if len(parts) < 2:
        return ("", parts[0].lower() if parts else "", [])
    
    first = parts[0].lower()
    last = parts[-1].lower()
    middle = parts[1:-1] if len(parts) > 2 else []
    
    return (first, last, [p.lower().rstrip('.') for p in middle])


def _canonicalize_person_name(name: str, surname_map: dict[str, str] = None) -> str:
    """
    Canonicalize a person name for grouping.
    Maps surnames to full names using surname_map if provided.
    Also handles specific known mappings.
    """
    if not name or pd.isna(name):
        return name
    
    name_str = str(name).strip()
    if not name_str:
        return name_str
    
    name_lower = name_str.lower()
    
    # Specific surname mappings
    if name_lower == "trump":
        return "Donald Trump"
    elif name_lower == "biden":
        return "Joe Biden"
    elif name_lower == "kennedy":
        return "Robert F. Kennedy"
    elif name_lower == "harris":
        return "Kamala Harris"
    
    # Use surname map if provided (for dynamic mapping like "Obama" -> "Barack Obama")
    if surname_map:
        name_tokens = name_str.split()
        if len(name_tokens) == 1:
            # Single token surname
            surname_lower = name_tokens[0].lower()
            if surname_lower in surname_map:
                return surname_map[surname_lower]
        elif len(name_tokens) == 2 and len(name_tokens[0]) == 1:
            # Initial + surname (e.g., "D. Trump")
            surname_lower = name_tokens[1].lower()
            if surname_lower in surname_map:
                return surname_map[surname_lower]
    
    # Normalize middle initials to include periods
    name_str = _normalize_middle_initial(name_str)
    
    # For full names, normalize to Title Case
    parts = name_str.split()
    if len(parts) >= 2:
        title_parts = []
        for part in parts:
            if '-' in part:
                hyphen_parts = [p.capitalize() for p in part.split('-')]
                title_parts.append('-'.join(hyphen_parts))
            elif '.' in part and len(part) == 2:
                # Middle initial with period
                title_parts.append(part[0].upper() + '.')
            else:
                title_parts.append(part.capitalize())
        return " ".join(title_parts)
    elif len(parts) == 1:
        return parts[0].capitalize()
    
    return name_str


def _build_surname_upgrade_map_from_table(influencer_table: pd.DataFrame) -> dict[str, str]:
    """
    Build a map from single-token surname -> dominant full name from the influencer_table.
    Only creates mappings when a full name with that surname is dominant.
    Uses mention_count to weight the dominance calculation.
    """
    if influencer_table is None or influencer_table.empty:
        return {}
    
    person_col = "person_list" if "person_list" in influencer_table.columns else None
    if not person_col:
        return {}
    
    # Count full names per surname (weighted by mention_count)
    surname_to_full_counts: dict[str, dict[str, int]] = {}
    
    # Get mention_count column if available, otherwise use 1 for each row
    count_col = "mention_count" if "mention_count" in influencer_table.columns else None
    
    for idx, row in influencer_table.iterrows():
        name = row[person_col]
        if pd.isna(name):
            continue
        
        name_str = str(name).strip()
        if not name_str:
            continue
        
        tokens = name_str.split()
        if len(tokens) >= 2:
            # Full name - extract surname
            surname = tokens[-1].lower()
            if surname not in surname_to_full_counts:
                surname_to_full_counts[surname] = {}
            
            # Use mention_count if available, otherwise 1
            count = int(row[count_col]) if count_col and pd.notna(row[count_col]) else 1
            surname_to_full_counts[surname][name_str] = surname_to_full_counts[surname].get(name_str, 0) + count
    
    # Build upgrade map: surname -> dominant full name
    upgrade_map: dict[str, str] = {}
    for surname, full_counts in surname_to_full_counts.items():
        if not full_counts:
            continue
        
        # Get total count and dominant full name
        total = sum(full_counts.values())
        dominant_full, dom_count = max(full_counts.items(), key=lambda kv: kv[1])
        
        # Only create mapping if dominant full name represents at least 60% of occurrences
        # and occurs at least 3 times
        if dom_count >= 3 and (dom_count / total) >= 0.6:
            upgrade_map[surname] = dominant_full
    
    return upgrade_map


def _filter_single_first_names(exploded_df: pd.DataFrame, final_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter out single-token first names that don't appear with a last name in the article text.
    For each single-token name, check if it appears with a last name in headline or body.
    If not, drop that row.
    """
    if exploded_df is None or exploded_df.empty:
        return exploded_df
    
    if final_df is None or final_df.empty:
        return exploded_df
    
    # Identify single-token names (first names only)
    single_token_mask = exploded_df["person_list"].apply(lambda x: len(str(x).split()) == 1)
    single_token_rows = exploded_df[single_token_mask].copy()
    
    if single_token_rows.empty:
        return exploded_df
    
    # Get text columns from final_df
    headline_col = "headline" if "headline" in final_df.columns else None
    body_col = "body" if "body" in final_df.columns else None
    
    if not headline_col and not body_col:
        # No text columns available, return as-is
        return exploded_df
    
    # Create a set of row_indexes to keep
    rows_to_keep = set(exploded_df[~single_token_mask].index.tolist())
    
    # Check each single-token name
    for idx, row in single_token_rows.iterrows():
        first_name = str(row["person_list"]).strip()
        if not first_name:
            continue
        
        row_index = row["row_index"]
        
        # Find the corresponding article in final_df
        article_row = final_df[final_df["row_index"] == row_index]
        if article_row.empty:
            # No matching article, drop this row
            continue
        
        article_text = ""
        if headline_col and pd.notna(article_row[headline_col].iloc[0]):
            article_text += " " + str(article_row[headline_col].iloc[0])
        if body_col and pd.notna(article_row[body_col].iloc[0]):
            article_text += " " + str(article_row[body_col].iloc[0])
        
        # Keep original case for pattern matching (need to check for capitalized last names)
        article_text_original = article_text
        article_text_lower = article_text.lower()
        first_name_lower = first_name.lower()
        
        # Check if first name appears with a last name (another capitalized word after it)
        # Pattern: "firstname Lastname" or "Firstname Lastname"
        # Look for the first name followed by a capitalized word (potential last name)
        # Pattern: firstname + space + capitalized word (in original case)
        pattern1 = rf"\b{re.escape(first_name_lower)}\s+[A-Z][a-z]+\b"
        # Pattern: Firstname + space + capitalized word (if first name is capitalized)
        pattern2 = rf"\b{first_name.capitalize()}\s+[A-Z][a-z]+\b"
        
        # Also check for "Lastname, Firstname" pattern
        pattern3 = rf"\b[A-Z][a-z]+\s*,\s*{re.escape(first_name_lower)}\b"
        pattern4 = rf"\b[A-Z][a-z]+\s*,\s*{first_name.capitalize()}\b"
        
        # Check if any pattern matches (use original case text)
        has_last_name = (
            re.search(pattern1, article_text_original) is not None or
            re.search(pattern2, article_text_original) is not None or
            re.search(pattern3, article_text_original) is not None or
            re.search(pattern4, article_text_original) is not None
        )
        
        if has_last_name:
            # First name appears with a last name, keep this row
            rows_to_keep.add(idx)
        # Otherwise, drop it (don't add to rows_to_keep)
    
    # Filter the dataframe to keep only valid rows
    filtered_df = exploded_df[exploded_df.index.isin(rows_to_keep)].copy()
    
    return filtered_df


def _group_and_aggregate_influencer_table(influencer_table: pd.DataFrame) -> pd.DataFrame:
    """
    Group influencer_table by canonicalized person name, emotion_body, and cluster_label.
    Aggregates metrics appropriately (sum counts, mean for numeric metrics).
    Combines name variants like "Robert F. Kennedy", "Robert F Kennedy", "Robert Kennedy" -> "Robert F. Kennedy"
    """
    if influencer_table is None or influencer_table.empty:
        return influencer_table
    
    # Build surname upgrade map from the table itself
    surname_map = _build_surname_upgrade_map_from_table(influencer_table)
    
    # First pass: canonicalize names and normalize middle initials
    influencer_table = influencer_table.copy()
    influencer_table["person_canonical"] = influencer_table["person_list"].apply(
        lambda x: _canonicalize_person_name(x, surname_map)
    )
    
    # Second pass: Group variants by name base (first + last name, ignoring middle parts)
    # Also handle single surnames that should be combined with full names
    # Create a mapping from (first, last) to the most complete canonical form
    name_base_to_canonical: dict[tuple, str] = {}
    
    # Build a map from surname -> full name for single surnames
    # Use the surname_map if available (has dominant full name), otherwise use any full name found
    surname_to_full: dict[str, str] = {}
    # First, use surname_map (has dominant full name based on frequency)
    if surname_map:
        for surname, full_name in surname_map.items():
            surname_to_full[surname] = full_name
    # Then, add any other full names we see (for surnames not in surname_map)
    for canonical_name in influencer_table["person_canonical"].dropna().unique():
        base = _get_name_base(canonical_name)
        # If it's a full name (has first name) and surname not already mapped, map it
        if base[0] and base[1] not in surname_to_full:  # Has first name and not already mapped
            surname_to_full[base[1]] = canonical_name
    
    for canonical_name in influencer_table["person_canonical"].dropna().unique():
        base = _get_name_base(canonical_name)
        # Group by first and last name only (ignore middle parts for grouping)
        base_key = (base[0], base[1])  # (first, last) - ignore middle for grouping
        
        # Special case: if this is a single surname (no first name), try to upgrade it
        if not base[0] and base[1] in surname_to_full:
            # Upgrade single surname to full name
            canonical_name = surname_to_full[base[1]]
            base = _get_name_base(canonical_name)
            base_key = (base[0], base[1])
        
        # If we haven't seen this base, or if this canonical form is more complete (has middle initial)
        if base_key not in name_base_to_canonical:
            name_base_to_canonical[base_key] = canonical_name
        else:
            # Prefer the form with middle initial/name if available
            existing = name_base_to_canonical[base_key]
            existing_base = _get_name_base(existing)
            
            # If current has middle parts and existing doesn't, use current
            if base[2] and not existing_base[2]:
                name_base_to_canonical[base_key] = canonical_name
            # If both have middle parts, prefer the one with period in initial
            elif base[2] and existing_base[2]:
                current_middle = ' '.join(base[2])
                existing_middle = ' '.join(existing_base[2])
                # Prefer form with period if available
                if '.' in current_middle and '.' not in existing_middle:
                    name_base_to_canonical[base_key] = canonical_name
            # If neither has middle parts, keep existing (or could prefer current, but existing is fine)
    
    # Map each canonical name to its most complete form
    def get_most_complete_form(name: str) -> str:
        if pd.isna(name):
            return name
        base = _get_name_base(name)
        # If single surname, try to upgrade to full name
        if not base[0] and base[1] in surname_to_full:
            upgraded = surname_to_full[base[1]]
            base = _get_name_base(upgraded)
        base_key = (base[0], base[1])  # Group by first + last only
        return name_base_to_canonical.get(base_key, name)
    
    influencer_table["person_canonical"] = influencer_table["person_canonical"].apply(get_most_complete_form)
    
    # Determine grouping columns
    group_cols = ["person_canonical"]
    if "emotion_body" in influencer_table.columns:
        group_cols.append("emotion_body")
    if "cluster_label" in influencer_table.columns:
        group_cols.append("cluster_label")
    
    # Build aggregation dictionary
    agg_dict = {}
    
    # Sum for count columns
    if "mention_count" in influencer_table.columns:
        agg_dict["mention_count"] = "sum"
    
    # Mean for numeric metrics
    numeric_cols = ["vipr_score", "vipr_weight", "sentiment_score", "circulation_size"]
    for col in numeric_cols:
        if col in influencer_table.columns:
            agg_dict[col] = "mean"
    
    # Keep cluster (should be consistent within cluster_label, but take first)
    if "cluster" in influencer_table.columns:
        agg_dict["cluster"] = "first"
    
    # Group and aggregate
    grouped = influencer_table.groupby(group_cols, as_index=False).agg(agg_dict)
    
    # Rename person_canonical back to person_list
    grouped = grouped.rename(columns={"person_canonical": "person_list"})
    
    # Re-sort by mention_count, then vipr_score if present
    sort_cols, sort_asc = ["mention_count"], [False]
    if "vipr_score" in grouped.columns:
        sort_cols.append("vipr_score")
        sort_asc.append(False)
    grouped = grouped.sort_values(sort_cols, ascending=sort_asc).reset_index(drop=True)
    
    return grouped


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
    # Add emotion_body if available (categorical, will use mode aggregation)
    emotion_col = "emotion_body" if "emotion_body" in df.columns else None
    
    interp_df = df[["row_index"] + metrics_cols].merge(df_pca[["row_index", "cluster"]], on="row_index", how="left")
    cluster_summary = interp_df.groupby("cluster")[metrics_cols].mean().reset_index()

    # Influencer table (optional)
    influencer_table = None
    if person_row_df is not None and isinstance(person_row_df, pd.DataFrame) and {"row_index", "persons"}.issubset(person_row_df.columns):
        # Prepare columns to merge
        merge_cols = ["row_index"] + metrics_cols
        if emotion_col:
            merge_cols.append(emotion_col)
        
        merged = (
            person_row_df[["row_index", "persons"]]
            .merge(df_pca[["row_index", "cluster"]], on="row_index", how="left")
            .merge(df[merge_cols], on="row_index", how="left")
        )
        merged["person_list"] = merged["persons"].apply(_split_persons)
        exploded = merged.explode("person_list").dropna(subset=["person_list"])

        # Filter out single first names that don't appear with a last name in article text
        exploded = _filter_single_first_names(exploded, df)

        # Aggregation: mean for numeric metrics, mode for emotion
        agg_dict = {m: "mean" for m in metrics_cols}
        agg_dict.update({"row_index": "count"})  # mention_count
        
        # Add emotion aggregation (mode - most common emotion)
        # Use a named function for proper pandas aggregation
        def get_mode(series):
            """Get the most common non-null value"""
            non_null = series.dropna()
            if len(non_null) == 0:
                return None
            mode_values = non_null.mode()
            return mode_values.iloc[0] if len(mode_values) > 0 else None
        
        if emotion_col:
            agg_dict[emotion_col] = get_mode
        
        influencer_table = (
            exploded.groupby(["person_list", "cluster"])
            .agg(agg_dict)
            .rename(columns={"row_index": "mention_count"})
            .reset_index()
        )
        
        # Handle emotion column - ensure it's properly formatted
        if emotion_col and emotion_col in influencer_table.columns:
            # Convert to string type and handle any aggregation artifacts
            influencer_table[emotion_col] = influencer_table[emotion_col].astype(str).replace('nan', None)

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

        # Group and aggregate by canonicalized person name, emotion_body, and cluster_label
        # This groups surnames with full names (e.g., "Obama" -> "Barack Obama")
        # while preserving separate rows for different emotion_body/cluster_label combinations
        influencer_table = _group_and_aggregate_influencer_table(influencer_table)

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
# Default paths (can be overridden via command-line arguments)
# Note: These files may have .parquet extension but contain CSV data
ROOT = Path(__file__).resolve().parent.parent  # .../capstone/capstone
DEFAULT_FINAL = ROOT / "data_storage" / "final_data" / "final_dataset_with_attribution.parquet"
DEFAULT_PERSON_ROWS = ROOT / "data_storage" / "final_data" / "persons_by_row.parquet"
DEFAULT_OUTDIR = ROOT / "data_storage" / "final_data"

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Execute PCA → KMeans → Influencer pipeline (save ONLY influencer_table.parquet).")
    ap.add_argument("--final", default=str(DEFAULT_FINAL), help=f"Path to final_df (CSV or Parquet). Default: {DEFAULT_FINAL}")
    ap.add_argument("--person-rows", dest="person_rows", default=str(DEFAULT_PERSON_ROWS), help=f"Path to persons_by_row CSV/Parquet (optional, builds influencer_table). Default: {DEFAULT_PERSON_ROWS}")
    ap.add_argument("--outdir", default=str(DEFAULT_OUTDIR), help=f"Output directory. Default: {DEFAULT_OUTDIR}")
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
        # persons_by_row is expected to be parquet; Parquet supported too if provided
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
        out_path = os.path.join(args.outdir, "influencer_table.parquet")
        out["influencer_table"].to_parquet(out_path, index=False)
        print(f"\nSaved influencer_table.parquet to: {os.path.abspath(out_path)}")
    else:
        print(
            "\nNo influencer_table produced. Ensure person_row_df is provided and includes "
            "{'row_index','persons'} with at least one non-empty 'persons' value.",
            file=sys.stderr,
        )

    # Minimal console summary
    print("\n=== PCA Influencer Pipeline Complete (no other parquets written) ===")
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