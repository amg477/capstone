#!/usr/bin/env python3
"""
clean_influencer_table.py

EXECUTION ORDER: Step 5 of 5 in the data processing pipeline.

This script cleans the influencer_table.parquet file by:
  - Removing rows where person_list contains single-word names or initials
  - Keeping rows with full names (first + last) or known single-name celebrities/public figures
  - Maintaining a whitelist of exception names (e.g., "Trump", "Biden", "Harris")

INPUT FILES:
  - data_storage/final_data/influencer_table.parquet (created by pca.py)

OUTPUT FILES:
  - data_storage/streamlit_app_data/influencer_table_cleaned.parquet

USAGE:
    python clean_influencer_table.py

NOTES:
  - Filters entire rows based on person_list column
  - Removes rows where person_list is a single name/initial NOT in exception list
  - Output goes to streamlit_app_data folder for use by Streamlit application

PIPELINE COMPLETE:
  After this script completes, all cleaned data files are ready in streamlit_app_data/
  for use by the Streamlit application.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Set

import pandas as pd

# =============================================================================
# Paths
# =============================================================================
ROOT = Path(__file__).resolve().parents[2]  # .../capstone/capstone
INPUT_FILE = ROOT / "data_storage" / "final_data" / "influencer_table.parquet"
OUTPUT_FILE = ROOT / "data_storage" / "streamlit_app_data" / "influencer_table_cleaned.parquet"

# =============================================================================
# Exception List - Single names/initials to KEEP (case-insensitive)
# =============================================================================
EXCEPTION_NAMES: Set[str] = {
    "adele",
    "akufo-addo",
    "altman",
    "beck",
    "beyonce",
    "brady",
    "cardi",
    "carlson",
    "carter",
    "cheney",
    "cher",
    "cook",
    "cruise",
    "cruz",
    "downey",
    "ellen",
    "eminem",
    "fauci",
    "gaga",
    "gates",
    "giuliani",
    "gosling",
    "grande",
    "hanks",
    "harry",
    "harris",
    "ingraham",
    "jobs",
    "jordan",
    "kesha",
    "kimmel",
    "kobe",
    "lizzo",
    "macron",
    "maher",
    "mahomes",
    "mandaviya",
    "maddow",
    "merkel",
    "messi",
    "modi",
    "musk",
    "murdoch",
    "nadal",
    "noah",
    "obama",
    "oprah",
    "pelosi",
    "pitt",
    "prince",
    "reeves",
    "rihanna",
    "robbie",
    "roosevelt",
    "shakira",
    "shapiro",
    "sisi",
    "snoop",
    "stewart",
    "tedros",
    "tucker",
    "warren",
    "wilson",
    "zelensky",
    "zendaya",
    "zuckerburg",
    "zhou", 
    "trump",
    "biden",
    "harris",
    "kennedy",
    "obama",
    "pelosi",
    "schumer",
    "washington",
    "lincoln",
    "jefferson",
    "waltz", 
    "vance"
}


def is_single_name_or_initial(name: str) -> bool:
    """
    Check if a name is a single word or initial.
    
    Returns True for:
    - Single word names (e.g., "Anna", "Biden")
    - Single initials (e.g., "A.", "J")
    - Names starting with an initial (e.g., "J. Biden")
    - Names ending with an initial if only 2 parts (e.g., "Biden J.")
    
    Returns False for:
    - Full names with middle initials (e.g., "Robert F Kennedy")
    - Full names in general (e.g., "Joe Biden", "Kamala Harris")
    """
    if not name or pd.isna(name):
        return False
    
    name = str(name).strip()
    if not name:
        return False
    
    # Split by space to count words
    parts = [p.strip() for p in name.split() if p.strip()]
    
    # Rule 1: single-word name (no spaces)
    if len(parts) == 1:
        part = parts[0]
        # Single letter (with or without period) - e.g., "A", "A."
        if re.fullmatch(r"[A-Za-z]\.?$", part):
            return True
        # Normal one-word name (like "Biden", "Anna")
        return True
    
    # Rule 2: name STARTS with an initial (like 'J. Biden', 'A. Smith')
    if len(parts) >= 2:
        first_part = parts[0]
        if re.fullmatch(r"[A-Za-z]\.?$", first_part):
            return True
    
    # Rule 3: name ENDS with just an initial (like "Biden J.") - 2 parts only
    if len(parts) == 2:
        last_part = parts[-1]
        if re.fullmatch(r"[A-Za-z]\.?$", last_part):
            return True
    
    return False


def should_keep_name(name: str) -> bool:
    """
    Determine if a name should be kept.
    
    Returns True if:
    - Name is NOT a single name/initial, OR
    - Name IS a single name/initial BUT is in the exception list
    """
    if not name or pd.isna(name):
        return True  # Keep empty/NaN names
    
    name_str = str(name).strip()
    if not name_str:
        return True
    
    # Check if it's a single name/initial
    if not is_single_name_or_initial(name_str):
        return True  # Keep full names
    
    # It's a single name/initial - check if it's in exception list
    name_lower = name_str.lower()
    # Remove period if present for comparison
    name_lower_clean = name_lower.rstrip('.')
    
    return name_lower_clean in EXCEPTION_NAMES or name_lower in EXCEPTION_NAMES


def clean_influencer_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows from influencer_table where person_list contains
    single names/initials that are NOT in the exception list.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input influencer table
        
    Returns
    -------
    pd.DataFrame
        Cleaned influencer table
    """
    if 'person_list' not in df.columns:
        raise ValueError("Column 'person_list' not found in influencer_table")
    
    original_count = len(df)
    
    # Filter rows: keep if person_list is NOT a single name/initial,
    # OR if it IS a single name/initial but in exception list
    mask = df['person_list'].apply(should_keep_name)
    cleaned_df = df[mask].copy()
    
    removed_count = original_count - len(cleaned_df)
    
    print(f"[clean] Original rows: {original_count:,}")
    print(f"[clean] Removed rows: {removed_count:,}")
    print(f"[clean] Remaining rows: {len(cleaned_df):,}")
    print(f"[clean] Percentage kept: {len(cleaned_df)/original_count*100:.2f}%")
    
    return cleaned_df


def main():
    """Main execution function."""
    if not INPUT_FILE.exists():
        print(f"[ERROR] Input file not found: {INPUT_FILE}")
        return
    
    print(f"[load] Loading {INPUT_FILE.name}...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"[load] Loaded {len(df):,} rows × {len(df.columns)} columns")
    
    print(f"\n[clean] Cleaning influencer table...")
    print(f"[clean] Exception list contains {len(EXCEPTION_NAMES)} names")
    
    cleaned_df = clean_influencer_table(df)
    
    # Save cleaned version
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[save] Saving cleaned table to {OUTPUT_FILE.name}...")
    cleaned_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"[save] ✓ Saved {len(cleaned_df):,} rows")
    
    # Show some statistics
    print(f"\n[stats] Sample of removed names (if any):")
    removed_mask = ~df['person_list'].apply(should_keep_name)
    if removed_mask.any():
        removed_names = df[removed_mask]['person_list'].value_counts().head(20)
        print(removed_names)
    else:
        print("  No names were removed.")
    
    print(f"\n[stats] Sample of kept single names:")
    kept_single_mask = df['person_list'].apply(
        lambda x: is_single_name_or_initial(str(x)) and should_keep_name(x)
    )
    if kept_single_mask.any():
        kept_single_names = df[kept_single_mask]['person_list'].value_counts().head(20)
        print(kept_single_names)
    else:
        print("  No single names were kept.")


if __name__ == "__main__":
    main()

