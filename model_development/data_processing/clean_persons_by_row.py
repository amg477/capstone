#!/usr/bin/env python3
"""
Clean persons_by_row.parquet by removing single-word names and initials from
the comma-separated 'persons' column, except for a whitelist of known 
single-name celebrities/public figures.

Usage:
    python clean_persons_by_row.py
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
INPUT_FILE = ROOT / "data_storage" / "final_data" / "persons_by_row.parquet"
OUTPUT_FILE = ROOT / "data_storage" / "streamlit_app_data" / "persons_by_row_cleaned.parquet"

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
    "vance",
    "johnson"
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
    if pd.isna(name) or not name:
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
    if pd.isna(name) or not name:
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


def clean_persons_string(persons_str: str) -> str:
    """
    Clean a comma-separated string of person names by removing
    single names/initials that are NOT in the exception list.
    
    Parameters
    ----------
    persons_str : str
        Comma-separated string of person names
        
    Returns
    -------
    str
        Comma-separated string with single names/initials removed
        (except those in exception list)
    """
    if pd.isna(persons_str) or not persons_str:
        return ""
    
    persons_str = str(persons_str).strip()
    if not persons_str:
        return ""
    
    # Split by comma and clean each name
    names = [n.strip() for n in persons_str.split(",") if n.strip()]
    
    # Filter: keep only names that pass should_keep_name check
    kept_names = [name for name in names if should_keep_name(name)]
    
    # Rejoin with comma-space separator
    return ", ".join(kept_names)


def clean_persons_by_row(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean persons_by_row dataframe by removing single names/initials
    from the 'persons' column, except those in the exception list.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input persons_by_row dataframe
        
    Returns
    -------
    pd.DataFrame
        Cleaned persons_by_row dataframe
    """
    if 'persons' not in df.columns:
        raise ValueError("Column 'persons' not found in persons_by_row")
    
    original_count = len(df)
    
    # Create a copy to avoid modifying original
    cleaned_df = df.copy()
    
    # Clean the persons column
    cleaned_df['persons'] = cleaned_df['persons'].apply(clean_persons_string)
    
    # Count rows that had names removed
    original_non_empty = df['persons'].notna() & (df['persons'].astype(str).str.strip() != "")
    cleaned_non_empty = cleaned_df['persons'].notna() & (cleaned_df['persons'].astype(str).str.strip() != "")
    
    rows_with_names_removed = (original_non_empty & ~cleaned_non_empty).sum()
    rows_now_empty = cleaned_non_empty.sum()
    
    # Update has_person flag if it exists
    if 'has_person' in cleaned_df.columns:
        cleaned_df['has_person'] = (cleaned_df['persons'].notna() & 
                                     (cleaned_df['persons'].astype(str).str.strip() != "")).astype(int)
    
    print(f"[clean] Original rows: {original_count:,}")
    print(f"[clean] Rows that became empty after cleaning: {rows_with_names_removed:,}")
    print(f"[clean] Rows with at least one person remaining: {rows_now_empty:,}")
    print(f"[clean] Percentage with persons remaining: {rows_now_empty/original_count*100:.2f}%")
    
    return cleaned_df


def main():
    """Main execution function."""
    # Try multiple possible input file locations
    input_locations = [
        INPUT_FILE,
        ROOT / "data_storage" / "final_data" / "persons_by_row.parquet",
        ROOT / "data_storage" / "streamlit_app_data" / "persons_by_row.parquet",
        ROOT / "data_storage" / "streamlit_app_data" / "persons_by_row.parquet",
    ]
    
    df = None
    input_path = None
    
    for path in input_locations:
        if path.exists():
            try:
                print(f"[load] Attempting to load {path.name}...")
                if path.suffix == '.parquet':
                    df = pd.read_parquet(path)
                else:
                    df = pd.read_parquet(path)
                input_path = path
                print(f"[load] ✓ Successfully loaded from {path}")
                break
            except Exception as e:
                print(f"[load] ✗ Failed to load {path}: {e}")
                continue
    
    if df is None:
        print(f"[ERROR] Could not load persons_by_row from any of these locations:")
        for path in input_locations:
            print(f"  - {path}")
        return
    
    print(f"[load] Loaded {len(df):,} rows × {len(df.columns)} columns")
    print(f"[load] Columns: {list(df.columns)}")
    
    print(f"\n[clean] Cleaning persons_by_row...")
    print(f"[clean] Exception list contains {len(EXCEPTION_NAMES)} names")
    
    cleaned_df = clean_persons_by_row(df)
    
    # Save cleaned version
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n[save] Saving cleaned table to {OUTPUT_FILE.name}...")
    cleaned_df.to_parquet(OUTPUT_FILE, index=False)
    print(f"[save] ✓ Saved {len(cleaned_df):,} rows")
    
    # Show some statistics
    print(f"\n[stats] Sample of removed single names (top 20):")
    # Extract all names from original
    all_original_names = []
    for persons_str in df['persons'].dropna():
        names = [n.strip() for n in str(persons_str).split(",") if n.strip()]
        all_original_names.extend(names)
    
    # Extract all names from cleaned
    all_cleaned_names = []
    for persons_str in cleaned_df['persons'].dropna():
        names = [n.strip() for n in str(persons_str).split(",") if n.strip()]
        all_cleaned_names.extend(names)
    
    # Find removed names
    original_set = set(n.lower().rstrip('.') for n in all_original_names)
    cleaned_set = set(n.lower().rstrip('.') for n in all_cleaned_names)
    removed_set = original_set - cleaned_set
    
    if removed_set:
        # Count occurrences of removed names
        removed_counts = {}
        for name in all_original_names:
            name_lower = name.lower().rstrip('.')
            if name_lower in removed_set:
                removed_counts[name_lower] = removed_counts.get(name_lower, 0) + 1
        
        removed_series = pd.Series(removed_counts).sort_values(ascending=False).head(20)
        print(removed_series)
    else:
        print("  No names were removed.")
    
    print(f"\n[stats] Sample of kept single names (top 20):")
    kept_single = []
    for name in all_cleaned_names:
        if is_single_name_or_initial(name) and should_keep_name(name):
            kept_single.append(name.lower().rstrip('.'))
    
    if kept_single:
        kept_counts = pd.Series(kept_single).value_counts().head(20)
        print(kept_counts)
    else:
        print("  No single names were kept.")


if __name__ == "__main__":
    main()

