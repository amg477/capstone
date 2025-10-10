#!/usr/bin/env python3
"""
Script to create a subset of the final model dataset based on lowest attribution ability,
targeting approximately 1GB file size.
"""

import pandas as pd
import os
import sys

def create_subset_dataset():
    """
    Create a subset of the final model dataset based on lowest attribution ability.
    Target size: approximately 1GB
    """
    
    # File paths
    input_file = "data/files/processed/final_model_dataset.csv"
    output_file = "data/final_model_dataset_1gb.csv"
    
    print(f"Reading dataset from: {input_file}")
    
    # Get file size info
    original_size = os.path.getsize(input_file) / (1024**3)  # Size in GB
    print(f"Original file size: {original_size:.2f} GB")
    
    # Calculate target number of rows (roughly 1GB)
    target_size_gb = 1.0
    target_rows = int((target_size_gb / original_size) * 379226)  # Approximate based on current row count
    print(f"Target rows for ~1GB: {target_rows:,}")
    
    # Read the dataset in chunks to handle large file
    print("Reading dataset...")
    chunk_size = 10000
    chunks = []
    
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        chunks.append(chunk)
        if len(chunks) * chunk_size >= target_rows * 1.2:  # Read a bit more to ensure we have enough data
            break
    
    # Combine chunks
    df = pd.concat(chunks, ignore_index=True)
    print(f"Loaded {len(df):,} rows")
    
    # Identify attribution ability columns
    attribution_cols = ['pub_credit_share', 'max_term_credit']
    available_attribution_cols = [col for col in attribution_cols if col in df.columns]
    
    if not available_attribution_cols:
        print("No attribution ability columns found. Available columns:")
        print(df.columns.tolist())
        return
    
    print(f"Using attribution columns: {available_attribution_cols}")
    
    # Create a combined attribution score (lower values = lower attribution ability)
    # We'll use the minimum of available attribution columns
    df['attribution_score'] = df[available_attribution_cols].min(axis=1)
    
    # Sort by attribution score (ascending - lowest attribution ability first)
    df_sorted = df.sort_values('attribution_score', ascending=True)
    
    # Take the first target_rows (lowest attribution ability)
    df_subset = df_sorted.head(target_rows)
    
    # Remove the temporary attribution_score column
    df_subset = df_subset.drop('attribution_score', axis=1)
    
    print(f"Selected {len(df_subset):,} rows with lowest attribution ability")
    
    # Save the subset
    print(f"Saving subset to: {output_file}")
    df_subset.to_csv(output_file, index=False)
    
    # Check final file size
    final_size = os.path.getsize(output_file) / (1024**3)
    print(f"Final file size: {final_size:.2f} GB")
    
    # Show some statistics
    print("\nAttribution ability statistics for subset:")
    for col in available_attribution_cols:
        print(f"{col}: min={df_subset[col].min():.4f}, max={df_subset[col].max():.4f}, mean={df_subset[col].mean():.4f}")
    
    print(f"\nSubset created successfully: {output_file}")
    print(f"Original dataset: {len(df):,} rows, {original_size:.2f} GB")
    print(f"Subset dataset: {len(df_subset):,} rows, {final_size:.2f} GB")
    print(f"Reduction: {((len(df) - len(df_subset)) / len(df) * 100):.1f}% fewer rows")

if __name__ == "__main__":
    create_subset_dataset()
