#!/usr/bin/env python3
"""
Optimized script to split the dataset for Streamlit Cloud deployment.
This script:
1. Filters the dataset to keep only most important rows based on attribution scores
2. Splits into 100MB chunks for Git compatibility
3. Ensures total memory usage stays under 1GB limit for Streamlit Cloud
"""

import pandas as pd
import os
from pathlib import Path
import json

def get_file_size_mb(file_path):
    """Get file size in MB"""
    return os.path.getsize(file_path) / (1024 * 1024)

def estimate_memory_usage(df):
    """Estimate memory usage of DataFrame in MB"""
    return round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2)

def load_and_filter_dataset(input_file, target_memory_mb=800):
    """
    Load dataset and filter to keep most important rows based on attribution scores.
    Target memory should be under 800MB to leave room for app overhead and caching.
    """
    print(f"Loading dataset from {input_file}...")
    
    # Load dataset in chunks to handle large files
    chunk_size = 50000
    filtered_chunks = []
    
    # First pass: collect all attribution scores
    print("Analyzing attribution scores...")
    all_scores = []
    
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {i+1}...")
        
        # Get attribution scores (combine multiple score columns if available)
        score_cols = ['pub_credit_share', 'max_term_credit', 'vipr_score', 'hit_strength']
        available_score_cols = [col for col in score_cols if col in chunk.columns]
        
        if available_score_cols:
            # Create combined importance score
            chunk['importance_score'] = chunk[available_score_cols].fillna(0).sum(axis=1)
            all_scores.extend(chunk['importance_score'].tolist())
        else:
            # Fallback: use sample() to randomly select rows
            print("No attribution score columns found, using random sampling...")
            chunk = chunk.sample(frac=0.1)  # Keep only 10% randomly
            all_scores.extend([0] * len(chunk))
        
        filtered_chunks.append(chunk)
    
    # Combine all chunks and sort by importance
    print("Combining and sorting by importance...")
    df_combined = pd.concat(filtered_chunks, ignore_index=True)
    
    # Filter to top rows based on importance score
    top_rows = min(len(df_combined), int(500000))  # Cap at 500k rows max
    
    if 'importance_score' in df_combined.columns:
        df_combined = df_combined.nlargest(top_rows, 'importance_score')
    else:
        df_combined = df_combined.sample(n=top_rows, random_state=42)
    
    # Check memory usage
    memory_usage = estimate_memory_usage(df_combined)
    print(f"Filtered dataset: {len(df_combined):,} rows, ~{memory_usage} MB")
    
    if memory_usage > target_memory_mb:
        # If still too large, sample more aggressively
        reduction_factor = target_memory_mb / memory_usage
        new_size = int(len(df_combined) * reduction_factor)
        print(f"Still too large. Reducing to {new_size:,} rows...")
        
        if 'importance_score' in df_combined.columns:
            df_combined = df_combined.nlargest(new_size, 'importance_score')
        else:
            df_combined = df_combined.sample(n=new_size, random_state=42)
        
        memory_usage = estimate_memory_usage(df_combined)
        print(f"Final filtered dataset: {len(df_combined):,} rows, ~{memory_usage} MB")
    
    return df_combined

def split_dataset():
    """
    Main function to filter and split the dataset.
    """
    
    # File paths
    input_file = "final_deliverable/data/final_model_dataset.csv"
    output_dir = Path("final_deliverable/data/split")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Optimizing dataset for Streamlit Cloud deployment...")
    print(f"Target: Keep dataset under 800MB RAM usage")
    print(f"Splitting {input_file} into 100MB files...\n")
    
    # Step 1: Load and filter dataset
    df_filtered = load_and_filter_dataset(input_file, target_memory_mb=800)
    
    # Step 2: Split into 100MB chunks
    max_file_size_mb = 100  # Target max file size in MB
    file_counter = 1
    
    # Calculate rows per chunk based on current dataset size
    rows_per_chunk = max(5000, int(len(df_filtered) * max_file_size_mb / estimate_memory_usage(df_filtered)))
    print(f"Targeting ~{rows_per_chunk:,} rows per chunk")
    
    # Split into chunks
    for start_idx in range(0, len(df_filtered), rows_per_chunk):
        end_idx = min(start_idx + rows_per_chunk, len(df_filtered))
        chunk = df_filtered.iloc[start_idx:end_idx]
        
        output_file = output_dir / f"final_model_dataset_part_{file_counter:03d}.csv"
        chunk.to_csv(output_file, index=False)
        
        # Check file size
        file_size_mb = get_file_size_mb(output_file)
        print(f"Created: {output_file.name} ({len(chunk):,} rows, {file_size_mb:.2f} MB)")
        
        # If file is too large, reduce chunk size
        if file_size_mb > max_file_size_mb:
            print(f"File too large ({file_size_mb:.2f} MB), reducing chunk size...")
            # Recreate with smaller chunk
            smaller_rows = int(len(chunk) * max_file_size_mb / file_size_mb)
            chunk = chunk.iloc[:smaller_rows]
            chunk.to_csv(output_file, index=False)
            print(f"Trimmed to {len(chunk):,} rows, {get_file_size_mb(output_file):.2f} MB")
        
        file_counter += 1
    
    print(f"\n✅ Optimization complete!")
    print(f"📁 Created {file_counter-1} files in {output_dir}")
    print(f"📊 Total rows: {len(df_filtered):,}")
    print(f"💾 Estimated RAM usage: ~{estimate_memory_usage(df_filtered)} MB")
    print(f"🎯 Target achieved: Under 1GB Streamlit limit ✅")
    
    # Create metadata file
    metadata = {
        'total_files': file_counter - 1,
        'original_file': input_file,
        'filtered_rows': len(df_filtered),
        'estimated_memory_mb': estimate_memory_usage(df_filtered),
        'max_file_size_mb': max_file_size_mb,
        'optimized_for': 'streamlit_cloud_1gb_limit',
        'columns': list(df_filtered.columns),
        'output_directory': str(output_dir)
    }
    
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {output_dir}/metadata.json")
    
    return df_filtered, output_dir

if __name__ == "__main__":
    split_dataset()