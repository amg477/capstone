#!/usr/bin/env python3
"""
Split Dataset Script for Streamlit Cloud deployment
Generates optimally sized chunks for GitHub and Streamlit compatibility
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
    return len(df) * 8 * len(df.columns) / (1024 * 1024)

def load_and_filter_dataset(input_file, target_memory_mb=800):
    """Load and filter dataset to fit within memory constraints"""
    
    print(f"Loading dataset from {input_file}...")
    print("Analyzing attribution scores...")
    
    # Load dataset in chunks
    chunk_size = 50000
    filtered_chunks = []
    
    df_info = pd.read_csv(input_file, nrows=0)
    total_estimated_size = get_file_size_mb(input_file)
    
    print(f"Total file size: {total_estimated_size:.2f} MB")
    
    all_scores = []
    
    for i, chunk in enumerate(pd.read_csv(input_file, chunksize=chunk_size)):
        print(f"Processing chunk {i+1}...")
        
        # Get attribution scores
        score_cols = ['pub_credit_share', 'max_term_credit', 'vipr_score', 'hit_strength']
        available_score_cols = [col for col in score_cols if col in chunk.columns]
        
        if available_score_cols:
            # Create combined importance score
            chunk['importance_score'] = chunk[available_score_cols].fillna(0).sum(axis=1)
            all_scores.extend(chunk['importance_score'].tolist())
        else:
            # Fallback: use random sampling
            print("No attribution score columns found, using random sampling...")
            chunk = chunk.sample(frac=0.1)
            all_scores.extend([0] * len(chunk))
        
        filtered_chunks.append(chunk)
            
    # Combine and sort by importance
    print("Combining and sorting by importance...")
    df_combined = pd.concat(filtered_chunks, ignore_index=True)
    
    # Filter to top rows based on importance score
    top_rows = min(len(df_combined), int(500000))
    
    if 'importance_score' in df_combined.columns:
        df_combined = df_combined.nlargest(top_rows, 'importance_score')
    else:
        df_combined = df_combined.sample(n=top_rows, random_state=42)
    
    # Check memory usage
    estimated_memory = estimate_memory_usage(df_combined)
    print(f"Filtered dataset: {len(df_combined):,} rows, ~{estimated_memory:.2f} MB")
    
    if estimated_memory > target_memory_mb:
        print(f"Size reduction needed. Reducing to {int(target_memory_mb * len(df_combined) / estimated_memory):,} rows...")
        df_combined = df_combined.head(int(target_memory_mb * len(df_combined) / estimated_memory))
        print(f"Final filtered dataset: {len(df_combined):,} rows, ~{estimate_memory_usage(df_combined):.2f} MB")
    
    return df_combined

def split_dataset():
    """Main function to filter and split the dataset"""
    
    # File paths
    input_file = "data/processed/final_model_dataset.csv"
    output_dir = Path("data/processed/split")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Optimizing dataset for Streamlit Cloud deployment...")
    print(f"Target: Keep dataset under 800MB RAM usage")
    print(f"Splitting {input_file} into 100MB files...\n")
    
    # Step 1: Load and filter dataset
    df_filtered = load_and_filter_dataset(input_file, target_memory_mb=800)
    
    # Step 2: Split into 100MB chunks
    max_file_size_mb = 100
    file_counter = 1
    
    # Calculate rows per chunk  
    rows_per_chunk = max(5000, int(len(df_filtered) * (max_file_size_mb - 10) / estimate_memory_usage(df_filtered)))
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
            smaller_rows = int(len(chunk) * (max_file_size_mb - 5) / file_size_mb)
            for attempt in range(3):
                chunk = chunk.iloc[:smaller_rows]
                chunk.to_csv(output_file, index=False)
                final_size = get_file_size_mb(output_file)
                print(f"Attempt {attempt+1}: Trimmed to {len(chunk):,} rows, {final_size:.2f} MB")
                if final_size <= max_file_size_mb:
                    break
                smaller_rows = int(len(chunk) * 0.8)
        
        file_counter += 1
    
    print(f"\n✅ Optimization complete!")
    print(f"📁 Created {file_counter-1} files in {output_dir}")
    print(f"📊 Total rows: {len(df_filtered):,}")
    print(f"💾 Estimated RAM usage: ~{estimate_memory_usage(df_filtered):.2f} MB")
    print(f"🎯 Target achieved: Under 1GB Streamlit limit ✅")
    
    # Save metadata
    metadata = {
        'total_files': file_counter - 1,
        'filtered_rows': len(df_filtered),
        'estimated_memory_mb': estimate_memory_usage(df_filtered),
        'max_file_size_mb': max_file_size_mb,
        'optimized_for': 'streamlit_cloud_1gb_limit',
        'columns': list(df_filtered.columns),
        'timestamp': pd.Timestamp.now().isoformat()
    }
    
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"📋 Metadata saved to: {metadata_file}")

if __name__ == "__main__":
    split_dataset()