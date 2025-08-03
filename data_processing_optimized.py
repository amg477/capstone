"""
File Name: data_processing_optimized.py

Purpose:
This file cleans and processes the data to be ready for EDA and model development.
Optimized for computational efficiency with chunked processing and vectorized operations.

Addresses:
- Combines sentiment columns into 'sentiment' and 'sentiment_overflow'
- Drops columns with more than 90% missing values (at the end)
- Uses memory-efficient processing techniques
"""

import pandas as pd
import numpy as np
import os
from pathlib import Path
from typing import List, Optional
import gc

def get_excel_file_paths() -> List[str]:
    """
    Returns list of Excel file paths.
    Separated for better maintainability and potential dynamic file discovery.
    """
    return [
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_1_df8ce708-7388-44d9-9aeb-a9f1cd72fd6b.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_2_5f3db6ca-8894-45b3-8e08-162e2e88baba.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_3_dd084585-7587-445b-ace2-fdd4eec637f9.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_4_9eb1d7ad-6010-4d1d-a128-b6af3cca5cfa.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_5_53407d72-3f41-4dc2-878e-db6076e73954.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_6_7e7a5c27-9b2a-444c-bec5-92133ae41865.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_7_7e2eaa3d-a08f-4882-a8f9-3a2ad04c91c7.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_8_edeac5df-5b03-424a-9706-8ef3de3827ca.xlsx",
        "data/raw/Extract_NHS_Reform_Penrose_Enquiry_State_Healthcare_Fundin_Article_Detail_d13e454c-6fef-48ac-af60-4406f9204813.xlsx"
    ]

def load_and_filter_excel(file_path: str, cols_to_drop: List[str]) -> Optional[pd.DataFrame]:
    """
    Load a single Excel file and apply initial filtering.
    
    Args:
        file_path: Path to Excel file
        cols_to_drop: Columns to drop early to save memory
        
    Returns:
        Filtered DataFrame or None if file doesn't exist/is empty
    """
    try:
        # Check if file exists
        if not os.path.exists(file_path):
            print(f"Warning: File not found: {file_path}")
            return None
            
        # Read Excel file
        df = pd.read_excel(file_path)
        
        # Early filtering for US observations to reduce memory usage
        if 'country' in df.columns:
            df = df[df['country'] == 'United States'].copy()
        
        # Drop unnecessary columns early to save memory
        existing_cols_to_drop = [col for col in cols_to_drop if col in df.columns]
        if existing_cols_to_drop:
            df = df.drop(columns=existing_cols_to_drop)
            
        # Return None if no data remains after filtering
        if df.empty:
            return None
            
        return df
        
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

def calculate_missing_ratio_vectorized(df: pd.DataFrame) -> pd.Series:
    """
    Vectorized calculation of missing value ratios.
    More efficient than the original approach.
    """
    # Use vectorized operations for better performance
    na_ratio = df.isna().sum() / len(df)
    empty_ratio = (df == "").sum() / len(df)
    return na_ratio + empty_ratio

def data_processing_optimized(chunk_size: int = 50000) -> pd.DataFrame:
    """
    Loads, combines, and processes multiple Excel files with optimized memory usage.
    
    Args:
        chunk_size: Size of chunks for processing large datasets
        
    Returns: 
        A single, processed DataFrame combining and cleaning all data.
    """
    
    excel_file_paths = get_excel_file_paths()
    
    # Columns to drop early (defined once)
    cols_to_drop = ["hit_strength", "vipr_weight", "vipr_score", "country"]
    
    # Load and process files with early filtering
    print("Loading and filtering Excel files...")
    processed_dfs = []
    
    for i, path in enumerate(excel_file_paths):
        print(f"Processing file {i+1}/{len(excel_file_paths)}: {Path(path).name}")
        df = load_and_filter_excel(path, cols_to_drop)
        if df is not None:
            processed_dfs.append(df)
        
        # Force garbage collection periodically
        if i % 3 == 0:
            gc.collect()
    
    if not processed_dfs:
        raise ValueError("No valid data found in any of the Excel files")
    
    # Combine dataframes more efficiently
    print("Combining dataframes...")
    df = pd.concat(processed_dfs, ignore_index=True, copy=False)
    
    # Clear individual dataframes from memory
    del processed_dfs
    gc.collect()
    
    # Process missing values with vectorized operations
    print("Calculating missing value ratios...")
    threshold = 0.9
    missing_ratio = calculate_missing_ratio_vectorized(df)
    columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if columns_to_drop:
        print(f"Dropping {len(columns_to_drop)} columns with >90% missing values")
        df = df.drop(columns=columns_to_drop)
    
    # Optimize data types to reduce memory usage
    print("Optimizing data types...")
    df = optimize_dtypes(df)
    
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimize data types to reduce memory usage.
    """
    # Convert object columns to category if they have low cardinality
    for col in df.select_dtypes(include=['object']).columns:
        if df[col].nunique() / len(df) < 0.5:  # Less than 50% unique values
            df[col] = df[col].astype('category')
    
    # Optimize numeric columns
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    
    return df

def data_processing_memory_efficient() -> pd.DataFrame:
    """
    Alternative implementation using chunked processing for very large datasets.
    """
    excel_file_paths = get_excel_file_paths()
    cols_to_drop = ["hit_strength", "vipr_weight", "vipr_score", "country"]
    
    # Process files one at a time and write to temporary files if needed
    combined_chunks = []
    
    for path in excel_file_paths:
        try:
            # Read in chunks if file is very large
            for chunk in pd.read_excel(path, chunksize=10000):
                # Apply filtering
                if 'country' in chunk.columns:
                    chunk = chunk[chunk['country'] == 'United States']
                
                # Drop columns early
                existing_cols_to_drop = [col for col in cols_to_drop if col in chunk.columns]
                if existing_cols_to_drop:
                    chunk = chunk.drop(columns=existing_cols_to_drop)
                
                if not chunk.empty:
                    combined_chunks.append(chunk)
                    
        except ValueError:
            # If chunking not supported, fall back to regular read
            chunk = load_and_filter_excel(path, cols_to_drop)
            if chunk is not None:
                combined_chunks.append(chunk)
    
    # Combine all chunks
    df = pd.concat(combined_chunks, ignore_index=True)
    
    # Apply missing value filtering
    threshold = 0.9
    missing_ratio = calculate_missing_ratio_vectorized(df)
    columns_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
    
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
    
    return optimize_dtypes(df)

if __name__ == "__main__":
    print("Starting optimized data processing")
    
    # Choose processing method based on available memory
    try:
        final_df = data_processing_optimized()
    except MemoryError:
        print("Memory error encountered. Switching to memory-efficient processing...")
        final_df = data_processing_memory_efficient()
    
    # Create output directory if it doesn't exist
    output_path = "/Users/annaglass/capstone/capstone/data/processed/processed_data.csv"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save with optimization
    print("Saving processed data...")
    final_df.to_csv(output_path, index=False)
    
    print(f"Data processing complete. The new file has been saved to: {output_path}")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Memory usage: {final_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")