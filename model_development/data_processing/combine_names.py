#!/usr/bin/env python3
"""
Script to combine similar names in persons_detected.csv
Combines names where one contains another (e.g., "trump" and "donald trump")
"""

import pandas as pd
import re
from collections import defaultdict

def normalize_name(name):
    """Normalize name for comparison by converting to lowercase and removing extra spaces"""
    if pd.isna(name) or name is None:
        return ""
    return re.sub(r'\s+', ' ', str(name).lower().strip())

def should_combine_names(name1, name2):
    """
    Determine if two names should be combined.
    Returns True if one name contains the other (but they're not identical).
    """
    norm1 = normalize_name(name1)
    norm2 = normalize_name(name2)
    
    if norm1 == norm2:
        return False
    
    # Check if one contains the other
    return norm1 in norm2 or norm2 in norm1

def combine_similar_names(df):
    """
    Combine similar names in the dataframe.
    When names are similar, combine their counts and keep the longer/more specific name.
    """
    # Filter out null/empty names
    df = df.dropna(subset=['person'])
    df = df[df['person'].str.strip() != '']
    
    # Create a copy to work with
    result_df = df.copy()
    
    # Group names by similarity
    name_groups = defaultdict(list)
    processed_names = set()
    
    for idx, row in df.iterrows():
        name = row['person']
        if name in processed_names:
            continue
            
        # Find all names that should be combined with this one
        similar_names = [name]
        for idx2, row2 in df.iterrows():
            name2 = row2['person']
            if name2 != name and name2 not in processed_names:
                if should_combine_names(name, name2):
                    similar_names.append(name2)
                    processed_names.add(name2)
        
        # Add all similar names to the group
        for similar_name in similar_names:
            name_groups[name].append(similar_name)
            processed_names.add(similar_name)
    
    # Create new dataframe with combined names
    combined_data = []
    
    for base_name, similar_names in name_groups.items():
        if not similar_names:
            continue
            
        # Calculate total count for all similar names
        total_count = 0
        for name in similar_names:
            count = df[df['person'] == name]['count'].iloc[0]
            total_count += count
        
        # Choose the longest/most specific name as the representative
        representative_name = max(similar_names, key=len)
        
        combined_data.append({
            'person': representative_name,
            'count': total_count
        })
    
    # Create new dataframe
    new_df = pd.DataFrame(combined_data)
    
    # Sort by count descending
    new_df = new_df.sort_values('count', ascending=False).reset_index(drop=True)
    
    return new_df

def main():
    """Main function to process the CSV file"""
    input_file = '/Users/annaglass/capstone/capstone/data/processed/persons_detected.csv'
    output_file = '/Users/annaglass/capstone/capstone/data/final/final_persons_detected.csv'
    
    print("Loading data...")
    df = pd.read_csv(input_file)
    print(f"Original data: {len(df)} rows")
    
    print("Combining similar names...")
    combined_df = combine_similar_names(df)
    print(f"Combined data: {len(combined_df)} rows")
    
    print("Saving results...")
    combined_df.to_csv(output_file, index=False)
    
    print(f"Results saved to: {output_file}")
    
    # Show some examples of the changes
    print("\nTop 20 names after combining:")
    print(combined_df.head(20))
    
    # Show reduction in rows
    reduction = len(df) - len(combined_df)
    print(f"\nReduced from {len(df)} to {len(combined_df)} rows ({reduction} names combined)")

if __name__ == "__main__":
    main()
