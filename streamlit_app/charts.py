"""
Chart Creation Functions
Handles all Plotly chart generation for visualizations
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import streamlit as st


def create_emotion_chart(df, final_df=None, persons_by_row_df=None, n=20, cluster_col=None, selected_clusters=None, selected_persons=None):
    """
    Create a horizontal stacked bar chart showing emotions for the top N individuals.
    Each bar represents one individual, stacked segments show emotion distribution.
    Uses pre-computed 'emotion_body' column from final_df.
    Only shows full names (first and last name), and combines Robert Kennedy variations.
    
    Args:
        df: Influencer dataframe (person-level aggregations) - already filtered by selected_persons if applicable
        final_df: Article-level dataframe (should have 'emotion_body' column)
        persons_by_row_df: Mapping of row_index to persons (optional)
        n: Number of top individuals to show (default 20) - only used if selected_persons is None/empty
        cluster_col: Column name for cluster filtering (optional)
        selected_clusters: List of clusters to filter by (optional)
        selected_persons: List of specifically selected persons to show (if provided, only these will be shown)
    """
    if df is None or df.empty:
        return None
    
    def is_full_name(name_str):
        """Check if name has at least first and last name (2+ words)"""
        if pd.isna(name_str):
            return False
        name_str = str(name_str).strip()
        words = name_str.split()
        return len(words) >= 2
    
    def normalize_robert_kennedy(name_str):
        """Normalize Robert Kennedy variations to single name"""
        if pd.isna(name_str):
            return name_str
        name_lower = str(name_str).lower().strip()
        # Check for Robert Kennedy variations
        if 'robert' in name_lower and ('kennedy' in name_lower or 'junior' in name_lower):
            return "Robert Kennedy"
        return name_str
    
    def normalize_trump(name_str):
        """Normalize Trump variations to 'Donald Trump', excluding family members"""
        if pd.isna(name_str):
            return name_str
        name_str_clean = str(name_str).strip()
        if not name_str_clean:
            return name_str
        name_lower = name_str_clean.lower()
        # Exclude family members (check for exact matches or names containing these)
        family_members = ['lady trump', 'eric trump', 'melania trump', 'meliana trump', 'barron trump', 'ivanka trump', 'tiffany trump']
        for family in family_members:
            if family in name_lower:
                return name_str  # Keep family members as-is
        
        # Check if it's a Trump variation - ANY name containing "trump" (except family members)
        if 'trump' in name_lower:
            return "Donald Trump"
        return name_str
    
    def normalize_elon_musk(name_str):
        """Normalize Elon Musk variations to 'Elon Musk'"""
        if pd.isna(name_str):
            return name_str
        name_str_clean = str(name_str).strip()
        if not name_str_clean:
            return name_str
        name_lower = name_str_clean.lower()
        if 'elon' in name_lower and 'musk' in name_lower:
            return "Elon Musk"
        elif 'musk' in name_lower and 'elon' not in name_lower:
            if name_lower.startswith('musk') or 'musk' in name_lower:
                return "Elon Musk"
        return name_str
    
    def normalize_anthony_fauci(name_str):
        """Normalize Anthony Fauci variations to 'Anthony Fauci'"""
        if pd.isna(name_str):
            return name_str
        name_str_clean = str(name_str).strip()
        if not name_str_clean:
            return name_str
        name_lower = name_str_clean.lower()
        name_clean = re.sub(r'^(dr\.?|doctor)\s*', '', name_lower)
        name_clean = name_clean.rstrip('.,;:')
        if 'anthony' in name_clean and 'fauci' in name_clean:
            return "Anthony Fauci"
        elif 'fauci' in name_clean or 'faucci' in name_clean:
            return "Anthony Fauci"
        return name_str
    
    def normalize_kamala_harris(name_str):
        """Normalize Harris variations to 'Kamala Harris'"""
        if pd.isna(name_str):
            return name_str
        name_str_clean = str(name_str).strip()
        if not name_str_clean:
            return name_str
        name_lower = name_str_clean.lower()
        if 'kamala' in name_lower and 'harris' in name_lower:
            return "Kamala Harris"
        elif 'harris' in name_lower and 'kamala' not in name_lower:
            words = name_lower.split()
            if len(words) == 1:
                return "Kamala Harris"
            elif len(words) == 2 and words[0] == 'harris':
                different_persons = ['harris eyre', 'harris goes', 'harris poll', 'harris time', 'harris walt', 'harris darby']
                if name_lower not in different_persons:
                    return "Kamala Harris"
        return name_str
    
    # IMPORTANT: Filter to full names FIRST, before any other filtering
    # Filter to only full names (first and last name, not single names)
    mask = df['person_list'].apply(is_full_name)
    filtered_df = df[mask].copy()
    
    # Normalize names - vectorized where possible
    filtered_df['person_list'] = filtered_df['person_list'].apply(normalize_robert_kennedy)
    filtered_df['person_list'] = filtered_df['person_list'].apply(normalize_trump)
    filtered_df['person_list'] = filtered_df['person_list'].apply(normalize_elon_musk)
    filtered_df['person_list'] = filtered_df['person_list'].apply(normalize_anthony_fauci)
    filtered_df['person_list'] = filtered_df['person_list'].apply(normalize_kamala_harris)
    
    # Aggregate by normalized name (in case we have duplicates after normalization)
    # This should happen BEFORE other filters to get accurate counts
    if 'mention_count' in filtered_df.columns:
        agg_dict = {'mention_count': 'sum'}
        if cluster_col and cluster_col in filtered_df.columns:
            agg_dict[cluster_col] = 'first'
        if 'sentiment_score' in filtered_df.columns:
            agg_dict['sentiment_score'] = 'mean'
        if 'circulation_size' in filtered_df.columns:
            agg_dict['circulation_size'] = 'mean'
        
        filtered_df = filtered_df.groupby('person_list').agg(agg_dict).reset_index()
    
    # Apply cluster filter AFTER aggregation (if provided)
    if cluster_col and selected_clusters and cluster_col in filtered_df.columns:
        filtered_df = filtered_df[filtered_df[cluster_col].isin(selected_clusters)]
    
    # If specific persons are selected, show only those (respect user's filter)
    # Otherwise, show top N by mention_count
    if selected_persons and len(selected_persons) > 0:
        # User has selected specific persons - show only those
        # Normalize selected persons for matching (handle Robert Kennedy and Trump)
        selected_normalized = set()
        for selected in selected_persons:
            # Normalize the selected name
            selected_norm = normalize_robert_kennedy(selected)
            selected_norm = normalize_trump(selected_norm)
            selected_norm = normalize_elon_musk(selected_norm)
            selected_norm = normalize_anthony_fauci(selected_norm)
            selected_norm = normalize_kamala_harris(selected_norm)
            selected_normalized.add(selected_norm.lower().strip())
            # Also add the original in case it matches
            selected_normalized.add(selected.lower().strip())
        
        # Filter to only include persons that match the selected names
        def matches_selected_person(person_name):
            if pd.isna(person_name):
                return False
            person_norm = normalize_robert_kennedy(person_name)
            person_norm = normalize_trump(person_norm)
            person_norm = normalize_elon_musk(person_norm)
            person_norm = normalize_anthony_fauci(person_norm)
            person_norm = normalize_kamala_harris(person_norm)
            person_lower = str(person_norm).lower().strip()
            # Check if this person matches any selected person
            for sel_norm in selected_normalized:
                if sel_norm in person_lower or person_lower in sel_norm:
                    return True
            return False
        
        filtered_df = filtered_df[filtered_df['person_list'].apply(matches_selected_person)]
        top_persons = filtered_df['person_list'].tolist()
    else:
        # No specific selection - show top N by mention_count
        top_n_df = filtered_df.nlargest(n, 'mention_count')
        top_persons = top_n_df['person_list'].tolist()
    
    # If we have article data and person mapping, aggregate emotions per person
    if final_df is not None and persons_by_row_df is not None and not persons_by_row_df.empty:
        # Ensure final_df has row_index
        if 'row_index' not in final_df.columns:
            final_df = final_df.reset_index().rename(columns={'index': 'row_index'})
        
        # Pre-explode persons_by_row_df once - much faster than doing it per person
        pbr_exploded = persons_by_row_df[['row_index', 'persons']].dropna()
        pbr_exploded = pbr_exploded.assign(person=pbr_exploded['persons'].astype(str).str.split(',')).explode('person')
        pbr_exploded['person'] = pbr_exploded['person'].str.strip()
        pbr_exploded = pbr_exploded[pbr_exploded['person'] != '']
        
        # Pre-normalize all person names once
        pbr_exploded['person_norm'] = pbr_exploded['person'].apply(normalize_robert_kennedy)
        pbr_exploded['person_norm'] = pbr_exploded['person_norm'].apply(normalize_trump)
        pbr_exploded['person_norm'] = pbr_exploded['person_norm'].apply(normalize_elon_musk)
        pbr_exploded['person_norm'] = pbr_exploded['person_norm'].apply(normalize_anthony_fauci)
        pbr_exploded['person_norm'] = pbr_exploded['person_norm'].apply(normalize_kamala_harris)
        
        # Normalize top_persons for matching
        top_persons_normalized = {}
        for person in top_persons:
            p_norm = normalize_robert_kennedy(person)
            p_norm = normalize_trump(p_norm)
            p_norm = normalize_elon_musk(p_norm)
            p_norm = normalize_anthony_fauci(p_norm)
            p_norm = normalize_kamala_harris(p_norm)
            top_persons_normalized[person] = p_norm
        
        # Filter exploded dataframe to only top persons - vectorized!
        pbr_filtered = pbr_exploded[pbr_exploded['person_norm'].isin(top_persons_normalized.values())]
        
        if pbr_filtered.empty:
            return None
        
        # Merge with final_df once - much faster than per-person merges
        # Only select needed columns from final_df
        final_df_cols = ['row_index', 'emotion_body']
        if 'emotion_body' not in final_df.columns:
            return None
        merged = pbr_filtered[['row_index', 'person_norm']].merge(
            final_df[final_df_cols],
            on='row_index',
            how='inner'
        )
        
        # Map normalized names back to display names
        norm_to_display = {v: k for k, v in top_persons_normalized.items()}
        merged['person_display'] = merged['person_norm'].map(norm_to_display)
        merged = merged[merged['person_display'].notna()]
        
        if merged.empty or 'emotion_body' not in merged.columns:
            return None
        
        # Group by person and emotion - vectorized aggregation!
        emotion_counts = merged.groupby(['person_display', 'emotion_body']).size().reset_index(name='count')
        emotion_counts = emotion_counts[emotion_counts['emotion_body'].notna()]
        
        if emotion_counts.empty:
            return None
        
        # Get total counts per person for sorting
        person_totals = emotion_counts.groupby('person_display')['count'].sum().reset_index(name='total')
        emotion_counts = emotion_counts.merge(person_totals, on='person_display')
        
        # Filter to only top persons (in case normalization created duplicates)
        emotion_counts = emotion_counts[emotion_counts['person_display'].isin(top_persons)]
        
        if emotion_counts.empty:
            return None
        
        # Get all unique emotions
        all_emotions = sorted(emotion_counts['emotion_body'].unique())
        
        # Pivot to wide format for plotting
        chart_df = emotion_counts.pivot_table(
            index='person_display',
            columns='emotion_body',
            values='count',
            fill_value=0
        ).reset_index()
        
        # Get totals for sorting
        chart_df['total'] = chart_df[all_emotions].sum(axis=1)
        
        # Melt back to long format for plotly
        chart_df = chart_df.melt(
            id_vars=['person_display', 'total'],
            value_vars=all_emotions,
            var_name='emotion',
            value_name='count'
        )
        
        chart_df = chart_df.rename(columns={'person_display': 'person'})
        
        # Create horizontal stacked bar chart
        fig = go.Figure()
        
        # Add a trace for each emotion
        # Use brand colors from Penta (defined in app.py, but we'll use a similar palette)
        emotion_colors = px.colors.qualitative.Set3[:len(all_emotions)]
        if len(all_emotions) > len(emotion_colors):
            # Extend color palette if needed
            emotion_colors.extend(px.colors.qualitative.Pastel[:len(all_emotions) - len(emotion_colors)])
        
        for i, emotion in enumerate(all_emotions):
            emotion_df = chart_df[chart_df['emotion'] == emotion]
            fig.add_trace(go.Bar(
                name=emotion.title() if emotion else 'Unknown',
                y=emotion_df['person'],
                x=emotion_df['count'],
                orientation='h',
                marker_color=emotion_colors[i % len(emotion_colors)],
                hovertemplate='<b>%{fullData.name}</b><br>%{y}<br>Count: %{x}<extra></extra>'
            ))
        
        # Sort persons by total count (descending - most mentioned at top)
        person_order = chart_df.groupby('person')['total'].first().sort_values(ascending=True).index.tolist()
        num_persons = len(person_order)
        
        fig.update_layout(
            title=f'Emotion Distribution for Top {n} Individuals',
            yaxis_title='Individual',
            xaxis_title='Number of Articles',
            barmode='stack',
            height=max(400, num_persons * 30),
            yaxis={
                'categoryorder': 'array',
                'categoryarray': person_order  # Most mentioned first (at top)
            },
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02
            ),
            hovermode='closest'
        )
        
        return fig
    
    # If emotion data is already in the influencer table, use it directly
    # (This would require pre-aggregated emotion data)
    return None

