"""
Chart Creation Functions
Handles all Plotly chart generation for visualizations
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re


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
    # This ensures we work with all full names, not just filtered subset
    filtered_df = df.copy()
    
    # Filter to only full names (first and last name, not single names)
    filtered_df = filtered_df[filtered_df['person_list'].apply(is_full_name)]
    
    # Normalize Robert Kennedy variations and Trump variations and other name variations
    filtered_df = filtered_df.copy()
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
        final_df_copy = final_df.copy()
        if 'row_index' not in final_df_copy.columns:
            final_df_copy = final_df_copy.reset_index().rename(columns={'index': 'row_index'})
        
        # Get articles for each top person and aggregate emotions
        emotion_data = []
        
        # Pre-process persons_by_row_df to split by comma
        persons_by_row_df_processed = persons_by_row_df.copy()
        persons_by_row_df_processed['persons_list'] = persons_by_row_df_processed['persons'].astype(str).str.split(',')
        
        for person in top_persons:
            person_normalized = normalize_robert_kennedy(person)
            person_normalized = normalize_trump(person_normalized)
            person_normalized = normalize_elon_musk(person_normalized)
            person_normalized = normalize_anthony_fauci(person_normalized)
            person_normalized = normalize_kamala_harris(person_normalized)
            person_normalized_lower = person_normalized.lower().strip()
            
            # Find rows where this SPECIFIC person is mentioned (exact match in comma-separated list)
            def person_in_row(row):
                # Handle list or scalar values
                if not isinstance(row, list):
                    if pd.isna(row):
                        return False
                    row = [row]  # Convert scalar to list
                
                persons_list = row if isinstance(row, list) else []
                for p in persons_list:
                    if p is None or (not isinstance(p, str) and pd.isna(p)):
                        continue
                    p_str = str(p).strip()
                    if not p_str:
                        continue
                    p_normalized = normalize_robert_kennedy(p_str)
                    p_normalized = normalize_trump(p_normalized)
                    p_normalized = normalize_elon_musk(p_normalized)
                    p_normalized = normalize_anthony_fauci(p_normalized)
                    p_normalized = normalize_kamala_harris(p_normalized)
                    p_normalized_lower = p_normalized.lower().strip()
                    
                    # For Robert Kennedy, match variations
                    if person_normalized == "Robert Kennedy":
                        if 'robert' in p_normalized_lower and ('kennedy' in p_normalized_lower or 'junior' in p_normalized_lower):
                            return True
                    # For Donald Trump, match variations
                    elif person_normalized == "Donald Trump":
                        if 'trump' in p_normalized_lower:
                            family_members = ['lady trump', 'eric trump', 'melania trump', 'meliana trump', 'barron trump']
                            if not any(family in p_normalized_lower for family in family_members):
                                return True
                    # For Elon Musk, match variations
                    elif person_normalized == "Elon Musk":
                        if 'musk' in p_normalized_lower:
                            return True
                    # For Anthony Fauci, match variations
                    elif person_normalized == "Anthony Fauci":
                        if 'fauci' in p_normalized_lower or 'faucci' in p_normalized_lower:
                            return True
                    # For Kamala Harris, match variations
                    elif person_normalized == "Kamala Harris":
                        if 'harris' in p_normalized_lower and ('kamala' in p_normalized_lower or p_normalized_lower == 'harris'):
                            return True
                    else:
                        # Exact match (case-insensitive)
                        if p_normalized_lower == person_normalized_lower:
                            return True
                return False
            
            person_rows = persons_by_row_df_processed[
                persons_by_row_df_processed['persons_list'].apply(person_in_row)
            ]
            
            if not person_rows.empty:
                # Get articles for this person
                person_articles = final_df_copy.merge(
                    person_rows[['row_index']],
                    on='row_index',
                    how='inner'
                )
                
                # Count emotions if available
                if 'emotion_body' in person_articles.columns:
                    # Filter out None/NaN emotions
                    valid_emotions = person_articles['emotion_body'].dropna()
                    emotion_counts = valid_emotions.value_counts().to_dict()
                else:
                    # If no emotion column, skip this person
                    emotion_counts = {}
                
                # Only add if we have emotion data
                if emotion_counts:
                    emotion_data.append({
                        'person': person,
                        'emotions': emotion_counts
                    })
        
        if not emotion_data:
            return None
        
        # Prepare data for stacked bar chart
        # Get all unique emotions across all persons
        all_emotions = set()
        for item in emotion_data:
            all_emotions.update(item['emotions'].keys())
        all_emotions = sorted([e for e in all_emotions if e])  # Remove None/empty
        
        if not all_emotions:
            return None
        
        # Create DataFrame for plotting
        chart_data = []
        for item in emotion_data:
            person = item['person']
            emotions = item['emotions']
            total_count = sum(emotions.values())
            for emotion in all_emotions:
                count = emotions.get(emotion, 0)
                chart_data.append({
                    'person': person,
                    'emotion': emotion,
                    'count': count,
                    'total': total_count
                })
        
        chart_df = pd.DataFrame(chart_data)
        
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
        # For horizontal bar charts, first item in array appears at top
        person_totals = chart_df.groupby('person')['total'].first().sort_values(ascending=True)
        person_order = person_totals.index.tolist()
        
        fig.update_layout(
            title=f'Emotion Distribution for Top {n} Individuals',
            yaxis_title='Individual',
            xaxis_title='Number of Articles',
            barmode='stack',
            height=max(400, len(emotion_data) * 30),
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

