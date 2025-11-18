"""
Chart Creation Functions
Handles all Plotly chart generation for visualizations
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import re
import streamlit as st


def create_person_emotion_chart(person_articles: pd.DataFrame, person_name: str, person2_articles: pd.DataFrame = None, person2_name: str = None) -> go.Figure:
    """Create a horizontal bar chart showing emotion distribution for a specific person, with optional comparison."""
    if person_articles is None or person_articles.empty or 'emotion_body' not in person_articles.columns:
        return None
    
    emotion_counts = person_articles['emotion_body'].value_counts().dropna().head(12)
    if emotion_counts.empty:
        return None
    
    # If comparing two people
    if person2_articles is not None and not person2_articles.empty and person2_name and 'emotion_body' in person2_articles.columns:
        emotion_counts2 = person2_articles['emotion_body'].value_counts().dropna().head(12)
        
        # Get all unique emotions from both
        all_emotions = set(emotion_counts.index) | set(emotion_counts2.index)
        
        # Create comparison DataFrame
        comparison_data = []
        for emotion in all_emotions:
            comparison_data.append({
                'Emotion': emotion,
                person_name: emotion_counts.get(emotion, 0),
                person2_name: emotion_counts2.get(emotion, 0)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values(person_name, ascending=True)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=person_name,
            x=comparison_df[person_name],
            y=comparison_df['Emotion'],
            orientation='h',
            marker_color='#12715D'  # Primary green
        ))
        fig.add_trace(go.Bar(
            name=person2_name,
            x=comparison_df[person2_name],
            y=comparison_df['Emotion'],
            orientation='h',
            marker_color='#4AB48E'  # Teal for comparison
        ))
        
        fig.update_layout(
            title=f'Emotion Comparison: {person_name} vs {person2_name}',
            xaxis_title='Number of Articles',
            yaxis_title='Emotion',
            height=300,
            barmode='group',
            yaxis={'categoryorder': 'total ascending'}
        )
        return fig
    
    # Single person view
    fig = px.bar(
        x=emotion_counts.values,
        y=emotion_counts.index,
        orientation='h',
        title=f'Emotions in Articles About {person_name}',
        labels={'x': 'Number of Articles', 'y': 'Emotion'},
        color_discrete_sequence=['#12715D']
    )
    fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
    return fig


def create_person_sentiment_chart(person_articles: pd.DataFrame, person_name: str, person2_articles: pd.DataFrame = None, person2_name: str = None) -> go.Figure:
    """Create a horizontal bar chart showing sentiment distribution for a specific person, with optional comparison."""
    if person_articles is None or person_articles.empty:
        return None
    
    # Ensure we have a copy to work with
    person_articles = person_articles.copy()
    
    # Normalize sentiment_band to title case if it exists
    if 'sentiment_band' in person_articles.columns:
        person_articles['sentiment_band'] = person_articles['sentiment_band'].astype(str).str.title()
        person_articles['sentiment_band'] = person_articles['sentiment_band'].replace(['Nan', 'None', 'nan', 'none'], pd.NA)
    
    if 'sentiment_band' not in person_articles.columns or person_articles['sentiment_band'].isna().all():
        if 'sentiment_score' in person_articles.columns:
            # Use appropriate thresholds for sentiment_score scale (-100 to 100)
            # Negative: < -10, Neutral: -10 to 10, Positive: > 10
            person_articles['sentiment_band'] = pd.cut(
                person_articles['sentiment_score'],
                bins=[-float('inf'), -10, 10, float('inf')],
                labels=['Negative', 'Neutral', 'Positive']
            )
        else:
            return None
    
    sentiment_counts = person_articles['sentiment_band'].value_counts().dropna()
    if sentiment_counts.empty:
        return None
    
    # If comparing two people
    if person2_articles is not None and not person2_articles.empty and person2_name:
        # Ensure we have a copy to work with
        person2_articles = person2_articles.copy()
        
        # Normalize sentiment_band to title case if it exists
        if 'sentiment_band' in person2_articles.columns:
            person2_articles['sentiment_band'] = person2_articles['sentiment_band'].astype(str).str.title()
            person2_articles['sentiment_band'] = person2_articles['sentiment_band'].replace(['Nan', 'None', 'nan', 'none'], pd.NA)
        
        if 'sentiment_band' not in person2_articles.columns or person2_articles['sentiment_band'].isna().all():
            if 'sentiment_score' in person2_articles.columns:
                person2_articles = person2_articles.copy()
                # Use appropriate thresholds for sentiment_score scale (-100 to 100)
                # Negative: < -10, Neutral: -10 to 10, Positive: > 10
                person2_articles['sentiment_band'] = pd.cut(
                    person2_articles['sentiment_score'],
                    bins=[-float('inf'), -10, 10, float('inf')],
                    labels=['Negative', 'Neutral', 'Positive']
                )
            else:
                return None
        
        sentiment_counts2 = person2_articles['sentiment_band'].value_counts().dropna()
        
        if sentiment_counts2.empty:
            return None
        
        # Get all sentiment bands
        all_sentiments = ['Negative', 'Neutral', 'Positive']
        
        # Convert to dictionaries for easier access, normalizing keys to title case
        sentiment_counts_dict = {str(k).title(): int(v) for k, v in sentiment_counts.items()}
        sentiment_counts2_dict = {str(k).title(): int(v) for k, v in sentiment_counts2.items()}
        
        # Create comparison DataFrame with safe column names
        comparison_data = []
        for sentiment in all_sentiments:
            person1_count = sentiment_counts_dict.get(sentiment, 0)
            person2_count = sentiment_counts2_dict.get(sentiment, 0)
            comparison_data.append({
                'Sentiment': sentiment,
                'Person1': person1_count,
                'Person2': person2_count
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=person_name,
            x=comparison_df['Person1'],
            y=comparison_df['Sentiment'],
            orientation='h',
            marker_color='#12715D'  # Primary green
        ))
        fig.add_trace(go.Bar(
            name=person2_name,
            x=comparison_df['Person2'],
            y=comparison_df['Sentiment'],
            orientation='h',
            marker_color='#4AB48E'  # Teal for comparison
        ))
        
        fig.update_layout(
            title=f'Sentiment Comparison: {person_name} vs {person2_name}',
            xaxis_title='Number of Articles',
            yaxis_title='Sentiment',
            height=300,
            barmode='group',
            yaxis={'categoryorder': 'array', 'categoryarray': ['Positive', 'Neutral', 'Negative']}
        )
        return fig
    
    # Single person view
    fig = px.bar(
        x=sentiment_counts.values,
        y=sentiment_counts.index,
        orientation='h',
        title=f'Sentiment in Articles About {person_name}',
        labels={'x': 'Number of Articles', 'y': 'Sentiment'},
        color_discrete_sequence=['#12715D']
    )
    fig.update_layout(height=300, yaxis={'categoryorder': 'total ascending'})
    return fig


def create_person_mentions_over_time_chart(person_articles: pd.DataFrame, person_name: str = None, person2_articles: pd.DataFrame = None, person2_name: str = None) -> go.Figure:
    """Create a line chart showing mentions over time for a person, with optional comparison."""
    if person_articles is None or person_articles.empty or 'published_datetime' not in person_articles.columns:
        return None
    
    mentions_df = person_articles[['published_datetime']].copy()
    mentions_df['published_datetime'] = pd.to_datetime(mentions_df['published_datetime'], errors='coerce')
    mentions_df = mentions_df.dropna(subset=['published_datetime'])
    
    if mentions_df.empty:
        return None
    
    # Convert to date, then back to datetime to avoid PyArrow date32 type issues with Plotly
    mentions_df['date'] = mentions_df['published_datetime'].dt.date
    mentions_series = (
        mentions_df.groupby('date')
        .size()
        .reset_index(name='mentions')
        .sort_values('date')
    )
    
    # Convert date back to datetime for Plotly compatibility
    mentions_series['date'] = pd.to_datetime(mentions_series['date'])
    
    # If comparing two people
    if person2_articles is not None and not person2_articles.empty and person2_name and 'published_datetime' in person2_articles.columns:
        mentions_df2 = person2_articles[['published_datetime']].copy()
        mentions_df2['published_datetime'] = pd.to_datetime(mentions_df2['published_datetime'], errors='coerce')
        mentions_df2 = mentions_df2.dropna(subset=['published_datetime'])
        
        if not mentions_df2.empty:
            mentions_df2['date'] = mentions_df2['published_datetime'].dt.date
            mentions_series2 = (
                mentions_df2.groupby('date')
                .size()
                .reset_index(name='mentions')
                .sort_values('date')
            )
            mentions_series2['date'] = pd.to_datetime(mentions_series2['date'])
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=mentions_series['date'],
                y=mentions_series['mentions'],
                mode='lines+markers',
                name=person_name or 'Person 1',
                line=dict(color='#12715D', width=2),
                marker=dict(size=4)
            ))
            fig.add_trace(go.Scatter(
                x=mentions_series2['date'],
                y=mentions_series2['mentions'],
                mode='lines+markers',
                name=person2_name,
                line=dict(color='#4AB48E', width=2),
                marker=dict(size=4)
            ))
            
            fig.update_layout(
                title=f'Mentions Over Time: {person_name or "Person 1"} vs {person2_name}',
                xaxis_title='Date',
                yaxis_title='Number of Articles',
                template='simple_white',
                margin=dict(l=10, r=10, t=40, b=10),
                height=350,
                hovermode='x unified'
            )
            return fig
    
    # Single person view
    title = f'Mentions Over Time: {person_name}' if person_name else 'Mentions Over Time'
    fig = px.line(
        mentions_series,
        x='date',
        y='mentions',
        labels={'date': 'Date', 'mentions': 'Number of Articles'},
        title=title,
        color_discrete_sequence=['#12715D']
    )
    fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=40, b=10),
        height=350,
    )
    return fig


def create_topic_sentiment_by_people_chart(
    pbr_long_topic: pd.DataFrame,
    final_df_topic: pd.DataFrame,
    top_n_people: list,
    num_people: int
) -> go.Figure:
    """Create a stacked horizontal bar chart showing sentiment distribution by people for a topic."""
    if pbr_long_topic is None or pbr_long_topic.empty or len(top_n_people) == 0:
        return None
    
    if 'sentiment_band' not in final_df_topic.columns:
        if 'sentiment_score' in final_df_topic.columns:
            final_df_topic = final_df_topic.copy()
            # Use appropriate thresholds for sentiment_score scale (-100 to 100)
            # Negative: < -10, Neutral: -10 to 10, Positive: > 10
            final_df_topic['sentiment_band'] = pd.cut(
                final_df_topic['sentiment_score'],
                bins=[-float('inf'), -10, 10, float('inf')],
                labels=['negative', 'neutral', 'positive']
            )
        else:
            return None
    
    pbr_filtered = pbr_long_topic[pbr_long_topic['person'].isin(top_n_people)].copy()
    pbr_filtered['row_index'] = pd.to_numeric(pbr_filtered['row_index'], errors='coerce')
    final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')
    
    sent_merge = pbr_filtered[['row_index', 'person']].merge(
        final_df_topic[['row_index', 'sentiment_band']],
        on='row_index',
        how='left'
    )
    sent_merge = sent_merge[sent_merge['sentiment_band'].notna()]
    
    if sent_merge.empty:
        return None
    
    sent_counts = (
        sent_merge.groupby(['person', 'sentiment_band'])
        .size()
        .reset_index(name='count')
    )
    all_bands = ['negative', 'neutral', 'positive']
    sent_pivot = sent_counts.pivot_table(
        index='person',
        columns='sentiment_band',
        values='count',
        fill_value=0
    ).reset_index()
    sent_pivot = sent_pivot[sent_pivot['person'].isin(top_n_people)]
    for band in all_bands:
        if band not in sent_pivot.columns:
            sent_pivot[band] = 0
    sent_pivot['total'] = sent_pivot[all_bands].sum(axis=1)
    sent_pivot = sent_pivot.sort_values('total', ascending=True).tail(num_people)
    sent_chart_df = sent_pivot.melt(
        id_vars=['person', 'total'],
        value_vars=all_bands,
        var_name='sentiment_band',
        value_name='count'
    )
    
    fig = go.Figure()
    band_colors = {'negative': '#D94841', 'neutral': '#D4A115', 'positive': '#4AB48E'}
    for band in all_bands:
        band_df = sent_chart_df[sent_chart_df['sentiment_band'] == band]
        fig.add_trace(go.Bar(
            name=band.title(),
            y=band_df['person'],
            x=band_df['count'],
            orientation='h',
            marker_color=band_colors.get(band, '#808080')
        ))
    person_order = sent_pivot.sort_values('total', ascending=True)['person'].tolist()
    fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(600, len(top_n_people) * 25),
        barmode='stack',
        xaxis_title='Number of Articles',
        yaxis_title='Person',
        yaxis={'categoryorder': 'array', 'categoryarray': person_order},
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        showlegend=True
    )
    return fig


def create_topic_emotion_by_people_chart(
    pbr_long_topic: pd.DataFrame,
    final_df_topic: pd.DataFrame,
    top_n_people: list,
    num_people: int
) -> go.Figure:
    """Create a stacked horizontal bar chart showing emotion distribution by people for a topic."""
    if pbr_long_topic is None or pbr_long_topic.empty or len(top_n_people) == 0:
        return None
    
    pbr_filtered_emotion = pbr_long_topic[pbr_long_topic['person'].isin(top_n_people)].copy()
    pbr_filtered_emotion['row_index'] = pd.to_numeric(pbr_filtered_emotion['row_index'], errors='coerce')
    final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')
    
    emotion_merge = pbr_filtered_emotion[['row_index', 'person']].merge(
        final_df_topic[['row_index', 'emotion_body']],
        on='row_index',
        how='left'
    )
    emotion_merge = emotion_merge[emotion_merge['emotion_body'].notna()]
    emotion_merge['emotion_body'] = emotion_merge['emotion_body'].str.capitalize()
    
    if emotion_merge.empty:
        return None
    
    emotion_counts = (
        emotion_merge.groupby(['person', 'emotion_body'])
        .size()
        .reset_index(name='count')
    )
    all_emotions = sorted(emotion_counts['emotion_body'].unique())
    emotion_pivot = emotion_counts.pivot_table(
        index='person',
        columns='emotion_body',
        values='count',
        fill_value=0
    ).reset_index()
    emotion_pivot = emotion_pivot[emotion_pivot['person'].isin(top_n_people)]
    emotion_pivot['total'] = emotion_pivot[all_emotions].sum(axis=1)
    emotion_pivot = emotion_pivot.sort_values('total', ascending=True).tail(num_people)
    emotion_chart_df = emotion_pivot.melt(
        id_vars=['person', 'total'],
        value_vars=all_emotions,
        var_name='emotion',
        value_name='count'
    )
    
    fig = go.Figure()
    emotion_colors = px.colors.qualitative.Set3[:len(all_emotions)]
    if len(all_emotions) > len(emotion_colors):
        emotion_colors.extend(
            px.colors.qualitative.Pastel[:len(all_emotions) - len(emotion_colors)]
        )
    for idx, emotion in enumerate(all_emotions):
        emotion_df = emotion_chart_df[emotion_chart_df['emotion'] == emotion]
        fig.add_trace(go.Bar(
            name=emotion.title() if emotion else 'Unknown',
            y=emotion_df['person'],
            x=emotion_df['count'],
            orientation='h',
            marker_color=emotion_colors[idx % len(emotion_colors)]
        ))
    person_order_emotion = emotion_pivot.sort_values('total', ascending=True)['person'].tolist()
    fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=max(600, len(top_n_people) * 25),
        barmode='stack',
        xaxis_title='Number of Articles',
        yaxis_title='Person',
        yaxis={'categoryorder': 'array', 'categoryarray': person_order_emotion},
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        showlegend=True
    )
    return fig


def create_topic_mentions_over_time_chart(
    pbr_long_topic: pd.DataFrame,
    final_df_topic: pd.DataFrame,
    top_n_people: list
) -> go.Figure:
    """Create a stacked area chart showing circulation over time by people for a topic."""
    if pbr_long_topic is None or pbr_long_topic.empty or len(top_n_people) == 0:
        return None
    
    pbr_filtered_time = pbr_long_topic[pbr_long_topic['person'].isin(top_n_people)].copy()
    pbr_filtered_time['row_index'] = pd.to_numeric(pbr_filtered_time['row_index'], errors='coerce')
    final_df_topic['row_index'] = pd.to_numeric(final_df_topic['row_index'], errors='coerce')
    
    time_merge = pbr_filtered_time[['row_index', 'person']].merge(
        final_df_topic[['row_index', 'published_datetime', 'circulation_size']],
        on='row_index',
        how='left'
    )
    time_merge = time_merge[time_merge['published_datetime'].notna()]
    time_merge = time_merge[time_merge['circulation_size'].notna()]
    
    if time_merge.empty:
        return None
    
    time_merge['published_datetime'] = pd.to_datetime(time_merge['published_datetime'], errors='coerce')
    time_merge = time_merge[time_merge['published_datetime'].notna()]
    time_merge['date'] = time_merge['published_datetime'].dt.date
    time_grouped = (
        time_merge.groupby(['date', 'person'])['circulation_size']
        .sum()
        .reset_index()
    )
    time_pivot = time_grouped.pivot_table(
        index='date',
        columns='person',
        values='circulation_size',
        fill_value=0
    ).sort_index()
    
    fig = go.Figure()
    colors = px.colors.qualitative.Set3[:len(top_n_people)]
    if len(top_n_people) > len(colors):
        colors.extend(px.colors.qualitative.Pastel[:len(top_n_people) - len(colors)])
    
    for idx, person in enumerate(top_n_people):
        if person in time_pivot.columns:
            fig.add_trace(go.Scatter(
                name=person,
                x=time_pivot.index,
                y=time_pivot[person],
                mode='lines',
                stackgroup='one',
                fill='tonexty' if idx > 0 else 'tozeroy',
                line=dict(width=0.6, color=colors[idx % len(colors)]),
                fillcolor=colors[idx % len(colors)]
            ))
    
    fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=500,
        xaxis_title='Date',
        yaxis_title='Circulation Size',
        hovermode='x unified',
        legend=dict(orientation="v", yanchor="top", y=1, xanchor="left", x=1.02),
        showlegend=True
    )
    return fig


def create_circulation_quartile_chart(final_df_topic: pd.DataFrame) -> go.Figure:
    """Create a bar chart showing circulation quartile distribution."""
    if final_df_topic is None or final_df_topic.empty:
        return None
    
    final_df_topic_circ = final_df_topic.copy()
    circ_data = final_df_topic_circ['circulation_size'].dropna()
    
    if len(circ_data) == 0:
        return None
    
    try:
        final_df_topic_circ['circulation_quartile'] = pd.qcut(
            circ_data,
            q=min(4, len(circ_data.unique())),
            labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'][:min(4, len(circ_data.unique()))],
            duplicates='drop'
        )
        quartile_counts = (
            final_df_topic_circ['circulation_quartile']
            .value_counts()
            .sort_index()
        )
    except (ValueError, TypeError):
        try:
            bins = pd.cut(
                circ_data,
                bins=4,
                labels=['Q1 (Lowest)', 'Q2', 'Q3', 'Q4 (Highest)'],
                duplicates='drop'
            )
            final_df_topic_circ.loc[circ_data.index, 'circulation_quartile'] = bins
            quartile_counts = (
                final_df_topic_circ['circulation_quartile']
                .value_counts()
                .sort_index()
            )
        except Exception:
            return None
    
    if quartile_counts.empty:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=quartile_counts.index.astype(str),
            y=quartile_counts.values,
            marker_color='#12715D',
            text=quartile_counts.values,
            textposition='outside'
        )
    ])
    fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=10, b=10),
        height=300,
        xaxis_title='Circulation Quartile',
        yaxis_title='Number of Articles',
        showlegend=False
    )
    return fig


def create_top_people_bar_chart(pbr_long_filtered: pd.DataFrame, n_top: int) -> go.Figure:
    """Create a horizontal bar chart showing top people by article count."""
    if pbr_long_filtered is None or pbr_long_filtered.empty:
        return None
    
    top_people_counts = pbr_long_filtered['person'].value_counts().reset_index()
    top_people_counts.columns = ['person', 'article_count']
    
    if top_people_counts.empty:
        return None
    
    top_people_counts = top_people_counts.head(n_top)
    fig = px.bar(
        top_people_counts.sort_values('article_count', ascending=True),
        x='article_count',
        y='person',
        orientation='h',
        labels={'article_count': 'Number of Articles', 'person': 'Individual'},
        title=f"Top {len(top_people_counts)} Individuals by Article Mentions"
    )
    fig.update_layout(
        template='simple_white',
        margin=dict(l=10, r=10, t=60, b=10),
        height=max(400, len(top_people_counts) * 22),
        yaxis={'categoryorder': 'array', 'categoryarray': top_people_counts.sort_values('article_count')['person']}
    )
    return fig


def create_sentiment_by_cluster_chart(influencer_view: pd.DataFrame, cluster_col: str) -> go.Figure:
    """Create a bar chart showing sentiment distribution by cluster."""
    if influencer_view is None or influencer_view.empty or cluster_col not in influencer_view.columns:
        return None
    
    sentiment_by_cluster = (
        influencer_view.groupby(cluster_col)['sentiment_score']
        .mean()
        .reset_index()
        .sort_values('sentiment_score', ascending=False)
        .head(10)
    )
    
    if sentiment_by_cluster.empty:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=sentiment_by_cluster[cluster_col], 
            y=sentiment_by_cluster['sentiment_score'],
            marker_color='#12715D'  # Dark green
        )
    ])
    fig.update_layout(
        title="Sentiment Distribution by Cluster (Top 10)",
        xaxis_title="Cluster",
        yaxis_title="Average Sentiment",
        height=300,
        showlegend=False,
    )
    return fig


def create_circulation_by_cluster_chart(influencer_view: pd.DataFrame, cluster_col: str) -> go.Figure:
    """Create a bar chart showing circulation by cluster."""
    if influencer_view is None or influencer_view.empty or cluster_col not in influencer_view.columns:
        return None
    
    circ_by_cluster = (
        influencer_view.groupby(cluster_col)['circulation_size']
        .mean()
        .reset_index()
        .sort_values('circulation_size', ascending=False)
        .head(10)
    )
    
    if circ_by_cluster.empty:
        return None
    
    fig = go.Figure(data=[
        go.Bar(
            x=circ_by_cluster[cluster_col], 
            y=circ_by_cluster['circulation_size'],
            marker_color='#12715D'  # Dark green
        )
    ])
    fig.update_layout(
        title="Top Clusters by Circulation",
        xaxis_title="Cluster",
        yaxis_title="Average Circulation",
        height=300,
        showlegend=False,
    )
    return fig

