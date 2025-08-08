import pandas as pd

def build_paths(mentions_df, events_df, window_days=7):
    """
    Build influence paths for each event based on mentions before the event.
    
    Args:
        mentions_df (pd.DataFrame): DataFrame with 'source' and 'timestamp'.
        events_df (pd.DataFrame): DataFrame with 'event_id' and 'timestamp'.
        window_days (int): Number of days before the event to collect mentions.

    Returns:
        pd.DataFrame: A dataframe with columns: path_id, path, conv.
    """
    # Ensure datetime format
    mentions_df['timestamp'] = pd.to_datetime(mentions_df['timestamp'])
    events_df['timestamp'] = pd.to_datetime(events_df['timestamp'])

    path_rows = []

    for _, event in events_df.iterrows():
        event_id = event['event_id']
        event_time = event['timestamp']

        # Filter mentions within the time window before the event
        start_time = event_time - pd.Timedelta(days=window_days)
        mentions_window = mentions_df[
            (mentions_df['timestamp'] >= start_time) & 
            (mentions_df['timestamp'] < event_time)
        ].sort_values('timestamp')

        if mentions_window.empty:
            continue

        # Build path string
        path = " > ".join(mentions_window['source'].tolist())
        
        path_rows.append({
            'path_id': event_id,
            'path': path,
            'conv': 1  # All paths here lead to conversion
        })

    return pd.DataFrame(path_rows)