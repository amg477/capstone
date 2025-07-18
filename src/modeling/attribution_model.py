from ChannelAttribution import markov_model

def run_attribution_model(paths_df, path_col='path', conv_col='conv', order=1):
    """
    Run Markov attribution modeling on a DataFrame of influence paths.

    Args:
        paths_df (pd.DataFrame): DataFrame with 'path' and 'conv' columns.
        path_col (str): Column name for path sequences (default: 'path').
        conv_col (str): Column name for conversion indicator (default: 'conv').
        order (int): Markov model order (1 = first-order, recommended).

    Returns:
        pd.DataFrame: Attribution results with columns:
            ['channel', 'total_conversions', 'removal_effect', 'attribution']
    """
    # Validate required columns
    if path_col not in paths_df.columns or conv_col not in paths_df.columns:
        raise ValueError(f"Input DataFrame must contain '{path_col}' and '{conv_col}' columns.")
    
    # Run the model
    result = markov_model(
        paths_df,
        var_path=path_col,
        var_conv=conv_col,
        order=order
    )
    
    return result