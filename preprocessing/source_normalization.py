def normalize_source(source):
    """
    Normalizes different source name variations to a common label.
    
    Args:
        source (str): The original source name (e.g., Twitter handle, outlet name).
    
    Returns:
        str: Normalized source name.
    """
    if not isinstance(source, str):
        return ""

    # Lowercase for consistent mapping
    source = source.lower().strip()

    # Define mappings
    mapping = {
        "nyt": "New York Times",
        "@nytimes": "New York Times",
        "nytimes": "New York Times",
        "kffhealthnews": "KFF",
        "@kff": "KFF",
        "kff": "KFF",
        "politico": "Politico",
        "@politico": "Politico",
        "@cnn": "CNN",
        "cnn": "CNN",
        "@foxnews": "Fox News",
        "foxnews": "Fox News",
        "Twitter": "twitter",
        "X":" twitter"
        # Add more mappings as needed
    }

    return mapping.get(source, source.title())