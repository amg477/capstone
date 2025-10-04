# config.py — Configuration for PolicyPath App

from pathlib import Path
from typing import Dict, Any

# App Configuration
APP_CONFIG = {
    "app_name": "PolicyPath",
    "app_description": "Your indispensable guide to healthcare policy influence",
    "page_config": {
        "page_title": "🏛️PolicyPath",
        "layout": "wide",
        "initial_sidebar_state": "expanded"
    }
}

# Colors & Styling
COLOR_SCHEME = {
    "primary": "#12715D",
    "accent": "#4AB48E", 
    "light": "#E5F4F1",
    "lighter": "#C8EADF",
    "dark": "#0A473B",
    "white": "#FFFFFF"
}

# Data Configuration
DATA_CONFIG = {
    "cache_ttl": 3600,  # Cache time-to-live in seconds
    "sample_size": {
        "default": 1000,
        "max": 5000,
        "charts": 50000  # Sampling threshold for charting
    },
    "file_locations": {
        "split_data_dir": "data/processed/split",
        "attribution_file": "data/processed/attribution_all_scored.csv",
        "logo_file": "final_deliverable/penta_logo.png",
        "network_file": "data/processed/network_edges.csv"
    }
}

# Chart Configuration
CHART_CONFIG = {
    "default_height": 360,
    "color_sequence": ["#12715D", "#4AB48E", "#CFECE4", "#E7F6F1"],
    "sankey_config": {
        "default_sources": 15,
        "default_targets": 6,
        "default_max_links": 120,
        "node_padding": 26,
        "link_thickness": 22
    }
}

# Search Configuration
SEARCH_CONFIG = {
    "min_search_length": 2,
    "max_suggestions": 5,
    "recent_searches_limit": 10,
    "search_columns": [
        "publication_name",
        "author_name", 
        "channel_name",
        "source_name",
        "publisher_name"
    ],
    "text_columns": [
        "headline",
        "content",
        "body",
        "text"
    ]
}

# Export Configuration
EXPORT_CONFIG = {
    "supported_formats": ["csv", "json"],
    "default_format": "csv",
    "max_filename_length": 40
}
