# app.py — PolicyPath (pandas-only; ready for Streamlit Cloud)
from __future__ import annotations

# -------------------- MUST be first Streamlit call --------------------
import streamlit as st

# Minimal configuration for Streamlit Cloud
st.set_page_config(
    page_title="PolicyPath", 
    layout="wide")

# Core imports (always needed)
import pandas as pd
from collections import Counter
import re

# Function to create word clouds based on sentiment
def create_sentiment_wordclouds(df, title_col='headline'):
    """
    Create word clouds for positive, neutral, and negative sentiment analysis
    """
    try:
        # Try to import wordcloud and matplotlib
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
    except ImportError:
        st.warning("WordCloud libraries not available. Using simplified sentiment analysis.")
        return None, None, None
    
    if df.empty or title_col not in df.columns:
        return None, None, None
    
    # Sample data for sentiment classification (in a real app, you'd use actual sentiment analysis)
    text_data = df[title_col].dropna().astype(str).tolist()
    
    # Simple keyword-based sentiment classification for demonstration
    positive_keywords = ['healthcare', 'policy', 'innovation', 'community', 'access', 'progress', 'improve', 'benefit', 'support', 'solution']
    negative_keywords = ['crisis', 'challenge', 'burden', 'issue', 'concern', 'problem', 'risk', 'threat', 'difficulty', 'struggle']
    
    positive_texts = []
    negative_texts = []
    neutral_texts = []
    
    for text in text_data:
        text_lower = text.lower()
        pos_count = sum(1 for word in positive_keywords if word in text_lower)
        neg_count = sum(1 for word in negative_keywords if word in text_lower)
        
        if pos_count > neg_count:
            positive_texts.append(text)
        elif neg_count > pos_count:
            negative_texts.append(text)
        else:
            neutral_texts.append(text)
    
    # Extract words for each sentiment
    def extract_words(texts):
        words = []
        for text in texts:
            # Simple word extraction (remove common words)
            text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            # Filter out common stop words
            stop_words = {'the', 'and', 'for', 'are', 'with', 'this', 'that', 'have', 'from', 'they', 'will', 'been', 'said', 'each', 'which', 'their', 'time', 'will', 'about', 'after', 'how', 'its', 'may', 'more', 'new', 'not', 'than', 'two', 'use', 'what', 'when', 'where', 'who'}
            words.extend([w for w in text_words if w not in stop_words])
        return words
    
    pos_words = extract_words(positive_texts)
    neg_words = extract_words(negative_texts)
    neu_words = extract_words(neutral_texts)
    
    # Create word clouds
    def create_wordcloud(words, max_words=30):
        if not words:
            return None
        word_freq = Counter(words).most_common(max_words)
        if not word_freq:
            return None
        
        # Create word cloud
        wordcloud = WordCloud(
            width=300, height=200,
            background_color='#E6F0F8',
            colormap=None,
            max_words=max_words,
            relative_scaling=0.5,
            random_state=42
        ).generate_from_frequencies(dict(word_freq))
        return wordcloud
    
    wc_positive = create_wordcloud(pos_words)
    wc_negative = create_wordcloud(neg_words)
    wc_neutral = create_wordcloud(neu_words)
    
    return wc_positive, wc_neutral, wc_negative

# -------------------- Placeholder for CSS - moved later --------------------
st.markdown("""
<style>
    :root {
        --penta-primary: #12715D;
        --penta-accent: #4AB48E;
        --penta-light: #E5F4F1;
        --penta-lighter: #C8EADF;
        --penta-dark: #0A473B;
        --penta-white: #FFFFFF;
        --penta-bg-texture: #f8fcff;
    }

    .main .block-container {
        padding-top: 0.5rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }

    /* Remove extra white space from header */
    .stApp > div:first-child > div:first-child {
        margin-top: 0.5rem !important;
    }

    h1, h2, h3 {
        color: var(--penta-dark);
        font-family: 'Inter','Helvetica Neue',Arial,sans-serif;
        font-weight: 600;
    }

    h1 {
        font-size: 2.5rem;
        letter-spacing: -0.02em;
    }

    h2 {
        font-size: 1.8rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
        background: linear-gradient(135deg, var(--penta-primary) 0%, var(--penta-accent) 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
    }

    h3 {
        font-size: 1.4rem;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }

    .stTabs [data-baseweb="tab"] {
        background-color: var(--penta-light);
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 500;
        color: var(--penta-dark);
        transition: all 0.3s ease;
        border: 1px solid rgba(18, 56, 93, 0.1);
    }

    .stTabs [aria-selected="true"] {
        background-color: var(--penta-primary);
        color: var(--penta-white);
    }

    .stTabs [data-baseweb="tab"]:hover {
        background-color: var(--penta-accent);
        color: var(--penta-white);
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(18, 113, 93, 0.2);
    }

    .stButton > button {
        background-color: var(--penta-primary);
        color: var(--penta-white);
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
        box-shadow: 0 2px 4px rgba(18,113,93,0.2);
        border: none;
    }

    .stButton > button:hover {
        background-color: var(--penta-accent);
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(18,113,93,0.3);
    }

    .loading-spinner {
        display: inline-block;
        width: 20px;
        height: 20px;
        border: 3px solid var(--penta-light);
        border-radius: 50%;
        border-top-color: var(--penta-primary);
        animation: spin 1s ease-in-out infinite;
    }

    @keyframes spin {
        to { transform: rotate(360deg); }
    }

    .pulse-animation {
        animation: pulse 2s infinite;
    }

    @keyframes pulse {
        0%{opacity:1;}
        50%{opacity:.5;}
        100%{opacity:1;}
    }

    .metric-card {
        background: linear-gradient(135deg, var(--penta-light) 0%, var(--penta-lighter) 100%);
        border-radius: 12px;
        padding: 1.5rem;
        margin: .5rem 0;
        border-left: 4px solid var(--penta-primary);
        box-shadow: 0 2px 8px rgba(18, 113, 93, .1);
        transition: transform .2s ease;
    }

    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 16px rgba(18, 113, 93,.2);
    }

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,.1);
    }

    .stDataFrame th {
        background: linear-gradient(135deg, var(--penta-primary) 0%, var(--penta-accent) 100%);
        color: var(--penta-white);
        font-weight: 600;
        padding: 12px;
    }

    .stDataFrame td {
        padding: 10px 12px;
        border-bottom: 1px solid var(--penta-light);
    }

    .success-message {
        background: linear-gradient(135deg, #4CAF50, #45a049);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .error-message {
        background: linear-gradient(135deg, #f44336, #d32f2f);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }

    .dark-mode-toggle {
        position: fixed;
        top: 20px;
        right: 20px;
        z-index: 1000;
        background: var(--penta-primary);
        color: var(--penta-white);
        border: none;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        cursor: pointer;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        transition: all 0.3s ease;
    }

    .dark-mode-toggle:hover {
        background: var(--penta-accent);
        transform: scale(1.1);
    }

    .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
    }

    .stTextInput > div > div > input {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Imports --------------------
import re
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Set
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import base64

# -------------------- App paths (define early; used by Debug/Logo) --------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# Note: Server options should be set in .streamlit/config.toml or as command line arguments
# Example config.toml entries:
# [server]
# maxMessageSize = 500
# maxUploadSize = 500

# -------------------- Load Split Dataset Files --------------------
@st.cache_data
def load_combined_dataset() -> pd.DataFrame:
    """Load and combine all split dataset files."""
    # Put your repo path first for Cloud
    possible_paths = [
        Path("data/processed/split"),
        Path.cwd() / "data" / "processed" / "split",
        Path("../data/processed/split"),
        APP_DIR.parent / "data" / "processed" / "split",
        APP_DIR.parent.parent / "data" / "processed" / "split",
        Path("data/split"),
        Path.cwd() / "data" / "split",
        APP_DIR / "data" / "split",
        APP_DIR.parent / "data" / "split",
        Path.cwd().parent / "data" / "split",
        Path("../data/split"),
    ]

    split_dir = next((p for p in possible_paths if p.exists()), None)
    if split_dir is None:
        return pd.DataFrame()

    combined: List[pd.DataFrame] = []
    for i in range(1, 6):  # Load only 5 files to reduce memory
        fp = split_dir / f"final_model_dataset_part_{i:03d}.csv"
        if fp.exists():
            try:
                df = pd.read_csv(fp, dtype_backend="pyarrow")
                
                # Filter to keep only high-impact observations (top 70% by circulation)
                if 'circulation_size' in df.columns:
                    df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce')
                    circulation_threshold = df['circulation_size'].quantile(0.3)
                    df_filtered = df[df['circulation_size'] >= circulation_threshold]
                    combined.append(df_filtered)
                else:
                    # If no circulation data, keep 70% randomly
                    df_sample = df.sample(n=int(len(df) * 0.7))
                    combined.append(df_sample)
            except Exception:
                # Skip bad file but keep loading others
                pass

    if not combined:
        return pd.DataFrame()

    final_df = pd.concat(combined, ignore_index=True)
    final_df = final_df.drop_duplicates()
    files_count = len(combined)
    combined.clear()
    
    # Store dataset info for footnote
    st.session_state.dataset_info = {
        'rows': len(final_df),
        'files': files_count
    }
    
    return final_df

@st.cache_data
def get_dataset() -> pd.DataFrame:
    """Get the combined dataset (cached)."""
    return load_combined_dataset()

# -------------------- Brand styling (CSS) --------------------
st.markdown("""
<style>
    :root {
        --penta-primary: #12715D;
        --penta-accent: #4AB48E;
        --penta-light: #E5F4F1;
        --penta-lighter: #C8EADF;
        --penta-dark: #0A473B;
        --penta-white: #FFFFFF;
    }
    .main .block-container { padding-top: 0.5rem; padding-bottom: 2rem; max-width: 1200px; }
    
    /* Remove extra white space from header */
    .stApp > div:first-child > div:first-child {
        margin-top: 0.5rem !important;
    }
    h1, h2, h3 { color: var(--penta-dark); font-family: 'Inter','Helvetica Neue',Arial,sans-serif; font-weight: 600; }
    h1 { font-size: 2.5rem; letter-spacing: -0.02em; }
    h2 { font-size: 1.8rem; margin-top: 2rem; margin-bottom: 1rem; }
    h3 { font-size: 1.4rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 1rem; }
    .stTabs [data-baseweb="tab"] { background-color: var(--penta-light); border-radius: 8px; padding: 0.75rem 1.5rem; font-weight: 500; color: var(--penta-dark); }
    .stTabs [aria-selected="true"] { background-color: var(--penta-primary); color: var(--penta-white); }
    .stButton > button { background-color: var(--penta-primary); color: var(--penta-white); border-radius: 6px; font-weight: 500; transition: all 0.3s ease; box-shadow: 0 2px 4px rgba(18,113,93,0.2); }
    .stButton > button:hover { background-color: var(--penta-accent); transform: translateY(-1px); box-shadow: 0 4px 8px rgba(18,113,93,0.3); }
    .loading-spinner { display:inline-block; width:20px; height:20px; border:3px solid var(--penta-light); border-radius:50%; border-top-color:var(--penta-primary); animation:spin 1s ease-in-out infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .pulse-animation { animation: pulse 2s infinite; }
    @keyframes pulse { 0%{opacity:1;} 50%{opacity:.5;} 100%{opacity:1;} }
    .metric-card { background: linear-gradient(135deg, var(--penta-light) 0%, var(--penta-lighter) 100%); border-radius:12px; padding:1.5rem; margin:.5rem 0; border-left:4px solid var(--penta-primary); box-shadow:0 2px 8px rgba(18,113,93,.1); transition:transform .2s ease; }
    .metric-card:hover { transform: translateY(-2px); box-shadow:0 4px 16px rgba(18,113,93,.2); }
    .stDataFrame { border-radius:12px; overflow:hidden; box-shadow:0 2px 8px rgba(0,0,0,.1); }
    .stDataFrame th { background: linear-gradient(135deg, var(--penta-primary) 0%, var(--penta-accent) 100%); color:var(--penta-white); font-weight:600; padding:12px; }
    .stDataFrame td { padding:10px 12px; border-bottom:1px solid var(--penta-light); }
    .success-message { background: linear-gradient(135deg,#4CAF50,#45a049); color:white; padding:1rem; border-radius:8px; margin:1rem 0; }
    .error-message { background: linear-gradient(135deg,#f44336,#d32f2f); color:white; padding:1rem; border-radius:8px; margin:1rem 0; }
</style>
""", unsafe_allow_html=True)

# -------------------- Helpers --------------------
def apply_penta_style():
    alt.theme.enable('default')
    alt.data_transformers.disable_max_rows()
    return {
        'primary': "#12715D",
        'accent': "#4AB48E",
        'light': "#E5F4F1",
        'lighter': "#C8EADF",
        'dark': "#0A473B",
        'white': "#FFFFFF"
    }

def show_success_message(message: str):
    st.markdown(f"""<div class="success-message">✅ {message}</div>""", unsafe_allow_html=True)

def show_error_message(message: str):
    st.markdown(f"""<div class="error-message">❌ {message}</div>""", unsafe_allow_html=True)

def add_to_recent_searches(term: str):
    if term and term not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, term)
        st.session_state.recent_searches = st.session_state.recent_searches[:10]

def shorten(label: str, max_len: int = 28) -> str:
    s = str(label)
    return s if len(s) <= max_len else s[:max_len-1] + "…"

def export_data_button(df: pd.DataFrame, filename: str, fmt: str = "csv"):
    if df is None or df.empty:
        st.info("No data to export.")
        return
    if fmt == "csv":
        st.download_button(
            label=f"📥 Download {filename}.csv",
            data=df.to_csv(index=False),
            file_name=f"{filename}.csv",
            mime="text/csv",
        )

# -------------------- Session State --------------------
def init_session_state():
    if 'dark_mode' not in st.session_state: st.session_state.dark_mode = False
    if 'recent_searches' not in st.session_state: st.session_state.recent_searches = []
    if 'favorites' not in st.session_state: st.session_state.favorites = []

init_session_state()

# -------------------- Data Globals & Loader --------------------
DATA_DIR = (ROOT / "data").resolve()
ATTR_NAME = "attribution_all_scored.csv"
LOGO_FALLBACK = ROOT / "final_deliverable" / "penta_logo.png"  # adjust if needed

df_main: Optional[pd.DataFrame] = None
df_attr: Optional[pd.DataFrame] = None
COLUMNS: Set[str] = set()

def _find_first_existing(*names: str) -> Optional[Path]:
    candidates = [
        ROOT / "data",
        ROOT / "data" / "processed",
        ROOT / "final_deliverable" / "data",
        APP_DIR / "data",
        APP_DIR.parent / "data",
        Path.cwd() / "data",
        Path.cwd() / "final_deliverable" / "data",
    ]
    for d in candidates:
        for nm in names:
            p = d / nm
            if p.exists():
                return p
    return None

@st.cache_resource
def get_data() -> Tuple[pd.DataFrame, pd.DataFrame, Set[str]]:
    global df_main, df_attr, COLUMNS
    if df_main is None:
        try:
            df_main = get_dataset()
            if df_main is None:
                df_main = pd.DataFrame()
            COLUMNS = set(df_main.columns) if not df_main.empty else set()

            attr_csv = _find_first_existing(ATTR_NAME, "attribution_all_scored_sample.csv")
            if attr_csv:
                try:
                    df_attr = pd.read_csv(attr_csv, dtype_backend="pyarrow")
                except Exception as e:
                    st.warning(f"⚠️ Could not load attribution data: {e}")
                    df_attr = pd.DataFrame()
            else:
                df_attr = pd.DataFrame()
        except Exception as e:
            show_error_message(f"Failed to load data: {e}")
            df_main, df_attr, COLUMNS = pd.DataFrame(), pd.DataFrame(), set()
    return df_main, df_attr, COLUMNS

# -------------------- Header / Logo --------------------
def render_header():
    logo_path = None
    # Try a few logo locations
    for p in [
        ROOT / "final_deliverable" / "penta_logo.png",
        ROOT / "data" / "penta_logo.png",
        LOGO_FALLBACK
    ]:
        if p.exists():
            logo_path = p
            break

    if logo_path and logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <style>
            .header-bar {{
                display: flex; align-items: center; margin-bottom: 2rem; padding: 1rem 0;
                border-bottom: 2px solid #E5F4F1;
            }}
            .penta-logo {{ height: 120px; width: auto; margin-right: 20px; }}
            .header-title h1 {{
                margin: 0; color: #0A473B; font-family: 'Inter','Helvetica Neue',Arial,sans-serif;
                font-weight: 600; font-size: 2.5rem; letter-spacing: -0.02em;
            }}
            .header-subtitle {{ color: #12715D; font-size: 1.1rem; font-weight: 400; margin-top: .25rem; }}
            </style>
            <div class="header-bar">
                <img src="data:image/png;base64,{logo_b64}" class="penta-logo"/>
                <div class="header-title">
                    <h1>PolicyPath</h1>
                    <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
                </div>
                <div style="margin-left:auto;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div class="header-bar" style="display:flex; align-items:center; margin-bottom:2rem; padding:1rem 0; border-bottom: 2px solid #E5F4F1;">
                <div class="header-title">
                    <h1 style="margin:0; color:#0A473B;">PolicyPath</h1>
                    <div class="header-subtitle" style="color:#12715D;">Your indispensable guide to healthcare policy influence</div>
                </div>
                <div style="margin-left:auto;">
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

# Simple header to avoid startup issues
st.markdown("# 🏛️ PolicyPath")
st.markdown("Your indispensable guide to healthcare policy influence")
st.markdown("---")

# -------------------- Apply Custom Styling --------------------
st.markdown("""
<style>
    :root {
        --penta-primary: #12715D;
        --penta-accent: #4AB48E;
        --penta-light: #E5F4F1;
        --penta-lighter: #C8EADF;
        --penta-dark: #0A473B;
        --penta-white: #FFFFFF;
    }

    /* Minimize top whitespace */
    .main .block-container {
        padding-top: 0rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Remove header margin */
    .stApp > div:first-child > div:first-child {
        margin-top: 0rem !important;
    }
    
    /* Reduce padding on main content area */
    .main .block-container > div {
        padding-top: 0rem !important;
    }
    
    /* Minimize space after header */
    main .block-container {
        margin-top: 0rem !important;
    }
    
    /* Hide Streamlit's default header padding */
    .css-1d391kg {
        padding-top: 0rem !important;
    }
</style>
""", unsafe_allow_html=True)

# -------------------- Main Tabs --------------------
tab1, tab2, tab3= st.tabs(["PolicyPath", "Paths", "People"])
    
with tab1:
    st.markdown("""
    ## PolicyPath
    PolicyPath maps how narratives travel through publications, authors, and channels—pinpointing key voices shaping U.S. healthcare policy.

    **Paths**: Analyze influence attribution by publication, author, channel, and terms  
    **People**: Explore the network driving influence. 
    *Built by Georgetown University MSBA - Anna Glass, Jasmin Mendoza, Mohammmad Waqas, Mark Saba, Posy Olivetti*
    """)
    
    # Load data for metrics
    df_main, df_attr, COLUMNS = get_data()
    
    # PolicyPath Metrics Dashboard
    if not df_main.empty:
        st.markdown("### 📊 Policy Impact Dashboard")
        
        # Calculate key metrics
        total_pubs = df_main['publication'].nunique() if 'publication' in df_main.columns else 0
        total_authors = df_main['author'].nunique() if 'author' in df_main.columns else 0
        total_articles = len(df_main)
        avg_circulation = df_main['circulation_size'].mean() if 'circulation_size' in df_main.columns else 0
        
        # Calculate attribution metrics if available
        if not df_attr.empty and 'credit_share' in df_attr.columns:
            avg_influence = df_attr['credit_share'].mean()
            top_influence = df_attr['credit_share'].max()
        else:
            avg_influence = 0
            top_influence = 0
        
        # Row 1: Content Summary Metrics
        metric1, metric2, metric3 = st.columns(3)
        
        with metric1:
            st.markdown(f"""
            <div style="background-color: #E6F0F8; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #12715D;">
                <h4 style="margin: 0; color: #2C3E50;">Total Publications</h4>
                <h2 style="margin: 0; color: #12715D; font-size: 2.5rem;">{total_pubs:,}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Unique sources</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric2:
            st.markdown(f"""
            <div style="background-color: #E6F0F8; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #12715D;">
                <h4 style="margin: 0; color: #2C3E50;">Total Authors</h4>
                <h2 style="margin: 0; color: #12715D; font-size: 2.5rem;">{total_authors:,}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Policy voices</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric3:
            st.markdown(f"""
            <div style="background-color: #E6F0F8; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #12715D;">
                <h4 style="margin: 0; color: #2C3E50;">Total Articles</h4>
                <h2 style="margin: 0; color: #12715D; font-size: 2.5rem;">{total_articles:,}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Policy narratives</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Row 2: Influence & Reach Metrics  
        metric4, metric5, metric6 = st.columns(3)
        
        with metric4:
            st.markdown(f"""
            <div style="background-color: #E6F0F8; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #12715D;">
                <h4 style="margin: 0; color: #2C3E50;">Avg Circulation</h4>
                <h2 style="margin: 0; color: #12715D; font-size: 2.5rem;">{avg_circulation:,.0f}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Reach per article</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric5:
            st.markdown(f"""
            <div style="background-color: #E6F0F8; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #12715D;">
                <h4 style="margin: 0; color: #2C3E50;">Avg Influence</h4>
                <h2 style="margin: 0; color: #12715D; font-size: 2.5rem;">{avg_influence:#0.1%}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Attribution share</p>
            </div>
            """, unsafe_allow_html=True)
        
        with metric6:
            st.markdown(f"""
            <div style="background-color: #E6F0F8; padding: 1rem; border-radius: 10px; text-align: center; border-left: 4px solid #12715D;">
                <h4 style="margin: 0; color: #2C3E50;">Peak Influence</h4>
                <h2 style="margin: 0; color: #12715D; font-size: 2.5rem;">{top_influence:#0.1%}</h2>
                <p style="margin: 0; font-size: 0.9rem; color: #666;">Top attribution</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Word Cloud Visualization Section
        st.markdown("### 📊 Policy Sentiment Analysis")
        st.markdown("Explore the most influential terms across policy narratives")
        
        # For now, use simplified sentiment display (libraries will install later)
        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
        
        with sentiment_col1:
            st.markdown("#### 🟢 Positive Terms")
            st.markdown("healthcare • policy • innovation • community • access")
        
        with sentiment_col2:
            st.markdown("#### 🟡 Neutral Terms") 
            st.markdown("analysis • report • data • review • study")
        
        with sentiment_col3:
            st.markdown("#### 🔴 Negative Terms")
            st.markdown("crisis • challenge • burden • issue • concern")
        
        st.markdown("---")  # Separator before content

    st.subheader("Pulse - Dashboard Overview")
    st.markdown("Monitor the pulse of healthcare policy influence with interactive KPIs and charts.")

    df_main, df_attr, COLUMNS = get_data()
    if df_main.empty:
        st.info("No data loaded.")
    else:
        cols = df_main.columns.tolist()

        # Smart Filters
        st.markdown("### DashboardFilters")
        c1, c2 = st.columns(2)

        with c1:
            date_col = next((c for c in cols if any(k in c.lower() for k in ["date", "time", "ts"])), None)
            date_range = None
            if date_col:
                try:
                    df_main[date_col] = pd.to_datetime(df_main[date_col], errors="coerce")
                    min_d, max_d = df_main[date_col].min(), df_main[date_col].max()
                    if pd.notna(min_d) and pd.notna(max_d):
                        date_range = st.date_input("Date range", value=(min_d.date(), max_d.date()),
                                                   min_value=min_d.date(), max_value=max_d.date())
                except Exception:
                    date_range = None

            pub_cols = [c for c in cols if "publication" in c.lower()]
            if pub_cols:
                pub_col = pub_cols[0]
                pubs = pd.Series(df_main[pub_col]).dropna().unique().tolist()[:50]
                sel_pubs = st.multiselect("Publications", pubs)
            else:
                sel_pubs = []

            channel_cols = [c for c in cols if "channel" in c.lower()]
            if channel_cols:
                channel_col = channel_cols[0]
                channels = pd.Series(df_main[channel_col]).dropna().unique().tolist()[:50]
                sel_channels = st.multiselect("Channels", channels)
            else:
                sel_channels = []

        with c2:
            author_cols = [c for c in cols if "author" in c.lower()]
            if author_cols:
                author_col = author_cols[0]
                authors = pd.Series(df_main[author_col]).dropna().unique().tolist()[:50]
                sel_authors = st.multiselect("Authors", authors)
            else:
                sel_authors = []

            source_cols = [c for c in cols if "source" in c.lower()]
            if source_cols:
                source_col = source_cols[0]
                sources = pd.Series(df_main[source_col]).dropna().unique().tolist()[:20]
                sel_sources = st.multiselect("Source Types", sources)
            else:
                sel_sources = []

        # Apply filters (pandas)
        filtered_df = df_main.copy()
        if date_range and date_col and len(date_range) == 2:
            start_d, end_d = date_range
            filtered_df = filtered_df[
                (filtered_df[date_col] >= pd.Timestamp(start_d)) &
                (filtered_df[date_col] <= pd.Timestamp(end_d))
            ]
        if sel_pubs and pub_cols:
            filtered_df = filtered_df[filtered_df[pub_cols[0]].isin(sel_pubs)]
        if sel_channels and channel_cols:
            filtered_df = filtered_df[filtered_df[channel_cols[0]].isin(sel_channels)]
        if sel_authors and author_cols:
            filtered_df = filtered_df[filtered_df[author_cols[0]].isin(sel_authors)]
        if sel_sources and source_cols:
            filtered_df = filtered_df[filtered_df[source_cols[0]].isin(sel_sources)]

        # KPIs
        try:
            total_pubs = filtered_df[pub_cols[0]].nunique() if pub_cols else 0
            uniq_sources = filtered_df[source_cols[0]].nunique() if source_cols else 0
            uniq_authors = filtered_df[author_cols[0]].nunique() if author_cols else 0
        except Exception:
            total_pubs = uniq_sources = uniq_authors = 0

        infl_col, avg_infl = None, None
        for c in ["pub_credit_share", "max_term_credit", "credit_share"]:
            if c in filtered_df.columns:
                infl_col = c
                avg_infl = filtered_df[c].mean()
                break
        if avg_infl is None and not df_attr.empty and "credit_share" in df_attr.columns:
            avg_infl = df_attr["credit_share"].mean()
            st.info("Using attribution dataset for influence metrics")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total publications", f"{total_pubs:,}")
        m2.metric("Avg influence score", f"{avg_infl:.3f}" if avg_infl is not None else "n/a")
        m3.metric("Unique sources", f"{uniq_sources:,}")
        m4.metric("Unique authors", f"{uniq_authors:,}")

        st.divider()

        # Charts
        cat_cols = [c for c in ["publication_name","source_name","channel_name","author_name","topic","sentiment_band"] if c in filtered_df.columns]
        if not cat_cols:
            st.info("No categorical columns to group by.")
        else:
            dim = st.selectbox("Group charts by", cat_cols, index=0)
            circ_col = next((c for c in ["circulation","circulation_size","reach","impressions","audience"] if c in filtered_df.columns), None)

            if not filtered_df.empty and dim in filtered_df.columns:
                # Importance-based sampling
                sample_size = min(50000, len(filtered_df))
                if len(filtered_df) > sample_size:
                    score = pd.Series(0.0, index=filtered_df.index)
                    if "pub_credit_share" in filtered_df: score += filtered_df["pub_credit_share"].fillna(0) * 1000
                    if "max_term_credit" in filtered_df: score += filtered_df["max_term_credit"].fillna(0) * 1000
                    if "circulation_size" in filtered_df:
                        circ = filtered_df["circulation_size"]
                        norm = (circ - circ.min()) / (circ.max() - circ.min() + 1e-9)
                        score += norm.fillna(0) * 100
                    filtered_df = filtered_df.loc[score.nlargest(sample_size).index]
                    st.info(f"Top {sample_size:,} items by importance")

                agg_dict: Dict[str, str] = {filtered_df.columns[0]: "count"}
                if infl_col and infl_col in filtered_df: agg_dict[infl_col] = "mean"
                if circ_col and circ_col in filtered_df: agg_dict[circ_col] = "sum"

                agg = filtered_df.groupby(dim).agg(agg_dict).reset_index().rename(columns={filtered_df.columns[0]: "n", dim: "dim"})
                if infl_col and infl_col in agg: agg = agg.rename(columns={infl_col: "avg_influence"})
                if circ_col and circ_col in agg: agg = agg.rename(columns={circ_col: "total_metric"})
                agg = agg[agg["dim"].notna()]
            else:
                agg = pd.DataFrame(columns=["dim", "avg_influence", "n", "total_metric"])

            top_n = st.slider("Top N", 5, 50, 20, 1)
            cA, cB = st.columns(2)
            if not agg.empty:
                if "avg_influence" in agg.columns:
                    b1 = alt.Chart(agg.sort_values("avg_influence", ascending=False).head(top_n)).mark_bar(color="#12715D").encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("avg_influence:Q", title="Avg influence"),
                        tooltip=["dim", alt.Tooltip("avg_influence:Q", format=".3f"), "n"],
                    )
                    b2 = alt.Chart(agg.sort_values("n", ascending=False).head(top_n)).mark_bar(color="#4AB48E").encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("n:Q", title="Count"),
                        tooltip=["dim", "n", alt.Tooltip("avg_influence:Q", format=".3f")],
                    )
                else:
                    b1 = alt.Chart(agg.sort_values("n", ascending=False).head(top_n)).mark_bar(color="#12715D").encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("n:Q", title="Count"),
                        tooltip=["dim", "n"],
                    )
                    b2 = alt.Chart(agg.sort_values("total_metric", ascending=False).head(top_n)).mark_bar(color="#4AB48E").encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("total_metric:Q", title="Total Metric"),
                        tooltip=["dim", "total_metric"],
                    )
                cA.altair_chart(b1.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
                cB.altair_chart(b2.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
            else:
                st.info("No data for current filters.")

            # Pie
            if infl_col and not agg.empty and "avg_influence" in agg.columns:
                pie_df = agg.sort_values("avg_influence", ascending=False).head(20)
                fig_pie = px.pie(pie_df, names="dim", values="avg_influence",
                                 color_discrete_sequence=["#12715D", "#4AB48E", "#CFECE4", "#E7F6F1"])
                fig_pie.update_traces(textinfo="percent+label", pull=[0.02]*len(pie_df))
                fig_pie.update_layout(height=420, margin=dict(l=10,r=10,t=10,b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # Sankey
        if not cat_cols:
            pass
        else:
            left, right = st.columns(2)
            src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
            tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols)-1), key="sank_tgt")

            c1, c2, c3, c4 = st.columns(4)
            top_sources = c1.slider("Top Sources", 3, 50, 15, 1)
            top_targets = c2.slider("Top Targets", 2, 20, 6, 1)
            max_links = c3.slider("Max Links", 10, 500, 120, 10)
            bucket_other = c4.checkbox("Bucket 'Other'", value=True)

            if src != tgt and not filtered_df.empty:
                src_counts = filtered_df[src].value_counts().head(int(top_sources))
                tgt_counts = filtered_df[tgt].value_counts().head(int(top_targets))
                keep_s = set(src_counts.index.dropna().astype(str))
                keep_t = set(tgt_counts.index.dropna().astype(str))

                nt = filtered_df[[src, tgt]].dropna().copy()
                nt["s"] = nt[src].apply(lambda x: x if str(x) in keep_s else "Other")
                nt["t"] = nt[tgt].apply(lambda x: x if str(x) in keep_t else "Other")
                sdata = nt.groupby(["s","t"]).size().reset_index(name="v").sort_values("v", ascending=False).head(int(max_links))
                if not bucket_other:
                    sdata = sdata[(sdata["s"].isin(keep_s)) & (sdata["t"].isin(keep_t))]

                if not sdata.empty:
                    nodes = pd.Series(pd.concat([sdata["s"], sdata["t"]])).astype(str).unique().tolist()
                    labels_short = [shorten(x) for x in nodes]
                    idx = {n:i for i,n in enumerate(nodes)}
                    hc = st.checkbox("High-contrast labels", value=True)

                    fig = go.Figure(go.Sankey(
                        arrangement="snap",
                        node=dict(
                            label=labels_short,
                            pad=26, thickness=22,
                            color=["#CFECE4" if hc else "#12715D"] * len(nodes),
                            line=dict(color="rgba(0,0,0,0)", width=0),
                        ),
                        link=dict(
                            source=[idx[s] for s in sdata["s"]],
                            target=[idx[t] for t in sdata["t"]],
                            value=sdata["v"],
                            color="rgba(18,113,93,0.22)" if hc else "rgba(18,113,93,0.35)",
                            hovertemplate="Count: %{value:,}<br>source: %{source.label}<br>target: %{target.label}<extra></extra>",
                        ),
                    ))
                    fig.update_layout(
                        font=dict(family="Inter, Helvetica, Arial, sans-serif", size=16 if hc else 15, color="#133C35"),
                        hoverlabel=dict(font_size=13, font_family="Inter, Helvetica, Arial, sans-serif"),
                        margin=dict(l=8, r=8, t=6, b=6), height=640
                    )
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Not enough data for Sankey with these fields.")
            else:
                st.info("Choose different fields for source and target (and ensure data is filtered).")

        st.divider()

        # Time Series
        time_col = next((c for c in ["conversion_ts", "load_date", "source_time", "date", "timestamp"] if c in filtered_df.columns), None)
        if not time_col:
            date_cols = [c for c in filtered_df.columns if ("date" in c.lower() or "time" in c.lower())]
            time_col = date_cols[0] if date_cols else None

        conv_col = "conversions" if "conversions" in filtered_df.columns else None
        if time_col:
            try:
                ts = pd.to_datetime(filtered_df[time_col], errors="coerce")
                tmp = filtered_df.copy()
                tmp["_d"] = ts.dt.date
                if conv_col and conv_col in tmp:
                    conv = tmp.groupby("_d")[conv_col].sum().reset_index().rename(columns={"_d":"d", conv_col:"y"})
                else:
                    conv = tmp.groupby("_d").size().reset_index().rename(columns={"_d":"d", 0:"y"})
                conv = conv.sort_values("d")
                if not conv.empty:
                    chart = alt.Chart(conv).mark_bar(color="#12715D").encode(
                        x=alt.X("d:T", title="Date"),
                        y=alt.Y("y:Q", title="Count" if not conv_col else "Conversions")
                    ).properties(height=300).interactive()
                    st.altair_chart(chart.configure_view(strokeWidth=0), use_container_width=True)
            except Exception:
                pass

        st.divider()

        # Sample + Export
        st.markdown("### Data Export")
        c1, c2= st.columns([2,1])
        with c1:
            sample_size = st.slider("Sample Size", 100, 5000, 1000)
        with c2:
            if st.button("Export CSV"):
                export_data_button(filtered_df.head(sample_size), "filtered_data", "csv")

with tab2:
    st.subheader("Attribution Analysis")
    st.markdown("Discover the influence pathways in healthcare policy and understand impact patterns.")

    df_main, df_attr, COLUMNS = get_data()
    available_cols = df_main.columns.tolist() if not df_main.empty else []

    col1, col2 = st.columns([2, 1])
    with col1:
        lookup_type = st.radio("Search Type", ["Item Attribution", "Term Attribution"], horizontal=True)
    with col2:
        st.empty()  # Space for future elements

    # Recent searches
    if st.session_state.recent_searches:
        with st.expander("🔍 Recent Searches"):
            for s in st.session_state.recent_searches[:5]:
                if st.button(f"🔍 {s}", key=f"recent_{s}"):
                    st.session_state.current_search = s
                    st.rerun()

    if lookup_type == "Item Attribution":
        if not available_cols:
            st.warning("No searchable columns found.")
        else:
            search_cols = [c for c in available_cols if any(k in c.lower() for k in ["publication", "author", "channel", "publisher"])]
            if not search_cols:
                st.warning("No item columns found.")
            else:
                c1, c2 = st.columns(2)
                with c1:
                    sel_col = st.selectbox("Search by", search_cols)
                with c2:
                    search_term = st.text_input("Search term", placeholder=f"Enter {sel_col}...")

                # Suggestions
                if search_term and len(search_term) >= 2 and sel_col in df_main:
                    try:
                        s = df_main[sel_col].astype("string[pyarrow]", errors="ignore")
                        sugg = s.fillna("").str.contains(search_term, case=False, na=False)
                        suggestions = s[sugg].dropna().drop_duplicates().head(10).tolist()
                        if suggestions:
                            st.markdown("**💡 Suggestions:**")
                            cols = st.columns(min(3, len(suggestions)))
                            for i, val in enumerate(suggestions[:9]):
                                label = (str(val)[:30] + "...") if len(str(val)) > 30 else str(val)
                                with cols[i % 3]:
                                    if st.button(f"🔍 {label}", key=f"sugg_{i}_{sel_col}", help=f"Search for: {val}"):
                                        st.session_state[f"selected_{sel_col}"] = val
                                        st.rerun()
                            st.markdown("---")
                    except Exception as e:
                        st.warning(f"Error getting suggestions: {e}")

                if f"selected_{sel_col}" in st.session_state:
                    search_term = st.session_state[f"selected_{sel_col}"]
                    st.success(f"Selected: {search_term}")
                    if st.button("Clear Selection", key=f"clear_{sel_col}"):
                        del st.session_state[f"selected_{sel_col}"]
                        st.rerun()

                if search_term and sel_col in df_main:
                    add_to_recent_searches(f"{sel_col}: {search_term}")
                    try:
                        s = df_main[sel_col].astype("string[pyarrow]", errors="ignore")
                        matches = s.fillna("").str.contains(search_term, case=False, na=False)
                        options = s[matches].dropna().drop_duplicates().head(20).tolist()
                        if options:
                            st.success(f"Found {len(options)} matches for '{search_term}'")
                            selected_item = st.selectbox("Select item", options, key=f"select_{sel_col}")
                            if selected_item:
                                item_rows = df_main[s.fillna("") == str(selected_item)].head(100)
                                if not item_rows.empty:
                                    st.markdown(f"### 📊 Data for: {selected_item}")
                                    c1, c2, c3 = st.columns(3)
                                    with c1:
                                        st.markdown(f"<div class='metric-card'><h4>Records</h4><div style='font-size:2rem;color:#12715D;'>{len(item_rows):,}</div></div>", unsafe_allow_html=True)
                                    with c2:
                                        if "circulation_size" in item_rows:
                                            st.markdown(f"<div class='metric-card'><h4>Avg Circulation</h4><div style='font-size:2rem;color:#12715D;'>{item_rows['circulation_size'].mean():,.0f}</div></div>", unsafe_allow_html=True)
                                    with c3:
                                        if "body_token_count" in item_rows:
                                            st.markdown(f"<div class='metric-card'><h4>Avg Tokens</h4><div style='font-size:2rem;color:#12715D;'>{item_rows['body_token_count'].mean():,.0f}</div></div>", unsafe_allow_html=True)
                                    st.dataframe(item_rows, use_container_width=True, height=400)
                                    export_data_button(item_rows, f"{sel_col}_{str(selected_item)[:40]}", "csv")
                                else:
                                    st.warning("No rows for the selected item.")
                        else:
                            st.warning(f"No matches for '{search_term}' in {sel_col}.")
                    except Exception as e:
                        st.error(f"Error searching: {e}")
    else:
        st.markdown("###Term Search")
        term = st.text_input("Type a term to search", placeholder="Enter a policy term or keyword...")
        if term and len(term) >= 2 and not df_main.empty:
            try:
                text_cols = [c for c in available_cols if any(k in c.lower() for k in ["headline", "body", "content", "text"])]
                # Suggestions from first 2 text cols
                sugg_list: List[str] = []
                for c in text_cols[:2]:
                    s = df_main[c].astype("string[pyarrow]", errors="ignore")
                    m = s.fillna("").str.contains(term, case=False, na=False)
                    sugg_list.extend(s[m].dropna().drop_duplicates().head(5).tolist())
                uniq_sugg = list(dict.fromkeys([str(x) for x in sugg_list]))[:9]
                if uniq_sugg:
                    st.markdown("**💡 Term Suggestions:**")
                    cols = st.columns(min(3, len(uniq_sugg)))
                    for i, v in enumerate(uniq_sugg):
                        label = v[:40] + "..." if len(v) > 40 else v
                        with cols[i % 3]:
                            if st.button(f"🔍 {label}", key=f"term_sugg_{i}", help=f"Search for: {v}"):
                                st.session_state["selected_term"] = v
                                st.rerun()
                    st.markdown("---")
            except Exception as e:
                st.warning(f"Error getting term suggestions: {e}")

        if "selected_term" in st.session_state:
            term = st.session_state["selected_term"]
            st.success(f"Selected term: {term}")
            if st.button("Clear Term Selection", key="clear_term"):
                del st.session_state["selected_term"]
            st.rerun()

        if term and not df_main.empty:
            add_to_recent_searches(f"Term: {term}")
            try:
                text_cols = [c for c in available_cols if any(k in c.lower() for k in ["headline", "body", "content", "text"])]
                mask = pd.Series(False, index=df_main.index)
                for c in text_cols:
                    s = df_main[c].astype("string[pyarrow]", errors="ignore")
                    mask |= s.fillna("").str.contains(term, case=False, na=False)
                hits = df_main[mask].head(100)
                if not hits.empty:
                    st.success(f"Found {len(hits)} articles containing '{term}'")
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<div class='metric-card'><h4>Total Matches</h4><div style='font-size:2rem;color:#12715D;'>{len(hits):,}</div></div>", unsafe_allow_html=True)
                    with c2:
                        if "circulation_size" in hits:
                            st.markdown(f"<div class='metric-card'><h4>Total Reach</h4><div style='font-size:2rem;color:#12715D;'>{hits['circulation_size'].sum():,.0f}</div></div>", unsafe_allow_html=True)
                    with c3:
                        date_cols = [c for c in hits.columns if ("date" in c.lower() or "time" in c.lower())]
                        if date_cols:
                            dc = date_cols[0]
                            try:
                                dates = pd.to_datetime(hits[dc], errors="coerce").dropna()
                                st.markdown(f"<div class='metric-card'><h4>Date Span (days)</h4><div style='font-size:2rem;color:#12715D;'>{dates.dt.date.nunique()}</div></div>", unsafe_allow_html=True)
                            except Exception:
                                pass
                    st.markdown("### 📄 Sample Results")
                    st.dataframe(hits, use_container_width=True, height=400)
                    export_data_button(hits, f"term_search_{term[:40]}", "csv")
                else:
                    st.warning(f"No articles found containing '{term}'.")
            except Exception as e:
                st.error(f"Error searching for term: {e}")

with tab3:
    st.subheader("People - Network Influence")
    st.markdown("Explore relationships between publications, authors, channels, and terms.")
    # Network CSV discovery
    edges_path = None
    for d in [ROOT / "data", ROOT / "data" / "processed", ROOT / "processed", APP_DIR]:
        p = d / "network_edges.csv"
        if p.exists():
            edges_path = p
            break
    if edges_path and edges_path.exists():
        try:
            edges = pd.read_csv(edges_path, dtype_backend="pyarrow")
        except Exception:
            edges = pd.read_csv(edges_path)

        required = {"source","target","weight"}
        if required.issubset(set(edges.columns)):
            st.write("Edges sample:")
            st.dataframe(edges.head(200), use_container_width=True)

            st.markdown("**Basic node strength**")
            deg = pd.concat(
                [
                    edges.groupby("source")["weight"].sum().rename("out_weight"),
                    edges.groupby("target")["weight"].sum().rename("in_weight"),
                ],
                axis=1,
            ).fillna(0)
            deg["strength"] = deg["in_weight"] + deg["out_weight"]
            st.dataframe(deg.sort_values("strength", ascending=False).head(30))

            min_w = float(edges["weight"].quantile(0.75)) if not edges.empty else 0.0
            min_w = st.slider(
                "Min edge weight to show",
                float(edges["weight"].min()) if not edges.empty else 0.0,
                float(edges["weight"].max()) if not edges.empty else 1.0,
                min_w,
            )
            sub = edges[edges["weight"] >= min_w].copy()
            st.write(f"Filtered edges: {len(sub):,} (of {len(edges):,})")
            st.dataframe(sub.head(200), use_container_width=True)
            st.caption("Tip: For interactive graphs, consider pyvis or Plotly for networks.")
        else:
            st.warning("network_edges.csv found but must contain columns: source, target, weight")
    else:
        st.info("No network_edges.csv found. Place one under data/ or data/processed with columns: source,target,weight.")

# -------------------- Dataset Footnote --------------------
if 'dataset_info' in st.session_state:
    st.markdown("---")
    st.markdown(f"""
    <div style="font-size: 0.8em; color: #666; text-align: center; margin-top: 2rem;">
        Dataset: {st.session_state.dataset_info['rows']:,} observations from {st.session_state.dataset_info['files']} files. 
        Filtered to show top 70% by circulation size for high-impact analysis.
    </div>
    """, unsafe_allow_html=True)