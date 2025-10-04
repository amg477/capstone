# ui_formatting.py — UI Components & Styling

import base64
from pathlib import Path
import streamlit as st

from config import COLOR_SCHEME, APP_CONFIG

def apply_css_styling():
    """Apply custom CSS styling to the app."""
    st.markdown(f"""
    <style>
        :root {{
            --penta-primary: {COLOR_SCHEME['primary']};
            --penta-accent: {COLOR_SCHEME['accent']};
            --penta-light: {COLOR_SCHEME['light']};
            --penta-lighter: {COLOR_SCHEME['lighter']};
            --penta-dark: {COLOR_SCHEME['dark']};
            --penta-white: {COLOR_SCHEME['white']};
            --penta-bg-texture: #f8fcff;
        }}
        
        /* Main background with textured light blue */
        .stApp {{
            background: var(--penta-bg-texture);
            background-image: 
                radial-gradient(circle at 20% 80%, rgba(120, 220, 255, 0.1) 0%, transparent 50%),
                radial-gradient(circle at 80% 20%, rgba(120, 220, 255, 0.15) 0%, transparent 50%),
                radial-gradient(circle at 40% 40%, rgba(120, 220, 255, 0.05) 0%, transparent 50%);
            background-size: 400px 400px, 600px 600px, 800px 800px;
            background-position: 0 0, 100px 100px, 200px 200px;
            background-attachment: fixed;
        }}
        
        .main .block-container {{
            padding-top: 0.5rem;
            padding-bottom: 2rem;
            max-width: 1200px;
        }}
        
        /* Remove extra white space from header */
        .stApp > div:first-child > div:first-child {{
            margin-top: 0.5rem !important;
        }}
        
        h1, h2, h3 {{
            color: var(--penta-dark);
            font-family: 'Inter','Helvetica Neue',Arial,sans-serif;
            font-weight: 600;
        }}
        h1 {{ font-size: 2.5rem; letter-spacing: -0.02em; }}
        h2 {{ font-size: 1.8rem; margin-top: 2rem; margin-bottom: 1rem; }}
        h3 {{ font-size: 1.4rem; margin-top: 1.5rem; margin-bottom: 0.75rem; }}
        
        .stTabs [data-baseweb="tab-list"] {{ gap: 1rem; }}
        .stTabs [data-baseweb="tab"] {{
            background-color: var(--penta-light);
            border-radius: 8px;
            padding: 0.75rem 1.5rem;
            font-weight: 500;
            color: var(--penta-dark);
        }}
        .stTabs [aria-selected="true"] {{
            background-color: var(--penta-primary);
            color: var(--penta-white);
        }}
        
        .stButton > button {{
            background-color: var(--penta-primary);
            color: var(--penta-white);
            border-radius: 6px;
            font-weight: 500;
            transition: all 0.3s ease;
            box-shadow: 0 2px 4px rgba(18,113,93,0.2);
        }}
        .stButton > button:hover {{
            background-color: var(--penta-accent);
            transform: translateY(-1px);
            box-shadow: 0 4px 8px rgba(18,113,93,0.3);
        }}
        
        .metric-card {{
            background: linear-gradient(135deg, var(--penta-light) 0%, var(--penta-lighter) 100%);
            border-radius: 12px;
            padding: 1.5rem;
            margin: 0.5rem 0;
            border-left: 4px solid var(--penta-primary);
            box-shadow: 0 2px 8px rgba(18,113,93,.1);
            transition: transform .2s ease;
        }}
        .metric-card:hover {{
            transform: translateY(-2px);
            box-shadow: 0 4px 16px rgba(18,113,93,.2);
        }}
        
        .stDataFrame {{
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 8px rgba(0,0,0,.1);
        }}
        .stDataFrame th {{
            background: linear-gradient(135deg, var(--penta-primary) 0%, var(--penta-accent) 100%);
            color: var(--penta-white);
            font-weight: 600;
            padding: 12px;
        }}
        .stDataFrame td {{
            padding: 10px 12px;
            border-bottom: 1px solid var(--penta-light);
        }}
        
        .success-message {{
            background: linear-gradient(135deg, #4CAF50, #45a049);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        .error-message {{
            background: linear-gradient(135deg, #f44336, #d32f2f);
            color: white;
            padding: 1rem;
            border-radius: 8px;
            margin: 1rem 0;
        }}
        
        .dark-mode-toggle {{
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
        }}
        .dark-mode-toggle:hover {{
            background: var(--penta-accent);
            transform: scale(1.1);
        }}
    </style>
    """, unsafe_allow_html=True)

def render_header():
    """Render the app header with logo."""
    logo_path = Path(APP_CONFIG["logo_file"]) if "logo_file" in APP_CONFIG else None
    
    # Try to find logo file
    possible_logo_paths = [
        Path("final_deliverable/penta_logo.png"),
        Path.cwd() / "final_deliverable/penta_logo.png",
        Path("../final_deliverable/penta_logo.png"),
        logo_path,
    ]
    
    logo_file = None
    for p in possible_logo_paths:
        if p and p.exists():
            logo_file = p
            break
    
    if logo_file:
        try:
            with open(logo_file, "rb") as f:
                logo_b64 = base64.b64encode(f.read()).decode("utf-8")
            
            st.markdown(f"""
            <style>
            .header-bar {{
                display: flex;
                align-items: center;
                margin-bottom: 2rem;
                padding: 1rem 0;
                border-bottom: 2px solid #E5F4F1;
            }}
            .penta-logo {{
                height: 120px;
                width: auto;
                margin-right: 20px;
            }}
            .header-title h1 {{
                margin: 0;
                color: #0A473B;
                font-family: 'Inter','Helvetica Neue',Arial,sans-serif;
                font-weight: 600;
                font size: 2.5rem;
                letter-spacing: -0.02em;
            }}
            .header-subtitle {{
                color: #12715D;
                font-size: 1.1rem;
                font-weight: 400;
                margin-top: 0.25rem;
            }}
            </style>
            <div class="header-bar">
                <img src="data:image/png;base64,{logo_b64}" class="penta-logo"/>
                <div class="header-title">
                    <h1>{APP_CONFIG['app_name']}</h1>
                    <div class="header-subtitle">{APP_CONFIG['app_description']}</div>
                </div>
                <div style="margin-left:auto; display:flex; align-items:center; gap:1rem;">
                    <button class="dark-mode-toggle" onclick="toggleDarkMode()" title="Toggle Dark Mode">🌙</button>
                </div>
            </div>
            <script>
            function toggleDarkMode() {{
                document.body.classList.toggle('dark-mode');
                localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
            }}
            if (localStorage.getItem('darkMode') === 'true') {{
                document.body.classList.add('dark-mode');
            }}
            </script>
            """, unsafe_allow_html=True)
        except Exception as e:
            print(f"Error loading logo: {e}")
            render_header_text_only()
    else:
        render_header_text_only()

def render_header_text_only():
    """Render header without logo."""
    st.markdown(f"""
    <div class="header-bar" style="display:flex; align-items:center; margin-bottom:2rem; padding:1rem 0; border-bottom: 2px solid #E5F4F1;">
        <div class="header-title">
            <h1 style="margin:0; color:#0A473B;">{APP_CONFIG['app_name']}</h1>
            <div class="header-subtitle" style="color:#12715D;">{APP_CONFIG['app_description']}</div>
        </div>
        <div style="margin-left:auto;">
            <button class="dark-mode-toggle" onclick="toggleDarkMode()" title="Toggle Dark Mode">🌙</button>
        </div>
    </div>
    <script>
    function toggleDarkMode() {{
        document.body.classList.toggle('dark-mode');
        localStorage.setItem('darkMode', document.body.classList.contains('dark-mode'));
    }}
    if (localStorage.getItem('darkMode') === 'true') {{
        document.body.classList.add('dark-mode');
    }}
    </script>
    """, unsafe_allow_html=True)

def show_success_message(message: str):
    """Display a success message."""
    st.markdown(f'<div class="success-message">✅ {message}</div>', unsafe_allow_html=True)

def show_error_message(message: str):
    """Display an error message.""" 
    st.markdown(f'<div class="error-message">❌ {message}</div>', unsafe_allow_html=True)

def render_metric_card(title: str, value: str, subtitle: str = ""):
    """Render a styled metric card."""
    subtitle_html = f'<div style="font-size:0.9rem;color:#666;margin-top:0.25rem;">{subtitle}</div>' if subtitle else ""
    st.markdown(f"""
    <div class="metric-card">
        <h4 style="margin:0;font-size:1rem;">{title}</h4>
        <div style="font-size:2rem;color:{COLOR_SCHEME['primary']};font-weight:600;">{value}</div>
        {subtitle_html}
    </div>
    """, unsafe_allow_html=True)

def render_debug_info():
    """Render debug information panel."""
    from pathlib import Path
    
    with st.expander("🔧 Debug Information", expanded=False):
        st.write("**Current working directory:**", Path.cwd())
        st.write("**App working directory:**", Path(__file__).parent)
        
        # Check data paths
        data_paths = [
            Path("data/processed/split"),
            Path.cwd() / "data/processed/split",
            Path("../data/processed/split"),
        ]
        
        st.write("**Data Paths:**")
        for p in data_paths:
            exists = p.exists()
            st.write(f"- {p.resolve()}: {'✅ EXISTS' if exists else '❌ NOT FOUND'}")
            if exists:
                files = list(p.glob("final_model_dataset_part_*.csv"))
                st.write(f"  Found {len(files)} split files")

def init_session_state():
    """Initialize Streamlit session state."""
    defaults = {
        'dark_mode': False,
        'recent_searches': [],
        'favorites': [],
        'saved_views': {},
        'current_view': None,
        'current_search': None,
    }
    
    for key, default_value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = default_value
