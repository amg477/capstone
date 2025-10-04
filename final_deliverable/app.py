# app.py — PolicyPath (pandas-only; ready for Streamlit Cloud)
from __future__ import annotations

# -------------------- MUST be first Streamlit call --------------------
import streamlit as st

# Minimal configuration for Streamlit Cloud
st.set_page_config(
    page_title="PolicyPath",
    layout="wide"
)

# -------------------- Imports --------------------
import re
from typing import Dict, Optional, Tuple, List, Set
import pandas as pd
import altair as alt
import plotly.express as px
import plotly.graph_objects as go
import base64
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from textblob import TextBlob
from pathlib import Path

# -------------------- CSS Injection --------------------
@st.cache_resource
def inject_css(path: str = "style.css"):
    """
    Load external CSS once (replaces all inline <style> blocks).
    Searches the app root, then final_deliverable/.
    """
    try:
        p = Path(path)
        if not p.exists():
            fallback = Path("final_deliverable/style.css")
            p = fallback if fallback.exists() else None
        if p:
            css_content = p.read_text(encoding='utf-8')
            st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)
            return True
        else:
            st.warning("Custom style.css not found. Using Streamlit defaults.")
            return False
    except Exception as e:
        st.warning(f"Could not load CSS: {e}. Using Streamlit defaults.")
        return False

# -------------------- Sentiment Wordclouds --------------------
def create_sentiment_wordclouds_from_attribution(df_attr: pd.DataFrame, text_col: str = "terms"):
    """
    Create word clouds for positive, neutral, and negative sentiment analysis using TextBlob.
    Uses attribution terms data instead of headlines for more relevant sentiment analysis.
    """
    if df_attr is None or df_attr.empty or text_col not in df_attr.columns:
        return None, None, None, 0, 0, 0

    # sample first 50 for performance
    text_data = [str(x) for x in df_attr[text_col].dropna().tolist()[:50]]

    positive_texts, negative_texts, neutral_texts = [], [], []
    for text in text_data:
        try:
            polarity = TextBlob(text).sentiment.polarity
            if polarity > 0.1:
                positive_texts.append(text)
            elif polarity < -0.1:
                negative_texts.append(text)
            else:
                neutral_texts.append(text)
        except Exception:
            continue

    def extract_words(texts: List[str]) -> List[str]:
        words: List[str] = []
        stop_words = {
            'the', 'and', 'for', 'are', 'with', 'this', 'that', 'have', 'from', 'they', 'will', 'been', 'said',
            'each', 'which', 'their', 'time', 'about', 'after', 'how', 'its', 'may', 'more', 'new', 'not', 'than',
            'two', 'use', 'what', 'when', 'where', 'who', 'but', 'also', 'can', 'get', 'much', 'such', 'now', 'were',
            'would', 'because'
        }
        for text in texts:
            text_words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
            words.extend([w for w in text_words if w not in stop_words])
        return words

    pos_words = extract_words(positive_texts)
    neg_words = extract_words(negative_texts)
    neu_words = extract_words(neutral_texts)

    def create_wordcloud(words: List[str]):
        if not words:
            return None
        word_text = " ".join(words).strip()
        if not word_text:
            return None
        # keep white background for light theme
        return WordCloud(width=400, height=300, background_color="white").generate(word_text)

    wc_positive = create_wordcloud(pos_words)
    wc_negative = create_wordcloud(neg_words)
    wc_neutral = create_wordcloud(neu_words)

    return wc_positive, wc_neutral, wc_negative, len(positive_texts), len(negative_texts), len(neutral_texts)

# -------------------- App paths (define early; used by Debug/Logo) --------------------
APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent

# -------------------- Load Split Dataset Files --------------------
@st.cache_data
def load_combined_dataset() -> pd.DataFrame:
    """Load and combine a subset of split dataset files for memory efficiency."""
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
                # Keep top 70% by circulation (high-impact)
                if 'circulation_size' in df.columns:
                    df['circulation_size'] = pd.to_numeric(df['circulation_size'], errors='coerce')
                    circulation_threshold = df['circulation_size'].quantile(0.1)
                    combined.append(df[df['circulation_size'] >= circulation_threshold])
                else:
                    combined.append(df.sample(n=int(len(df) * 0.9)))
            except Exception:
                # skip bad file but continue
                pass

    if not combined:
        return pd.DataFrame()

    final_df = pd.concat(combined, ignore_index=True).drop_duplicates()
    files_count = len(combined)
    combined.clear()

    st.session_state.dataset_info = {'rows': len(final_df), 'files': files_count}
    return final_df

@st.cache_data
def get_dataset() -> pd.DataFrame:
    return load_combined_dataset()

# -------------------- Helpers --------------------
def apply_penta_style():
    """Optional Altair defaults (call once if desired)."""
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

def add_to_recent_searches(term: str):
    if term and term not in st.session_state.recent_searches:
        st.session_state.recent_searches.insert(0, term)
        st.session_state.recent_searches = st.session_state.recent_searches[:10]

def shorten(label: str, max_len: int = 28) -> str:
    s = str(label).strip()
    if len(s) <= max_len:
        return s
    # Try to break at word boundaries for better readability
    if ' ' in s:
        words = s.split()
        result = ""
        for word in words:
            if len(result + " " + word) <= max_len - 1:
                result += (" " + word) if result else word
            else:
                break
        if result:
            return result + "…"
    return s[:max_len-1] + "…"

def export_data_button(df: pd.DataFrame, filename: str, fmt: str = "csv"):
    if df is None or df.empty:
        st.info("No data to export.")
        return
    if fmt == "csv":
        st.download_button(
            label=f"Download {filename}.csv",
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
            with st.spinner("Loading dataset..."):
                df_main = get_dataset()
                if df_main is None:
                    df_main = pd.DataFrame()
                COLUMNS = set(df_main.columns) if not df_main.empty else set()

            attr_csv = _find_first_existing(ATTR_NAME, "attribution_all_scored_sample.csv")
            if attr_csv:
                try:
                    with st.spinner("Loading attribution data..."):
                        df_attr = pd.read_csv(attr_csv, dtype_backend="pyarrow")
                except Exception as e:
                    st.warning(f"⚠️ Could not load attribution data: {e}")
                    df_attr = pd.DataFrame()
            else:
                df_attr = pd.DataFrame()
        except Exception as e:
            st.error(f"Failed to load data: {e}")
            df_main, df_attr, COLUMNS = pd.DataFrame(), pd.DataFrame(), set()
    return df_main, df_attr, COLUMNS

# -------------------- Header / Logo --------------------
def render_header():
    logo_path = None
    for p in [
        ROOT / "final_deliverable" / "penta_logo.png",
        ROOT / "data" / "penta_logo.png",  # optional extra fallback
        LOGO_FALLBACK,
    ]:
        if p.exists():
            logo_path = p
            break

    if logo_path and logo_path.exists():
        with open(logo_path, "rb") as f:
            logo_b64 = base64.b64encode(f.read()).decode("utf-8")
        st.markdown(
            f"""
            <div class="header-bar">
                <img src="data:image/png;base64,{logo_b64}" class="penta-logo" alt="Penta logo"/>
                <div class="header-title">
                    <h1>PolicyPath</h1>
                    <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
                </div>
                <div class="header-spacer"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="header-bar">
                <div class="header-title">
                    <h1>PolicyPath</h1>
                    <div class="header-subtitle">Your indispensable guide to healthcare policy influence</div>
                </div>
                <div class="header-spacer"></div>
            </div>
            """,
            unsafe_allow_html=True,
        )

# -------------------- Main App --------------------
def main():
    """Main application function with error handling."""
    try:
        # Load CSS after Streamlit is fully initialized
        inject_css()
        
        render_header()
        tab1, tab2, tab3 = st.tabs(["PolicyPath", "Paths", "People"])
        
        return tab1, tab2, tab3
    except Exception as e:
        st.error(f"Application error: {e}")
        st.info("Please refresh the page or contact support if the issue persists.")
        return None, None, None

# Run main app
tab1, tab2, tab3 = main()

if tab1 is None:
    st.stop()

with tab1:
    _ = apply_penta_style()  # optional; sets Altair defaults

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
        total_pubs = df_main['publication_name'].nunique() if 'publication_name' in df_main.columns else 0
        total_authors = df_main['author_name'].nunique() if 'author_name' in df_main.columns else 0
        total_articles = len(df_main)
        avg_circulation = df_main['circulation_size'].mean() if 'circulation_size' in df_main.columns else 0

        # Attribution metrics if available
        if not df_attr.empty and 'credit_share' in df_attr.columns:
            avg_influence = df_attr['credit_share'].mean()
            top_influence = df_attr['credit_share'].max()
        else:
            avg_influence = 0
            top_influence = 0

        # Row 1: Content Summary Metrics
        metric1, metric2, metric3 = st.columns(3)
        with metric1:
            st.markdown(
                f"""
                <div class="metric-card">
                  <h4>Total Publications</h4>
                  <div style="font-size:2.5rem;">{total_pubs:,}</div>
                  <p>Unique sources</p>
                </div>
                """, unsafe_allow_html=True
            )
        with metric2:
            st.markdown(
                f"""
                <div class="metric-card">
                  <h4>Total Authors</h4>
                  <div style="font-size:2.5rem;">{total_authors:,}</div>
                  <p>Policy voices</p>
                </div>
                """, unsafe_allow_html=True
            )
        with metric3:
            st.markdown(
                f"""
                <div class="metric-card">
                  <h4>Total Articles</h4>
                  <div style="font-size:2.5rem;">{total_articles:,}</div>
                  <p>Policy narratives</p>
                </div>
                """, unsafe_allow_html=True
            )

        # Row 2: Influence & Reach Metrics
        metric4, metric5, metric6 = st.columns(3)
        with metric4:
            st.markdown(
                f"""
                <div class="metric-card">
                  <h4>Avg Circulation</h4>
                  <div style="font-size:2.5rem;">{avg_circulation:,.0f}</div>
                  <p>Reach per article</p>
                </div>
                """, unsafe_allow_html=True
            )
        with metric5:
            st.markdown(
                f"""
                <div class="metric-card">
                  <h4>Avg Influence</h4>
                  <div style="font-size:2.5rem;">{avg_influence:#0.1%}</div>
                  <p>Attribution share</p>
                </div>
                """, unsafe_allow_html=True
            )
        with metric6:
            st.markdown(
                f"""
                <div class="metric-card">
                  <h4>Peak Influence</h4>
                  <div style="font-size:2.5rem;">{top_influence:#0.1%}</div>
                  <p>Top attribution</p>
                </div>
                """, unsafe_allow_html=True
            )

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

                agg = (
                    filtered_df
                    .groupby(dim)
                    .agg(agg_dict)
                    .reset_index()
                    .rename(columns={filtered_df.columns[0]: "n", dim: "dim"})
                )
                if infl_col and infl_col in agg: agg = agg.rename(columns={infl_col: "avg_influence"})
                if circ_col and circ_col in agg: agg = agg.rename(columns={circ_col: "total_metric"})
                agg = agg[agg["dim"].notna()]
            else:
                agg = pd.DataFrame(columns=["dim", "avg_influence", "n", "total_metric"])

            top_n = st.slider("Top N", 5, 50, 20, 1)
            cA, cB = st.columns(2)
            if not agg.empty:
                if "avg_influence" in agg.columns:
                    b1 = alt.Chart(agg.sort_values("avg_influence", ascending=False).head(top_n)).mark_bar().encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("avg_influence:Q", title="Avg influence"),
                        tooltip=["dim", alt.Tooltip("avg_influence:Q", format=".3f"), "n"],
                    )
                    b2 = alt.Chart(agg.sort_values("n", ascending=False).head(top_n)).mark_bar().encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("n:Q", title="Count"),
                        tooltip=["dim", "n", alt.Tooltip("avg_influence:Q", format=".3f")],
                    )
                else:
                    b1 = alt.Chart(agg.sort_values("n", ascending=False).head(top_n)).mark_bar().encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("n:Q", title="Count"),
                        tooltip=["dim", "n"],
                    )
                    b2 = alt.Chart(agg.sort_values("total_metric", ascending=False).head(top_n)).mark_bar().encode(
                        y=alt.Y("dim:N", sort="-x", title=None),
                        x=alt.X("total_metric:Q", title="Total Metric"),
                        tooltip=["dim", "total_metric"],
                    )
                cA.altair_chart(b1.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
                cB.altair_chart(b2.properties(height=360).configure_view(strokeWidth=0), use_container_width=True)
            else:
                st.info("No data for current filters.")

            # Pie (when avg_influence exists)
            if infl_col and not agg.empty and "avg_influence" in agg.columns:
                pie_df = agg.sort_values("avg_influence", ascending=False).head(20)
                fig_pie = px.pie(pie_df, names="dim", values="avg_influence")
                fig_pie.update_traces(textinfo="percent+label", pull=[0.02]*len(pie_df))
                fig_pie.update_layout(height=420, margin=dict(l=10, r=10, t=10, b=10))
                st.plotly_chart(fig_pie, use_container_width=True)

        st.divider()

        # Sankey
        if cat_cols:
            st.markdown("### 🔄 Flow Analysis")
            st.markdown("Visualize relationships between different data dimensions. *Automatically filters out noise and dominant 'Other' categories for cleaner visualization.*")
            
            left, right = st.columns(2)
            src = left.selectbox("Sankey source", cat_cols, index=0, key="sank_src")
            tgt = right.selectbox("Sankey target", cat_cols, index=min(1, len(cat_cols)-1), key="sank_tgt")

            c1, c2, c3, c4 = st.columns(4)
            top_sources = c1.slider("Top Sources", 3, 20, 8, 1)
            top_targets = c2.slider("Top Targets", 2, 15, 5, 1)
            max_links = c3.slider("Max Links", 5, 50, 20, 5)
            bucket_other = c4.checkbox("Bucket 'Other'", value=True)

            if src != tgt and not filtered_df.empty:
                src_counts = filtered_df[src].value_counts().head(int(top_sources))
                tgt_counts = filtered_df[tgt].value_counts().head(int(top_targets))
                keep_s = set(src_counts.index.dropna().astype(str))
                keep_t = set(tgt_counts.index.dropna().astype(str))

                nt = filtered_df[[src, tgt]].dropna().copy()
                nt["s"] = nt[src].apply(lambda x: x if str(x) in keep_s else "Other")
                nt["t"] = nt[tgt].apply(lambda x: x if str(x) in keep_t else "Other")
                sdata = (
                    nt.groupby(["s", "t"]).size().reset_index(name="v")
                    .sort_values("v", ascending=False).head(int(max_links))
                )
                if not bucket_other:
                    sdata = sdata[(sdata["s"].isin(keep_s)) & (sdata["t"].isin(keep_t))]

                if not sdata.empty:
                    # Filter out very small flows to reduce clutter
                    min_flow_threshold = sdata["v"].quantile(0.1)  # Remove bottom 10% of flows
                    sdata_clean = sdata[sdata["v"] >= min_flow_threshold].copy()
                    
                    # Remove "Other" if it's too dominant (more than 50% of total flow)
                    total_flow = sdata_clean["v"].sum()
                    other_flow = sdata_clean[sdata_clean["s"] == "Other"]["v"].sum()
                    if other_flow / total_flow > 0.5:
                        sdata_clean = sdata_clean[sdata_clean["s"] != "Other"]
                    
                    if not sdata_clean.empty:
                        nodes = pd.Series(pd.concat([sdata_clean["s"], sdata_clean["t"]])).astype(str).unique().tolist()
                        labels_short = [shorten(x, max_len=20) for x in nodes]  # Shorter labels
                        idx = {n: i for i, n in enumerate(nodes)}

                        # Define a better color palette
                        colors = [
                            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
                            "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5"
                        ]
                        node_colors = [colors[i % len(colors)] for i in range(len(nodes))]

                        fig = go.Figure(go.Sankey(
                            arrangement="snap",
                            node=dict(
                                label=labels_short,
                                pad=35, 
                                thickness=30,
                                line=dict(width=0),
                                color=node_colors,
                            ),
                            link=dict(
                                source=[idx[s] for s in sdata_clean["s"]],
                                target=[idx[t] for t in sdata_clean["t"]],
                                value=sdata_clean["v"],
                                color="rgba(0,0,0,0.2)",
                                hovertemplate="<b>%{source.label}</b> → <b>%{target.label}</b><br>Count: %{value:,}<extra></extra>",
                            ),
                        ))
                        fig.update_layout(
                            title=f"Flow Analysis: {src} → {tgt}",
                            font=dict(family="Arial, sans-serif", size=16, color="black"),
                            margin=dict(l=40, r=40, t=80, b=40), 
                            height=600,
                            plot_bgcolor="white",
                            paper_bgcolor="white"
                        )
                        # Remove text shadows and outlines
                        fig.update_traces(
                            textfont=dict(
                                family="Arial, sans-serif",
                                size=16,
                                color="black"
                            )
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("After filtering, not enough significant flows to display.")
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
                    conv = tmp.groupby("_d")[conv_col].sum().reset_index().rename(columns={"_d": "d", conv_col: "y"})
                else:
                    conv = tmp.groupby("_d").size().reset_index().rename(columns={"_d": "d", 0: "y"})
                conv = conv.sort_values("d")
                if not conv.empty:
                    chart = alt.Chart(conv).mark_bar().encode(
                        x=alt.X("d:T", title="Date"),
                        y=alt.Y("y:Q", title="Count" if not conv_col else "Conversions")
                    ).properties(height=300).interactive()
                    st.altair_chart(chart.configure_view(strokeWidth=0), use_container_width=True)
            except Exception:
                pass

        st.divider()

        # Sample + Export
        st.markdown("### Data Export")
        c1, c2 = st.columns([2, 1])
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

    st.markdown("### 📊 Attribution Terms Sentiment Analysis")
    st.markdown("Analyze sentiment patterns in policy attribution terms using TextBlob")

    if not df_attr.empty:
        # Filter for terms where kind = "term"
        if 'kind' in df_attr.columns:
            terms_df = df_attr[df_attr['kind'] == 'term'].copy()
            if not terms_df.empty:
                # Look for value column (the actual term text)
                value_col = 'value' if 'value' in terms_df.columns else None
                if value_col:
                    try:
                        result = create_sentiment_wordclouds_from_attribution(terms_df, value_col)
                        if result[0] is None:
                            st.info("📦 Required libraries installing... Showing simplified analysis.")
                            sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
                            with sentiment_col1:
                                st.markdown("#### 🟢 Positive Attribution Terms")
                                st.markdown("innovation • progress • benefit • solution • improvement")
                            with sentiment_col2:
                                st.markdown("#### 🟡 Neutral Attribution Terms")
                                st.markdown("analysis • report • study • data • research")
                            with sentiment_col3:
                                st.markdown("#### 🔴 Negative Attribution Terms")
                                st.markdown("concern • issue • challenge • risk • problem")
                        else:
                            wc_positive, wc_neutral, wc_negative, pos_count, neu_count, neg_count = result
                            cloud1, cloud2, cloud3 = st.columns(3)

                            with cloud1:
                                st.markdown(f"#### 🟢 Positive Attribution Terms ({pos_count} terms)")
                                if wc_positive:
                                    try:
                                        plt.figure(figsize=(8, 6))
                                        plt.imshow(wc_positive, interpolation='bilinear')
                                        plt.axis("off")
                                        plt.title("Positive Attribution Terms", fontsize=10)
                                        st.pyplot(plt.gcf())
                                        plt.close()
                                    except Exception as e:
                                        st.warning(f"Could not display positive wordcloud: {e}")
                                else:
                                    st.info("No positive attribution terms found")

                            with cloud2:
                                st.markdown(f"#### 🟡 Neutral Attribution Terms ({neu_count} terms)")
                                if wc_neutral:
                                    try:
                                        plt.figure(figsize=(8, 6))
                                        plt.imshow(wc_neutral, interpolation='bilinear')
                                        plt.axis("off")
                                        plt.title("Neutral Attribution Terms", fontsize=10)
                                        st.pyplot(plt.gcf())
                                        plt.close()
                                    except Exception as e:
                                        st.warning(f"Could not display neutral wordcloud: {e}")
                                else:
                                    st.info("No neutral attribution terms found")

                            with cloud3:
                                st.markdown(f"#### 🔴 Negative Attribution Terms ({neg_count} terms)")
                                if wc_negative:
                                    try:
                                        plt.figure(figsize=(8, 6))
                                        plt.imshow(wc_negative, interpolation='bilinear')
                                        plt.axis("off")
                                        plt.title("Negative Attribution Terms", fontsize=10)
                                        st.pyplot(plt.gcf())
                                        plt.close()
                                    except Exception as e:
                                        st.warning(f"Could not display negative wordcloud: {e}")
                                else:
                                    st.info("No negative attribution terms found")

                            total_terms = pos_count + neu_count + neg_count
                            if total_terms > 0:
                                st.markdown("---")
                                summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                                with summary_col1:
                                    st.metric("Total Terms", f"{total_terms:,}")
                                with summary_col2:
                                    st.metric("Positive %", f"{(pos_count/total_terms)*100:.1f}%")
                                with summary_col3:
                                    st.metric("Neutral %", f"{(neu_count/total_terms)*100:.1f}%")
                                with summary_col4:
                                    st.metric("Negative %", f"{(neg_count/total_terms)*100:.1f}%")
                    except Exception as e:
                        st.error(f"Error in attribution sentiment analysis: {e}")
                        st.info("Using fallback attribution sentiment analysis")
                        sentiment_col1, sentiment_col2, sentiment_col3 = st.columns(3)
                        with sentiment_col1:
                            st.markdown("#### 🟢 Positive Attribution Terms")
                            st.markdown("innovation • progress • benefit • solution • improvement")
                        with sentiment_col2:
                            st.markdown("#### 🟡 Neutral Attribution Terms")
                            st.markdown("analysis • report • study • data • research")
                        with sentiment_col3:
                            st.markdown("#### 🔴 Negative Attribution Terms")
                            st.markdown("concern • issue • challenge • risk • problem")
                else:
                    st.warning("No 'value' column found in attribution data for terms.")
            else:
                st.info("No terms found where kind = 'term' in attribution data.")
        else:
            st.warning("No 'kind' column found in attribution data.")
    else:
        st.info("No attribution data found.")

    st.markdown("---")
    
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
                    # Don't use st.rerun() here - let the search happen naturally

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
                                        # Don't use st.rerun() here - let the selection happen naturally
                            st.markdown("---")
                    except Exception as e:
                        st.warning(f"Error getting suggestions: {e}")

                if f"selected_{sel_col}" in st.session_state:
                    search_term = st.session_state[f"selected_{sel_col}"]
                    st.success(f"Selected: {search_term}")
                    if st.button("Clear Selection", key=f"clear_{sel_col}"):
                        del st.session_state[f"selected_{sel_col}"]
                        # Don't use st.rerun() here - let the clearing happen naturally

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
                                        st.markdown(f"<div class='metric-card'><h4>Records</h4><div style='font-size:2rem;'>{len(item_rows):,}</div></div>", unsafe_allow_html=True)
                                    with c2:
                                        if "circulation_size" in item_rows:
                                            st.markdown(f"<div class='metric-card'><h4>Avg Circulation</h4><div style='font-size:2rem;'>{item_rows['circulation_size'].mean():,.0f}</div></div>", unsafe_allow_html=True)
                                    with c3:
                                        if "body_token_count" in item_rows:
                                            st.markdown(f"<div class='metric-card'><h4>Avg Tokens</h4><div style='font-size:2rem;'>{item_rows['body_token_count'].mean():,.0f}</div></div>", unsafe_allow_html=True)
                                    st.dataframe(item_rows, use_container_width=True, height=400)
                                    export_data_button(item_rows, f"{sel_col}_{str(selected_item)[:40]}", "csv")
                                else:
                                    st.warning("No rows for the selected item.")
                        else:
                            st.warning(f"No matches for '{search_term}' in {sel_col}.")
                    except Exception as e:
                        st.error(f"Error searching: {e}")
    else:
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
                                # Don't use st.rerun() here - let the selection happen naturally
                    st.markdown("---")
            except Exception as e:
                st.warning(f"Error getting term suggestions: {e}")

        if "selected_term" in st.session_state:
            term = st.session_state["selected_term"]
            st.success(f"Selected term: {term}")
            if st.button("Clear Term Selection", key="clear_term"):
                del st.session_state["selected_term"]
                # Don't use st.rerun() here - let the clearing happen naturally

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
                        st.markdown(f"<div class='metric-card'><h4>Total Matches</h4><div style='font-size:2rem;'>{len(hits):,}</div></div>", unsafe_allow_html=True)
                    with c2:
                        if "circulation_size" in hits:
                            st.markdown(f"<div class='metric-card'><h4>Total Reach</h4><div style='font-size:2rem;'>{hits['circulation_size'].sum():,.0f}</div></div>", unsafe_allow_html=True)
                    with c3:
                        date_cols = [c for c in hits.columns if ("date" in c.lower() or "time" in c.lower())]
                        if date_cols:
                            dc = date_cols[0]
                            try:
                                dates = pd.to_datetime(hits[dc], errors="coerce").dropna()
                                st.markdown(f"<div class='metric-card'><h4>Date Span (days)</h4><div style='font-size:2rem;'>{dates.dt.date.nunique()}</div></div>", unsafe_allow_html=True)
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

        required = {"source", "target", "weight"}
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
    st.markdown(
        f"""
        <div style="font-size: 0.8em; text-align: center; margin-top: 2rem;">
            Dataset: {st.session_state.dataset_info['rows']:,} observations from {st.session_state.dataset_info['files']} files.
            Filtered to show top 90% by circulation size for high-impact analysis.
        </div>
        """,
        unsafe_allow_html=True
    )