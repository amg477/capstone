# app_modular.py — Modular PolicyPath Application

import streamlit as st
import pandas as pd

# Import our modular components
from config import APP_CONFIG
from ui_formatting import (
    apply_css_styling, 
    render_header, 
    init_session_state,
    render_debug_info
)
from data_handler import get_data
from dashboard_analytics import (
    render_kpi_metrics,
    render_chart_with_grouping,
    render_sankey_chart,
    render_filter_controls,
    apply_filters,
    get_chart_column_options
)

# Initialize Streamlit page config
st.set_page_config(
    page_title=APP_CONFIG["page_config"]["page_title"],
    layout=APP_CONFIG["page_config"]["layout"],
    initial_sidebar_state=APP_CONFIG["page_config"]["initial_sidebar_state"]
)

# Initialize session state and styling
init_session_state()
apply_css_styling()
render_header()

# Debug Information
render_debug_info()

# Main App Logic
@st.cache_data
def load_app_data():
    """Load and cache app data."""
    return get_data()

# Load data
df_main, df_attr, columns = load_app_data()

if df_main.empty:
    st.error("❌ No data loaded. Please check data paths.")
    st.stop()

# Main Tabs
tab_home, tab_paths, tab_pulse, tab_people = st.tabs([
    "🏛️ PolicyPath Home", 
    "🎯 Paths", 
    "📊 Pulse", 
    "🕸️ People"
])

with tab_home:
    st.markdown("""
    ## Welcome to 🏛️PolicyPath
    **Your indispensable guide to healthcare policy influence**
    
    PolicyPath maps how narratives travel through publications, authors, and channels—pinpointing key voices shaping U.S. healthcare policy.
    
    ### Key Capabilities
    🎯 **Paths**: Analyze influence attribution by publication, author, channel, and terms  
    📊 **Pulse**: Monitor KPIs and narrative trends  
    🕸️ **People**: Explore the network driving influence
    
    ### Quick Stats
    """)
    
    # Show key metrics on home page
    render_kpi_metrics(df_main, df_attr)

with tab_paths:
    st.subheader("🎯 Paths - Attribution Analysis")
    st.markdown("Discover the influence pathways in healthcare policy.")
    
    # Search Type Selection
    col1, col2 = st.columns([2, 1])
    with col1:
        lookup_type = st.radio("Search Type", ["Item Attribution", "Term Attribution"], horizontal=True)
    
    available_cols = df_main.columns.tolist() if not df_main.empty else []
    
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
                        s = df_main[sel_col].astype("string", errors="ignore")
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
                    try:
                        s = df_main[sel_col].astype("string", errors="ignore")
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
                                        if "circulation_size" in item_rows.columns:
                                            avg_circulation = item_rows['circulation_size'].mean()
                                            st.markdown(f"<div class='metric-card'><h4>Avg Circulation</h4><div style='font-size:2rem;color:#12715D;'>{avg_circulation:,.0f}</div></div>", unsafe_allow_html=True)
                                    with c3:
                                        if "body_token_count" in item_rows.columns:
                                            avg_tokens = item_rows['body_token_count'].mean()
                                            st.markdown(f"<div class='metric-card'><h4>Avg Tokens</h4><div style='font-size:2rem;color:#12715D;'>{avg_tokens:,.0f}</div></div>", unsafe_allow_html=True)
                                    
                                    st.dataframe(item_rows, use_container_width=True, height=400)
                                    
                                    # Export buttons
                                    c1, c2 = st.columns(2)
                                    with c1:
                                        if st.button("📥 Export CSV"):
                                            csv = item_rows.to_csv(index=False)
                                            st.download_button(
                                                label="Download CSV",
                                                data=csv,
                                                file_name=f"{sel_col}_{str(selected_item)[:40]}.csv",
                                                mime="text/csv"
                                            )
                                    with c2:
                                        if st.button("📊 Export JSON"):
                                            json = item_rows.to_json(orient="records", indent=2)
                                            st.download_button(
                                                label="Download JSON",
                                                data=json,
                                                file_name=f"{sel_col}_{str(selected_item)[:40]}.json",
                                                mime="application/json"
                                            )
                                else:
                                    st.warning("No rows found for the selected item.")
                        else:
                            st.warning(f"No matches for '{search_term}' in {sel_col}.")
                    except Exception as e:
                        st.error(f"Error searching: {e}")
    else:  # Term Attribution
        st.markdown("### 🔍 Term Search")
        term = st.text_input("Type a term to search", placeholder="Enter a policy term or keyword...")
        
        if term and len(term) >= 2 and not df_main.empty:
            try:
                text_cols = [c for c in available_cols if any(k in c.lower() for k in ["headline", "body", "content", "text"])]
                
                # Search across text columns
                mask = pd.Series(False, index=df_main.index)
                for c in text_cols:
                    s = df_main[c].astype("string", errors="ignore")
                    mask |= s.fillna("").str.contains(term, case=False, na=False)
                
                hits = df_main[mask].head(100)
                if not hits.empty:
                    st.success(f"Found {len(hits)} articles containing '{term}'")
                    
                    # Show key metrics
                    c1, c2, c3 = st.columns(3)
                    with c1:
                        st.markdown(f"<div class='metric-card'><h4>Total Matches</h4><div style='font-size:2rem;color:#12715D;'>{len(hits):,}</div></div>", unsafe_allow_html=True)
                    with c2:
                        if "circulation_size" in hits.columns:
                            total_reach = hits['circulation_size'].sum()
                            st.markdown(f"<div class='metric-card'><h4>Total Reach</h4><div style='font-size:2rem;color:#12715D;'>{total_reach:,.0f}</div></div>", unsafe_allow_html=True)
                    with c3:
                        # Calculate unique publications mentioning the term
                        if "publication_name" in hits.columns:
                            unique_pubs = hits['publication_name'].nunique()
                            st.markdown(f"<div class='metric-card'><h4>Publications</h4><div style='font-size:2rem;color:#12715D;'>{unique_pubs:,}</div></div>", unsafe_allow_html=True)
                    
                    st.markdown("### 📄 Sample Results")
                    st.dataframe(hits, use_container_width=True, height=400)
                    
                    # Export buttons
                    c1, c2 = st.columns(2)
                    with c1:
                        csv = hits.to_csv(index=False)
                        st.download_button(
                            label="📥 Download CSV",
                            data=csv,
                            file_name=f"term_search_{term[:40]}.csv",
                            mime="text/csv"
                        )
                    with c2:
                        json = hits.to_json(orient="records", indent=2)
                        st.download_button(
                            label="📊 Download JSON",
                            data=json,
                            file_name=f"term_search_{term[:40]}.json",
                            mime="application/json"
                        )
                else:
                    st.warning(f"No articles found containing '{term}'.")
            except Exception as e:
                st.error(f"Error searching for term: {e}")

with tab_pulse:
    st.subheader("📊 Pulse - Real-time Analytics")
    st.markdown("Monitor the pulse of healthcare policy influence.")
    
    # Render filters
    filters = render_filter_controls(df_main)
    
    # Apply filters  
    filtered_df = apply_filters(df_main, filters)
    
    st.markdown(f"**Showing {len(filtered_df):,} records** (of {len(df_main):,} total)")
    
    # Render KPI metrics
    render_kpi_metrics(filtered_df, df_attr)
    
    st.divider()
    
    # Charts
    chart_options = get_chart_column_options(filtered_df)
    
    if chart_options['categorical']:
        col1, col2 = st.columns(2)
        
        with col1:
            group_col = st.selectbox("Group charts by", chart_options['categorical'])
            
        with col2:
            top_n = st.slider("Show top N", 5, 50, 20)
        
        # Render grouping chart
        render_chart_with_grouping(filtered_df, group_col, top_n=top_n)
        
        # Sankey chart (if multiple categorical columns)
        if len(chart_options['categorical']) >= 2:
            st.divider()
            
            sankey_cols = st.columns(2)
            with sankey_cols[0]:
                source_col = st.selectbox("Sankey Source", chart_options['categorical'], key="sankey_source")
            with sankey_cols[1]:
                remaining_cols = [c for c in chart_options['categorical'] if c != source_col]
                target_col = st.selectbox("Sankey Target", remaining_cols, key="sankey_target")
            
            if source_col != target_col:
                render_sankey_chart(filtered_df, source_col, target_col)
    
    # Sample data table
    st.divider()
    st.markdown("### 📊 Sample Data")
    
    sample_size = st.slider("Sample Size", 100, 2000, 500)
    sample_df = filtered_df.head(sample_size)
    
    st.dataframe(sample_df, use_container_width=True, height=400)

with tab_people:
    st.subheader("🕸️ People - Network Intelligence")
    st.markdown("Explore relationships between publications, authors, channels, and terms.")
    
    # Network CSV discovery
    from pathlib import Path
    
    edges_path = None
    possible_paths = [
        Path("../data/processed/network_edges.csv"),
        Path.cwd() / "../data/processed/network_edges.csv",
        Path("../data/network_edges.csv"),
        Path.cwd() / "../data/network_edges.csv"
    ]
    
    for p in possible_paths:
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

        else:
            st.warning("network_edges.csv found but must contain columns: source, target, weight")
    else:
        st.info("No network_edges.csv found. Using attribution dataset...")
        
        # Show attribution data as network proxy
        if df_attr is not None and not df_attr.empty:
            st.markdown("### Attribution Data Available")
            st.markdown(f"**Found {len(df_attr):,} attribution records**")
            
            # Basic attribution stats
            if "tag_name" in df_attr.columns:
                top_tags = df_attr["tag_name"].value_counts().head(20)
                st.markdown("### Top Tags by Frequency")
                st.bar_chart(top_tags)
            
            # Show sample attribution data
            st.markdown("### Sample Attribution Data")
            st.dataframe(df_attr.head(500), use_container_width=True, height=400)
            
            # Export attribution data
            c1, c2 = st.columns(2)
            with c1:
                csv = df_attr.to_csv(index=False)
                st.download_button(
                    label="Download Attribution CSV",
                    data=csv,
                    file_name="attribution_analysis.csv",
                    mime="text/csv"
                )
            with c2:
                json = df_attr.to_json(orient="records", indent=2)
                st.download_button(
                    label="Download Attribution JSON", 
                    data=json,
                    file_name="attribution_analysis.json",
                    mime="application/json"
                )
        else:
            st.warning("No attribution or network data found.")
