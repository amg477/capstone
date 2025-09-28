# intro_tab.py — Instructions tab

import streamlit as st

def render(BRAND):
    st.subheader("How to use this app")
    st.markdown(
        """
**Filters (left sidebar)**  
Narrow the dataset by date, publication, channel, sentiment, author, and topic — all charts and tables update instantly.

**Attribution tab**  
*Item* → pick a **dimension** (e.g., `publication_name`) & a **value**; see credit/credit_share/rating, interpretation, and matching rows.  
*Term* → lookup a keyword from the attribution table and see matching articles.

**Dashboard tab**  
KPI cards → totals; two side-by-side bars (avg influence & volume); pie (influence share); **Sankey** flow (choose source/target); conversions over time; filtered sample table.

**Network tab**  
Shows a sample of `network_edges.csv` (columns: `source, target, weight`) & top nodes by strength if present.
"""
    )