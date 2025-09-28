# app.py — Influence Explorer (tabs + Azure bootstrap via structure.py)

from __future__ import annotations
import streamlit as st
from pathlib import Path

# Local modules
from structure import (
    BRAND, inject_base_css, connect_duckdb_with_azure, maybe_set_logo,
    render_header, build_v_enriched, build_sidebar_filters
)
import intro_tab
import attribution_tab
import dashboard_tab
import network_tab

# -----------------------------------------------------------------------------
# Page / CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="Influence Explorer", layout="wide")
inject_base_css()

# -----------------------------------------------------------------------------
# Connect to Azure + DuckDB, header, and enriched view
# -----------------------------------------------------------------------------
con, PATHS = connect_duckdb_with_azure()
PATHS = maybe_set_logo(PATHS)

render_header(paths=PATHS, title="Influence Explorer")

COLUMNS = build_v_enriched(con)   # creates v_enriched; returns set of column names from `v`

# -----------------------------------------------------------------------------
# Sidebar filters (shared across tabs)
# -----------------------------------------------------------------------------
filters = build_sidebar_filters(con, COLUMNS)

# -----------------------------------------------------------------------------
# Tabs
# -----------------------------------------------------------------------------
tab_help, tab_attr, tab_dash, tab_net = st.tabs(
    ["ℹ️ Instructions", "🧮 Attribution", "📊 Dashboard", "🕸️ Network"]
)

with tab_help:
    intro_tab.render(BRAND=BRAND)

with tab_attr:
    attribution_tab.render(con=con, COLUMNS=COLUMNS, filters=filters)

with tab_dash:
    dashboard_tab.render(con=con, COLUMNS=COLUMNS, filters=filters, BRAND=BRAND)

with tab_net:
    network_tab.render(BRAND=BRAND)