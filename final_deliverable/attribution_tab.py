# attribution_tab.py — Attribution lookups

from __future__ import annotations
import streamlit as st
import pandas as pd
from structure import explain_attribution, quote_ident

@st.cache_data
def _get_distinct_values(con, table, column, where_clause="", params=None):
    """Helper function to get distinct values from a table with caching."""
    try:
        query = f"SELECT DISTINCT {column} FROM {table}"
        if where_clause:
            query += f" WHERE {where_clause}"
        query += " ORDER BY 1"
        
        df = con.execute(query, params or {}).fetchdf()
        return df[column].astype(str).tolist()
    except Exception:
        return []

@st.cache_data
def _get_item_attribution_data(con, sel_dim, sel_val):
    """Get item attribution score and peers data with caching."""
    try:
        # Get attribution score
        score_df = con.execute(
            """
            SELECT dimension, value, credit, credit_share, rating
            FROM v_item_attr
            WHERE dimension=$d AND value=$v
            ORDER BY credit_share DESC
            LIMIT 1
            """,
            {"d": sel_dim, "v": sel_val},
        ).fetchdf()
        
        # Get peers for ranking
        peers_df = con.execute(
            "SELECT dimension, value, credit FROM v_item_attr WHERE dimension=$d", {"d": sel_dim}
        ).fetchdf()
        
        return score_df, peers_df
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

def _render_attribution_explanation(score_df, peers_df, dimension="dimension"):
    """Render attribution explanation if data is available."""
    if not score_df.empty:
        st.markdown("#### What this means")
        row = score_df.iloc[0].copy()
        if dimension != "dimension":
            row["dimension"] = dimension
        st.info(explain_attribution(row, peers_df))

def _render_matching_rows(con, COLUMNS, sel_dim, sel_val, where_sql, args, row_limit):
    """Render matching rows from the main dataset."""
    if sel_dim in COLUMNS:
        id_col = quote_ident(sel_dim)
        rows = con.execute(
            f"SELECT * FROM v WHERE {where_sql} AND {id_col}=$v LIMIT $lim",
            {**args, "v": sel_val, "lim": int(row_limit)},
        ).fetchdf()
        st.dataframe(rows, use_container_width=True)
    else:
        st.info(f"'{sel_dim}' isn't a column in `v`, so matching rows can't be shown.")

def render(con, COLUMNS, filters):
    """Main render function with optimized structure."""
    st.subheader("Attribution Lookups")
    lookup_type = st.radio("Lookup type", ["Item", "Term"], horizontal=True)
    
    where_sql, args, row_limit = filters["where_sql"], filters["args"], filters["row_limit"]

    if lookup_type == "Item":
        _render_item_attribution(con, COLUMNS, where_sql, args, row_limit)
    else:
        _render_term_attribution(con, where_sql, args, row_limit)

def _render_item_attribution(con, COLUMNS, where_sql, args, row_limit):
    """Handle item attribution lookups."""
    dims = _get_distinct_values(con, "v_item_attr", "dimension", "dimension IS NOT NULL")
    
    if not dims:
        st.info("No item attribution available.")
        return

    # Select dimension with smart default
    default_idx = dims.index("publication_name") if "publication_name" in dims else 0
    sel_dim = st.selectbox("Dimension", dims, index=default_idx)

    # Get values for selected dimension
    values = _get_distinct_values(con, "v_item_attr", "value", "dimension=$d AND value IS NOT NULL", {"d": sel_dim})
    
    if not values:
        st.info("No values for this dimension.")
        return

    sel_val = st.selectbox("Value", values)

    # Get attribution score and peers
    score_df, peers_df = _get_item_attribution_data(con, sel_dim, sel_val)
    st.dataframe(score_df, use_container_width=True)
    
    _render_attribution_explanation(score_df, peers_df)
    _render_matching_rows(con, COLUMNS, sel_dim, sel_val, where_sql, args, row_limit)

@st.cache_data
def _get_term_attribution_data(con, term):
    """Get term attribution score and peers data with caching."""
    try:
        # Get term attribution score
        tscore = con.execute(
            "SELECT value, credit, credit_share, rating FROM v_term_attr WHERE value=$v ORDER BY credit_share DESC",
            {"v": term},
        ).fetchdf()
        
        # Get peers for ranking
        peers = con.execute("SELECT 'term' AS dimension, value, credit FROM v_term_attr").fetchdf()
        
        return tscore, peers
    except Exception:
        return pd.DataFrame(), pd.DataFrame()

@st.cache_data
def _search_text_matches(con, where_sql, args, term, row_limit):
    """Search for text matches with caching."""
    try:
        text_expr = "LOWER(COALESCE(processed_headline,'') || ' ' || COALESCE(processed_body,''))"
        hits = con.execute(
            f"SELECT * FROM v WHERE {where_sql} AND {text_expr} LIKE $pat LIMIT $lim",
            {**args, "pat": f"%{term.lower()}%", "lim": int(row_limit)},
        ).fetchdf()
        return hits
    except Exception:
        return pd.DataFrame()

def _render_term_attribution(con, where_sql, args, row_limit):
    """Handle term attribution lookups."""
    term = st.text_input("Type a term to search")
    if not term:
        return

    # Get term attribution data
    tscore, peers = _get_term_attribution_data(con, term)
    st.dataframe(tscore, use_container_width=True)
    _render_attribution_explanation(tscore, peers, "term")

    # Search for matching rows in main dataset
    hits = _search_text_matches(con, where_sql, args, term, row_limit)
    st.dataframe(hits, use_container_width=True)