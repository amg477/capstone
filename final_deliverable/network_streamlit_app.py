import streamlit as st
import pandas as pd
from content_network import render_content_network


st.set_page_config(page_title="Publisher ↔ Term Content Network", layout="wide")

st.title("Publisher ↔ Term Content Network")
st.caption("Build a filtered, community-colored network from your sampled dataset and curated term lists.")

# ---- Inputs ----
st.sidebar.header("Data inputs")
uploaded_df = st.sidebar.file_uploader("final_model_dataset CSV (sampled)", type=["csv"])
uploaded_attr = st.sidebar.file_uploader("attribution_all_scored CSV", type=["csv"])

wl1 = st.sidebar.file_uploader("Whitelist: top 300 unigrams", type=["csv"])
wl2 = st.sidebar.file_uploader("Whitelist: top 1,000 bigrams", type=["csv"])
wl3 = st.sidebar.file_uploader("Whitelist: top 1,000 keywords", type=["csv"])
wl4 = st.sidebar.file_uploader("Whitelist: top 1,000 words", type=["csv"])

st.sidebar.header("Parameters")
publisher_col = st.sidebar.selectbox("Publisher column", ["publisher_name", "publication_name"], index=0)
min_term_weight = st.sidebar.number_input("Min global term weight", min_value=0.0, value=0.0, step=0.001, format="%.3f")
top_publishers = st.sidebar.slider("Top publishers", 10, 80, 30, step=5)
top_terms = st.sidebar.slider("Top terms", 10, 80, 30, step=5)
edge_q = st.sidebar.slider("Edge percentile cutoff", 0.5, 0.95, 0.75, step=0.05)
labels_per_type = st.sidebar.slider("Labels per type", 4, 20, 10, step=1)
use_max_credit = st.sidebar.checkbox("Prefer max_term_credit for edge weights", value=True)

generic_default = "online,video,link,ago,daily,big,act,force,alert,job,news"
generic_terms = st.sidebar.text_input("Generic terms to drop (comma-separated)", generic_default)

run = st.sidebar.button("Build network")

# ---- Run ----
if run:
    if not (uploaded_df and uploaded_attr and (wl1 or wl2 or wl3 or wl4)):
        st.warning("Please upload df, attr_df, and at least one whitelist CSV.")
        st.stop()

    df = pd.read_csv(uploaded_df)
    attr_df = pd.read_csv(uploaded_attr)

    # combine whitelist columns (first column only)
    wls = []
    for up in [wl1, wl2, wl3, wl4]:
        if up is None: continue
        try:
            s = pd.read_csv(up, header=None).iloc[:,0]
        except Exception:
            s = pd.read_csv(up).iloc[:,0]
        wls.append(s.dropna().astype(str).str.strip())
    whitelist = pd.concat(wls, ignore_index=True).dropna().astype(str).str.strip().tolist()

    fig, edges_filt, comm_tbl = render_content_network(
        df=df,
        attr_df=attr_df,
        whitelist_terms=whitelist,
        publisher_col=publisher_col,
        min_term_weight=min_term_weight,
        top_publishers=top_publishers,
        top_terms=top_terms,
        generic_terms=[t.strip().lower() for t in generic_terms.split(",") if t.strip()],
        edge_percentile_cutoff=edge_q,
        labels_per_type=labels_per_type,
        figsize=(12,8),
        title="Content Network: Publishers ↔ High-Impact Terms",
        use_max_term_credit_first=use_max_credit,
    )

    st.subheader("Network")
    st.pyplot(fig, clear_figure=True)

    st.subheader("Community Summary")
    st.dataframe(comm_tbl)

    st.subheader("Edges used to draw (filtered)")
    st.dataframe(edges_filt.sort_values("weight", ascending=False).reset_index(drop=True))
else:
    st.info("Upload files and set parameters in the sidebar, then click **Build network**.")
    

