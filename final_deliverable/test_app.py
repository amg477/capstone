import streamlit as st

st.set_page_config(page_title="PolicyPath Test", layout="wide")

st.title("🏛️ PolicyPath")
st.markdown("Loading test...")

st.success("✅ App is working!")

if st.button("Test Button"):
    st.write("Button clicked successfully!")

st.info("This is a minimal test to verify Streamlit deployment works.")
