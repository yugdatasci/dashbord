# ---------------------------
#   BASIC WORKING STREAMLIT APP
# ---------------------------

import streamlit as st     # MUST be the first Streamlit import

# Page config must be called before any other Streamlit command
st.set_page_config(
    page_title="Monetary Policy Dashboard",
    layout="wide"
)

# Title
st.title("Monetary Policy & Inflation Dashboard (Starter Version)")

# Description
st.write("""
This is a working Streamlit starter app.  
If you can see this page without errors, Streamlit is installed correctly  
and your file structure is correct.
""")

# Demo chart
import pandas as pd
import numpy as np

# Create mock data for a sample plot
data = pd.DataFrame({
    "Month": pd.date_range("2024-01-01", periods=12, freq="M"),
    "Inflation (%)": np.random.uniform(3, 8, 12)
})

st.line_chart(data.set_index("Month"))

st.success("Streamlit is working correctly! ✔ Now you can add your real dashboard code.")
