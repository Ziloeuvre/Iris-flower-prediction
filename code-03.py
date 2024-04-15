import pandas as pd
import streamlit as st

data = pd.read_csv("data/BRENT_data.csv")

st.title("Time Series Data")

st.line_chart(data.set_index("date"))
st.write("Testing")
# streamlit run code-03.py[ARGUMENTS]
