import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
DATA = Path(__file__).resolve().parents[1] / "data" / "cleaned_housing_data.csv"
st.title("Property Analysis")
if DATA.exists():
    df = pd.read_csv(DATA)
    state = st.selectbox("State", ["All"] + sorted(df["State"].astype(str).unique().tolist()))
    view = df if state == "All" else df[df["State"].astype(str) == state]
    st.metric("Properties", f"{len(view):,}")
    st.plotly_chart(px.box(view.sample(min(5000, len(view)), random_state=42), x="Property_Type", y="Price_in_Lakhs", title="Price by Property Type"), use_container_width=True)
    st.plotly_chart(px.scatter(view.sample(min(5000, len(view)), random_state=42), x="Size_in_SqFt", y="Price_in_Lakhs", color="Property_Type", title="Size vs Price"), use_container_width=True)
    st.dataframe(view.head(500), use_container_width=True)
else:
    st.error("Run scripts/data_cleaning.py first.")
