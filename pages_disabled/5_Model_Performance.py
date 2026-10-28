from pathlib import Path
import json
import streamlit as st
ROOT = Path(__file__).resolve().parents[1]
st.title("Model Performance")
for filename, title in [("classification_metrics.json", "Classification"), ("regression_metrics.json", "Regression")]:
    path = ROOT / "models" / filename
    st.subheader(title)
    if path.exists():
        st.json(json.loads(path.read_text(encoding="utf-8")))
    else:
        st.info("Run `python scripts/train_models.py` to create model metrics.")
