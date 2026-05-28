import joblib
import numpy as np
import pandas as pd
import streamlit as st
from matminer.featurizers.composition import ElementProperty
from matminer.featurizers.conversions import StrToComposition
from pymatgen.core import Composition

st.set_page_config(page_title="Cathode Voltage Screener", page_icon="🔋")

@st.cache_resource
def load_bundle():
    return joblib.load("models/screening_model.joblib")

@st.cache_resource
def load_featurizer():
    return ElementProperty.from_preset("magpie")

bundle = load_bundle()
model = bundle["model"]
feat_cols = bundle["feat_cols"]
halfwidth = bundle["conformal_halfwidth"]
coverage = bundle["coverage"]
ep = load_featurizer()

st.title("🔋 Li-Cathode Voltage Screener")
st.markdown(
    "Estimates the **average insertion voltage** of a Li cathode from its "
    "**framework composition** (the host structure without Li), using a "
    "gradient-boosting model trained on Materials Project data."
)

# --- honest limitations, stated up front ---
with st.expander("⚠️ What this tool can and cannot do — read first"):
    st.markdown(
        f"""
- This is a **screening aid**, not a precise predictor. Typical error is ~0.5 V.
- The **±{halfwidth:.2f} V** interval is a *conformal* {int(coverage*100)}% prediction
  interval, verified to contain the true value ~{int(coverage*100)}% of the time on
  held-out data.
- The interval is the **same width for every prediction**. The model's per-prediction
  uncertainty was tested and found *not* to track actual error, so it can't honestly
  flag which specific predictions are more or less reliable.
- It uses **composition only** — it ignores crystal structure, so polymorphs with the
  same formula get the same prediction.
- Best used to **rank/filter candidates coarsely**, e.g. flagging likely high-voltage
  chemistries, not for fine ranking.
        """
    )

formula = st.text_input(
    "Framework formula (host without Li)",
    value="CoO2",
    help="e.g. CoO2, FePO4, Mn2O4, NiO2",
)

if st.button("Predict voltage"):
    try:
        comp = Composition(formula)   # validates the formula
        row = pd.DataFrame({"framework_formula": [formula]})
        row = StrToComposition().featurize_dataframe(row, "framework_formula")
        row = ep.featurize_dataframe(row, "composition")
        X = row[feat_cols].values
        pred = float(model.predict(X)[0])

        low, high = pred - halfwidth, pred + halfwidth
        st.metric("Predicted average voltage", f"{pred:.2f} V")
        st.write(
            f"**{int(coverage*100)}% prediction interval:** "
            f"{max(low, 0):.2f} – {high:.2f} V"
        )
        if pred > 4.0:
            st.success("High-voltage candidate (point estimate > 4 V).")
        elif pred < 2.0:
            st.info("Low-voltage candidate (point estimate < 2 V).")
    except Exception as e:
        st.error(f"Could not parse or featurize '{formula}'. Error: {e}")

st.caption(
    "Trained on Materials Project insertion-electrode data. "
    "Model: LightGBM on Magpie composition descriptors. "
    "Uncertainty: split-conformal calibration. For research/demo use only."
)