"""Streamlit demo: predict engine condition from sensor readings.

Run from the repository root:

    streamlit run app/app.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import pandas as pd
import streamlit as st

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "model.pkl"

# (label, description, min, max): slider ranges come from the training-data statistics.
# The key order must match the column order the model was trained on.
FEATURES = {
    "Engine rpm": ("Engine RPM", "Revolutions per minute of the engine.", 61.0, 2239.0),
    "Lub oil pressure": ("Lub Oil Pressure", "Pressure of the lubricating oil.", 0.003384, 7.265566),
    "Fuel pressure": ("Fuel Pressure", "Pressure of the fuel.", 0.003187, 21.138326),
    "Coolant pressure": ("Coolant Pressure", "Pressure of the coolant.", 0.002483, 7.478505),
    "lub oil temp": ("Lub Oil Temperature", "Temperature of the lubricating oil.", 71.321974, 89.580796),
    "Coolant temp": ("Coolant Temperature", "Temperature of the coolant.", 61.673325, 195.527912),
    "Temperature_difference": (
        "Temperature Difference",
        "Coolant temperature minus lubricating-oil temperature.",
        -22.669427,
        119.008526,
    ),
}


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


def default_value(low: float, high: float) -> float:
    return (low + high) / 2


def reset_values() -> None:
    for name, (_, _, low, high) in FEATURES.items():
        st.session_state[name] = default_value(low, high)


def predict_condition(model, values: dict[str, float]) -> tuple[int, float]:
    """Return the predicted class and the model's probability for that class."""
    row = pd.DataFrame([values], columns=list(FEATURES))
    proba = model.predict_proba(row)[0]
    predicted = int(proba.argmax())
    return predicted, float(proba[predicted])


def main() -> None:
    st.set_page_config(page_title="Engine Condition Prediction", page_icon="🔧")
    st.title("Engine Condition Prediction")
    st.caption(
        "Adjust the sensor readings and let the Gradient Boosting model classify the engine. "
        "Educational demo, not a diagnostic tool."
    )

    try:
        model = load_model()
    except Exception as exc:  # incompatible scikit-learn version, missing file, ...
        st.error(
            "Could not load `models/model.pkl`. It was saved with scikit-learn 1.4.2, so install "
            "the pinned versions in `requirements.txt`, or regenerate the model with "
            "`python src/train_model.py`."
        )
        st.exception(exc)
        st.stop()

    st.sidebar.title("Feature Descriptions")
    for label, description, _, _ in FEATURES.values():
        st.sidebar.markdown(f"**{label}:** {description}")

    values = {}
    for name, (label, _, low, high) in FEATURES.items():
        if name not in st.session_state:
            st.session_state[name] = default_value(low, high)
        values[name] = st.slider(label, min_value=low, max_value=high, key=name)

    predict_col, reset_col, _ = st.columns([2, 2, 5])
    predict_clicked = predict_col.button("Predict Engine Condition", type="primary")
    reset_col.button("Reset Values", on_click=reset_values)

    if predict_clicked:
        result, probability = predict_condition(model, values)
        if result == 0:
            st.success(f"The engine is predicted to be in **normal condition** (model probability: {probability:.1%}).")
        else:
            st.warning(f"**Warning:** please investigate further (model probability: {probability:.1%}).")


if __name__ == "__main__":
    main()
