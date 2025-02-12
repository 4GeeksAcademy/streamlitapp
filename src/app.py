from pickle import load
import streamlit as st

# Load the trained model
model = load(open("/workspaces/streamlitapp/src/model.sav", "rb"))

# Updated class dictionary for target labels
class_dict = {
    "0": "Benign",
    "1": "Malignant"
}

st.title("Breast Cancer Prediction")

# Sliders for feature inputs
val1 = st.slider("Mean Radius", min_value=0.0, max_value=50.0, step=0.1)
val2 = st.slider("Mean Texture", min_value=0.0, max_value=50.0, step=0.1)
val3 = st.slider("Mean Perimeter", min_value=0.0, max_value=250.0, step=0.1)
val4 = st.slider("Mean Area", min_value=0.0, max_value=3000.0, step=1.0)
val5 = st.slider("Mean Smoothness", min_value=0.0, max_value=1.0, step=0.01)
val6 = st.slider("Mean Compactness", min_value=0.0, max_value=1.0, step=0.01)

# Prediction button
if st.button("Predict"):
    prediction = str(model.predict([[val1, val2, val3, val4, val5, val6]])[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
