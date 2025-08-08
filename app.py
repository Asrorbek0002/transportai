import streamlit as st
from fastai.vision.all import *
import  plotly.express as px

# to give title
st.title("This model classifies Transports")

# to load modelpip install streamlit
model = load_learner('transport_model.pt')

# to load file
uploaded_file = st.file_uploader("Upload an image", type=['jpg', 'png', 'jpeg'])

# to check image existance
if uploaded_file:
    try:
        img = PILImage.create(uploaded_file)
        pred, pred_idx, probs = model.predict(img)

        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)
        st.success(f"Prediction: {pred}")
        st.info(f"Confidence: {probs[pred_idx] * 100:.1f}")

        # Plotting
        fig = px.bar(x=probs * 100, y=model.dls.vocab)
        st.plotly_chart(fig)

    except Exception as e:
        st.error(f"Error during prediction: {e}")
else:
    st.warning("Please upload an image first!")












