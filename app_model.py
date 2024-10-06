import streamlit as st
import tensorflow as tf
from inference import predict_video
import tempfile
import os

@st.cache(allow_output_mutation=True)
def load_model():
    return tf.keras.models.load_model('models/best_model.h5')

def main():
    st.title("Deepfake Detection App")
    
    model = load_model()
    
    uploaded_file = st.file_uploader("Choose a video file", type=["mp4", "avi", "mov"])
    
    if uploaded_file is not None:
        # Save the uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False) 
        tfile.write(uploaded_file.read())
        
        # Predict
        prediction = predict_video(model, tfile.name)
        
        # Remove temporary file
        tfile.close()
        os.unlink(tfile.name)
        
        # Display result
        st.write(f"Probability of being a deepfake: {prediction:.2f}")
        
        if prediction > 0.5:
            st.warning("This video is likely a deepfake.")
        else:
            st.success("This video is likely genuine.")

if __name__ == "__main__":
    main()