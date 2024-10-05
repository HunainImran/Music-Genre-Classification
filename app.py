import streamlit as st
import numpy as np
import tensorflow.keras as keras
import librosa
from predict import preprocess_audio


genre_mapping = {
    0: "blues",
    1: "classical",
    2: "country",
    3: "disco",
    4: "hiphop",
    5: "jazz",
    6: "metal",
    7: "pop",
    8: "reggae",
    9: "rock"
}

# Load the trained model
model = keras.models.load_model("model.h5")

# Define function to make predictions
def predict_genre(audio_path):
    # Preprocess the audio file
    # You can reuse the preprocess_audio function from your predict.py file
    mfcc_features = preprocess_audio(audio_path, num_segments=10)
    mfcc_features = np.array(mfcc_features)
    mfcc_features = mfcc_features.reshape(mfcc_features.shape[0], -1, mfcc_features.shape[2])
    
    # Make predictions using the loaded model
    predictions = model.predict(mfcc_features)
    
    # Get the predicted genre labels
    predicted_genre_index = np.argmax(predictions, axis=1)
    # You may need to map the index to the actual genre label based on your dataset
    genre_counts = np.bincount(predicted_genre_index)

    # Get the index of the genre with the highest count
    final_genre_index = np.argmax(genre_counts)
    
    return final_genre_index

# Streamlit UI
def main():
    st.title("Music Genre Prediction")

    # File upload widget
    uploaded_file = st.file_uploader("Choose an audio file", type=["wav"])

    if uploaded_file is not None:
        st.audio(uploaded_file, format='audio/wav')

        # Predict genre on button click
        if st.button("Predict Genre"):
            predicted_genre = predict_genre(uploaded_file)
            st.write(f"Predicted Genre: {genre_mapping[predicted_genre]}")

# Run the Streamlit app
if __name__ == "__main__":
    main()
