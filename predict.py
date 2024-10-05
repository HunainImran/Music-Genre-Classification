import numpy as np
import tensorflow.keras as keras
import os
import math
import librosa

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

SAMPLE_RATE = 22050
TRACK_DURATION = 30  # measured in seconds
SAMPLES_PER_TRACK = SAMPLE_RATE * TRACK_DURATION

def preprocess_audio(audio_path, num_mfcc=13, n_fft=2048, hop_length=512, num_segments=5):
    """Extracts MFCCs from an audio file.

    :audio_path (str): Path to the audio file
    :num_mfcc (int): Number of coefficients to extract
    :n_fft (int): Interval we consider to apply FFT. Measured in # of samples
    :hop_length (int): Sliding window for FFT. Measured in # of samples
    :num_segments (int): Number of segments we want to divide the audio track into
    :return mfcc_features (list): List of MFCC features for each segment
    """

    # Initialize an empty list to store MFCC features
    mfcc_features = []

    # Load audio file
    signal, sample_rate = librosa.load(audio_path, sr=SAMPLE_RATE)

    # Calculate the number of samples per segment
    samples_per_segment = int(SAMPLES_PER_TRACK / num_segments)
    # Calculate the number of MFCC vectors per segment
    num_mfcc_vectors_per_segment = math.ceil(samples_per_segment / hop_length)

    # Process all segments of the audio file
    for d in range(num_segments):
        # Calculate start and finish sample for the current segment
        start = samples_per_segment * d
        finish = start + samples_per_segment

        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=signal[start:finish], sr=sample_rate,
                                    n_mfcc=num_mfcc, n_fft=n_fft, hop_length=hop_length)
        mfcc = mfcc.T

        # Store only MFCC features with the expected number of vectors
        if len(mfcc) == num_mfcc_vectors_per_segment:
            mfcc_features.append(mfcc.tolist())

    return mfcc_features

# Load the saved model
model = keras.models.load_model("model.h5")

# Example usage to predict genres for new data
new_audio_path = "Data/genres_original/metal/metal.00038.wav"
mfcc_features = preprocess_audio(new_audio_path, num_segments=10)
mfcc_features = np.array(mfcc_features)  # Convert to numpy array
mfcc_features = mfcc_features.reshape(mfcc_features.shape[0], -1, mfcc_features.shape[2])  # Reshape for model input
predictions = model.predict(mfcc_features)

# Get the predicted genre (based on the maximum probability)
predicted_genre_index = np.argmax(predictions, axis=1)
genre_counts = np.bincount(predicted_genre_index)

# Get the index of the genre with the highest count
final_genre_index = np.argmax(genre_counts)
print("Predicted genre index for each segment :", predicted_genre_index)
print("Predicted genre:", genre_mapping[final_genre_index])

__all__ = ['preprocess_audio']
