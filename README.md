# Music Genre Classification Using Audio Data 🎶

This project focuses on building a machine learning model to classify music genres based on audio features extracted from tracks. The project is built using TensorFlow and Keras, with preprocessing done through `librosa` to extract MFCC features from audio files.

## Features
- Extracts MFCC features from audio files
- Trains a deep neural network (DNN) model on the extracted features
- Provides a simple UI built with Streamlit for genre classification of uploaded audio files
- Displays performance metrics such as confusion matrix and accuracy graph


## Installation & Setup 🛠️

1. Clone the repository:
    ```bash
    git clone https://github.com/your-username/genre-classification.git
    cd genre-classification
    ```

2. Ensure you have the `librosa` library installed for audio processing:
    ```bash
    pip install librosa
    ```

## Dataset
The dataset used in this project is based on the [GTZAN Genre Collection](http://marsyas.info/downloads/datasets.html), which contains 10 genres with 100 audio files each. The dataset has been preprocessed into segments of MFCCs for model training.

## How It Works 🚀

### 1. Preprocessing
- **MFCC Extraction**: The audio files are split into segments, and MFCC features are extracted. These features are used as input for the model.
- **Segmenting**: Audio tracks are divided into multiple segments to provide more data for the model and ensure each part of the track is utilized.

### 2. Model Training
- **Architecture**: A deep neural network with multiple dense layers, dropout for regularization, and softmax activation at the output layer.
- **Loss Function**: The model uses `sparse_categorical_crossentropy` to handle multi-class classification.
- **Optimizer**: The `Adam` optimizer is used for its adaptive learning rate properties.
- **Activation Function**: `ReLU` is used in hidden layers to introduce non-linearity, and `softmax` is applied to the output for multi-class classification.
  
### 3. Prediction
- The model predicts genres based on new audio files uploaded through the Streamlit UI.
- It outputs the predicted genre and displays performance metrics like accuracy and confusion matrix.

### 4. Streamlit UI
- **Upload**: The user can upload a music file (preferably 30 seconds long).
- **Prediction**: The model predicts the genre and displays it.
- **Metrics**: The app shows a confusion matrix and accuracy plot for the model's performance.

## Usage 🚀

To run the Streamlit app:
```bash
streamlit run app.py
```

This will open a local server where you can upload an audio file and get genre predictions.

## Model Performance

- **Training Accuracy**: 85% 
- **Test Accuracy**: 70-80%
- **Precision, Recall, F1-Score**: Shown in the app.

  
## Future Work

- Expanding the dataset to include more genres and samples.
- Further tuning hyperparameters and model architecture for better performance.

---