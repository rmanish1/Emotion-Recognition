import numpy as np
import librosa
from tensorflow.keras.models import load_model
from audio_recorder_streamlit import audio_recorder
import io

# Load model
model = load_model("speech_model\\new_model.h5")

# Constants
CAT7 = ['fear', 'disgust', 'neutral', 'happy', 'sad', 'surprise', 'angry']

# Function to predict emotions
def speech_predict_emotion(audio_bytes):
    try:
        # Convert audio bytes to numpy array
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)

        # Extract MFCC features
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)

        # Pad or truncate mfccs to match the model input shape
        if mfccs.shape[1] < model.input_shape[-1]:
            mfccs = np.pad(mfccs, ((0, 0), (0, model.input_shape[-1] - mfccs.shape[1])), mode='constant')
        elif mfccs.shape[1] > model.input_shape[-1]:
            mfccs = mfccs[:, :model.input_shape[-1]]

        # Reshape for model input
        mfccs = mfccs[np.newaxis, :, :, np.newaxis]

        # Predict emotions
        predictions = model.predict(mfccs)[0]

        # Format predictions
        emotion_predictions = {emotion: proba for emotion, proba in zip(CAT7, predictions)}

        return emotion_predictions
    except Exception as e:
        return {"error": str(e)}

