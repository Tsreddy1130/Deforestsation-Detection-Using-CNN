import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import sounddevice as sd
import matplotlib.pyplot as plt
import io
from PIL import Image


# Load the trained model
model = tf.keras.models.load_model("my_model.h5")

# Audio settings
SAMPLERATE = 22050  # Adjust to match training data
DURATION = 10  # 10 seconds
N_MELS = 128  # Adjust to match training spectrograms

# Function to record audio
def record_audio(duration=DURATION, samplerate=SAMPLERATE):
    print("Recording...")
    audio = sd.rec(int(duration * samplerate), samplerate=samplerate, channels=1, dtype='float32')
    sd.wait()
    print("Recording complete.")
    return audio.flatten()

# Function to generate spectrogram
def generate_spectrogram(audio, samplerate=SAMPLERATE, n_mels=N_MELS):
    spectrogram = librosa.feature.melspectrogram(y=audio, sr=samplerate, n_mels=n_mels)
    spectrogram_db = librosa.power_to_db(spectrogram, ref=np.max)
    
    # Convert spectrogram to image-like format
    fig, ax = plt.subplots()
    ax.set_axis_off()
    librosa.display.specshow(spectrogram_db, sr=samplerate, cmap='gray_r')
    
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert('L')
    img = img.resize((128, 128))  # Adjust to match input shape
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=[0, -1])  # Add batch and channel dimensions
    
    return img_array

# Continuous classification loop
def classify_audio():
    while True:
        audio = record_audio()
        spectrogram = generate_spectrogram(audio)
        prediction = model.predict(spectrogram)
        predicted_class = np.argmax(prediction)
        
        print(f"Predicted Class: {predicted_class}")

if __name__ == "__main__":
    classify_audio()
