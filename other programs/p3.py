import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import librosa
import tensorflow as tf

import  sounddevice as sd
import io
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.preprocessing import image


import sounddevice as sd
import wave
import numpy as np

model = tf.keras.models.load_model(r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\my_model.h5")

#model.summary()


##############################################################################
def create_pngs_from_wavs(input_path, output_path):
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    dir = os.listdir(input_path)

    for i, file in enumerate(dir):
        input_file = os.path.join(input_path, file)
        output_file = os.path.join(output_path, file.replace('.wav', '.png'))
        create_spectrogram(input_file, output_file)


def create_spectrogram(audio_file, image_file):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)

    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    librosa.display.specshow(log_ms, sr=sr)

    fig.savefig(image_file)
    plt.close(fig)

def recording():
    sample_rate = 44100  # 44.1 kHz
    duration = 10  # 10 seconds
    filename = r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\sounds\test_input.wav"

    # Record audio
    print("Recording...")
    audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
    sd.wait()  # Wait for the recording to finish
    print("Recording complete.")

    # Save as WAV file
    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)  # Mono
        wf.setsampwidth(4)  # 16-bit audio (2 bytes per sample)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"Saved as {filename}")
#___________________________________________________________________________________________________________________________________________
for i in  range(3):
    #recording()
    print("Creating spectograms")
    create_pngs_from_wavs(r'C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\sounds',r'C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\spectograms')
    print("Spectograms created")

    #loadinng the images

    # x = image.load_img(r'C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\spectograms\background_00.png', target_size=(224, 224))
    # plt.xticks([])
    # plt.yticks([])
    # plt.imshow(x)
    #---------------------------------------------------------------------------------------------------------------------------------------
    # img_path = r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\spectograms"
    img = image.load_img(r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\spectograms\test_input.png", target_size=(224, 224,3))  # Resize to match model input size
    x = image.img_to_array(img)  # Convert to NumPy array
    x = np.expand_dims(x, axis=0)  # Add batch dimension
    x = x / 255.0  # Normalize (same as training)

    if(x.any()):
        print("image loaded succesfully")
        #print(x)

    ###predictions

    predictions = model.predict(x)

    # Define class labels (adjust according to your dataset)
    class_labels = ["baground", "Chainsaw", "engine", "strom"]

    # Print predictions
    for i, label in enumerate(class_labels):
        print(f"{label}: {predictions[0][i]:.4f}")

    # Get the predicted class
    predicted_class = class_labels[np.argmax(predictions)]
    print(f"Predicted Class: {predicted_class}")


    #-----------------------------------------------------------------------------
    #clearing space
    print("free up folder")

    # os.remove(r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\sounds\test_input.wav")
    # os.remove(r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\spectograms\test_input.png")

    #----------------------------------------------------------------------------------------