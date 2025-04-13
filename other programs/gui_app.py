import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import librosa
import librosa.display
import tensorflow as tf
import matplotlib.pyplot as plt
import sounddevice as sd
import wave
from tensorflow.keras.preprocessing import image
from tkinter import Tk, filedialog, Button, Label, Frame
from PIL import Image, ImageTk

# Load the pre-trained model
model = tf.keras.models.load_model("my_model.h5")
color="#eba134"
# Define class labels
class_labels = ["Background", "Chainsaw", "Engine", "Storm"]

def create_spectrogram(audio_file, image_file):
    y, sr = librosa.load(audio_file)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)
    
    plt.figure(figsize=(5, 5))
    librosa.display.specshow(log_ms, sr=sr)
    plt.axis('off')
    plt.savefig(image_file, bbox_inches='tight', pad_inches=0)
    plt.close()

def play_audio(file_path):
    y, sr = librosa.load(file_path, sr=None)
    sd.play(y, sr)

def stop_audio():
    sd.stop()

def predict_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio Files", "*.wav")])
    if not file_path:
        return
    
    spectrogram_path = "temp_spectrogram.png"
    create_spectrogram(file_path, spectrogram_path)
    
    img = image.load_img(spectrogram_path, target_size=(224, 224, 3))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0) / 255.0  # Normalize
    
    predictions = model.predict(x)
    predicted_class = class_labels[np.argmax(predictions)]
    result_label.config(text=f"Predicted Class: {predicted_class}")
    if predicted_class in ["Chainsaw", "Engine"]:
        color="#fa0213"
        l.config(text="THREAT DETECTED",fg=color)
    else:
        color="#1fb50e"
        l.config(text="NO THREAT DETECTED",fg=color)




    # Display the spectrogram
    img = Image.open(spectrogram_path)
    img = img.resize((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    
    # Enable playback and stop buttons
    play_button.config(state="normal", command=lambda: play_audio(file_path))
    stop_button.config(state="normal", command=stop_audio)

# GUI Setup
root = Tk()
root.title("Deforestsation Detection with Audio Classifier")
root.geometry("700x800")
root.configure(bg="#f4f4f4")
Label(root,text="Batch 16 EMBEDED PROTOTYPE ECE",font=('Times New Roman',25),fg='green').pack()
frame = Frame(root, bg="#d9d9d9", padx=20, pady=20)
frame.pack(pady=30, padx=10)

Button(frame, text="Select Audio File", command=predict_audio, bg="#ffcc00", fg="black", font=("Arial", 12, "bold")).pack(pady=10)
result_label = Label(frame, text="Prediction will appear here", bg="#d9d9d9", font=("Arial", 12))
result_label.pack()

image_label = Label(frame, bg="#d9d9d9")
image_label.pack()

play_button = Button(frame, text="Play Audio", state="disabled", bg="#4CAF50", fg="white", font=("Arial", 12, "bold"))
play_button.pack(pady=10)

stop_button = Button(frame, text="Stop Audio", state="disabled", bg="#d9534f", fg="white", font=("Arial", 12, "bold"))
stop_button.pack(pady=10)

l=Label(root,text="",fg=color,font=("Times New Roman",40))
l.pack()
K=Label(root,text="2200049154\n2200049135\n2200049120",fg=color,font=("Times New Roman",10)).pack()


root.mainloop()