import sounddevice as sd
import numpy as np
import wave
import os

# Folder to save recordings
save_folder = r"C:\Users\SESHIREDDY\Desktop\prototype_lab\project\test\sounds"
os.makedirs(save_folder, exist_ok=True)

def get_next_filename():
    """Finds the next available recording filename."""
    existing_files = [f for f in os.listdir(save_folder) if f.startswith("recording_") and f.endswith(".wav")]
    numbers = [int(f.split("_")[1].split(".")[0]) for f in existing_files if f.split("_")[1].split(".")[0].isdigit()]
    next_number = max(numbers) + 1 if numbers else 1
    return os.path.join(save_folder, f"recording_{next_number}.wav")

def record_audio_continuous(duration=10, sample_rate=16000):
    """Continuously records audio in 10s segments, saving sequentially."""
    try:
        print("Recording continuously... (Press Ctrl+C to stop)")
        while True:
            filename = get_next_filename()
            print(f"Recording -> {filename}")

            # Record in mono (1 channel)
            audio_data = sd.rec(int(sample_rate * duration), samplerate=sample_rate, channels=1, dtype=np.int16)
            sd.wait()  # Wait for recording to finish

            # Save the recording
            with wave.open(filename, "wb") as wf:
                wf.setnchannels(1)  # Mono
                wf.setsampwidth(2)  # 16-bit audio
                wf.setframerate(sample_rate)
                wf.writeframes(audio_data.tobytes())

            print(f"Saved: {filename}")

    except KeyboardInterrupt:
        print("\nRecording stopped.")

# Run the function
record_audio_continuous()
