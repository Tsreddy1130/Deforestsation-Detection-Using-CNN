import sounddevice as sd
import wave
import numpy as np




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
        wf.setsampwidth(2)  # 16-bit audio (2 bytes per sample)
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"Saved as {filename}")
recording()