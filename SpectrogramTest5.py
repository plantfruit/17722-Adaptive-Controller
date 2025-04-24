import numpy as np
import pyaudio
import matplotlib.pyplot as plt
import time

# Audio parameters
CHUNK = 1024 
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100

# Initialize PyAudio
p = pyaudio.PyAudio()

# Open audio stream
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

# Prepare plot
plt.ion() # Enable interactive mode
fig, ax = plt.subplots()
line, = ax.plot(np.fft.fftfreq(CHUNK, 1/RATE), np.abs(np.fft.fft(np.zeros(CHUNK))))
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Magnitude")
ax.set_title("Real-time FFT of Microphone Input")
ax.set_xlim(0, RATE/2) # Display only positive frequencies

try:
    while True:
        # Read audio data
        data = stream.read(CHUNK)
        audio_data = np.frombuffer(data, dtype=np.int16)

        # Calculate FFT
        fft_data = np.fft.fft(audio_data)
        fft_magnitude = np.abs(fft_data)

        # Update plot data
        line.set_ydata(fft_magnitude)

        # Redraw plot
        fig.canvas.draw()
        fig.canvas.flush_events()

        time.sleep(0.01) # Add a small delay to control the update rate

except KeyboardInterrupt:
    print("Stopped by user")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
    plt.ioff()
    plt.show()
