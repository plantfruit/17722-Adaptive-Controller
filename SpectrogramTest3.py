import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
import queue
import threading

# === Settings ===
fs = 48000  # Sample rate
frame_size = 2048  # Frame size (like minbuffersize)
channels = 1
dtype = 'int16'

# === Queue for audio frames ===
q = queue.Queue()

# === Audio callback ===
def audio_callback(indata, frames, time, status):
    if status:
        print(status)
    q.put(indata[:, 0].copy())  # Flatten 1 channel

# === Set up Stream ===
stream = sd.InputStream(
    samplerate=fs,
    channels=channels,
    dtype=dtype,
    blocksize=frame_size,
    callback=audio_callback
)

# === Plotting Setup ===
fig, ax = plt.subplots()
x = np.fft.rfftfreq(frame_size, 1/fs)
line, = ax.plot(x, np.zeros_like(x))
ax.set_ylim(0, 5000)  # adjust as needed
ax.set_xlim(0, fs / 2)
ax.set_title("Real-Time FFT")
ax.set_xlabel("Frequency (Hz)")
ax.set_ylabel("Amplitude")

# === Plot Update Loop ===
def update_plot():
    with stream:
        while plt.fignum_exists(fig.number):
            try:
                frame = q.get(timeout=1)
                fft_result = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))
                line.set_ydata(fft_result)
                fig.canvas.draw()
                fig.canvas.flush_events()
            except queue.Empty:
                continue

# === Run it ===
plt.ion()
threading.Thread(target=update_plot, daemon=True).start()
plt.show()
