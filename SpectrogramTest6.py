import pyaudio
import numpy as np

CHUNK = 1024
FORMAT = pyaudio.paFloat32
CHANNELS = 1
RATE = 44100

def audio_callback(in_data, frame_count, time_info, status):
    audio_data = np.frombuffer(in_data, dtype=np.float32)
    processed_data = audio_data * 0.5  # Simple amplitude reduction
    return (processed_data.tobytes(), pyaudio.paContinue)

p = pyaudio.PyAudio()
stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                frames_per_buffer=CHUNK,
                stream_callback=audio_callback)

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

class AudioVisualizer:
    def __init__(self):
        self.fig, self.ax = plt.subplots()
        self.line, = self.ax.plot([], [])
        self.ax.set_xlim(0, CHUNK)
        self.ax.set_ylim(-1, 1)

    def update(self, frame):
        audio_data = np.frombuffer(stream.read(CHUNK), dtype=np.float32)
        self.line.set_data(range(len(audio_data)), audio_data)
        return self.line,

    def animate(self):
        ani = FuncAnimation(self.fig, self.update, interval=20)
        plt.show()

av = AudioVisualizer()
