# -*- coding: utf-8 -*-
import sys
import pyaudio
import numpy as np
from scipy.fft import fft
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore, QtWidgets

fftLen = 2048
shift = 100
signal_scale = 1. / 2000

capture_setting = {
    "ch": 1,
    "fs": 16000,
    "chunk": shift
}

def spectrumAnalyzer():
    global fftLen, capture_setting, signal_scale

    ch = capture_setting["ch"]
    fs = capture_setting["fs"]
    chunk = capture_setting["chunk"]

    p = pyaudio.PyAudio()
    inStream = p.open(format=pyaudio.paInt16,
                      channels=ch,
                      rate=fs,
                      input=True,
                      frames_per_buffer=chunk)

    signal = np.zeros(fftLen, dtype=float)

    app = QtWidgets.QApplication([])
    app.quitOnLastWindowClosed()
    mainWindow = QtWidgets.QMainWindow()
    mainWindow.setWindowTitle("Spectrum Analyzer")
    mainWindow.resize(800, 300)
    centralWid = QtWidgets.QWidget()
    mainWindow.setCentralWidget(centralWid)
    lay = QtWidgets.QVBoxLayout()
    centralWid.setLayout(lay)

    specWid = pg.PlotWidget(name="spectrum")
    specItem = specWid.getPlotItem()
    specItem.setMouseEnabled(y=False)
    specItem.setYRange(0, 1000)
    specItem.setXRange(0, fftLen / 2, padding=0)
    specAxis = specItem.getAxis("bottom")
    specAxis.setLabel("Frequency [Hz]")
    specAxis.setScale(fs / 2. / (fftLen / 2 + 1))
    hz_interval = 500
    newXAxis = (np.arange(int(fs / 2 / hz_interval)) + 1) * hz_interval
    oriXAxis = newXAxis / (fs / 2. / (fftLen / 2 + 1))
    specAxis.setTicks([zip(oriXAxis, newXAxis)])
    lay.addWidget(specWid)

    mainWindow.show()

    while True:
        data = inStream.read(chunk, exception_on_overflow = False)
        num_data = np.frombuffer(data, dtype="int16")
        signal = np.roll(signal, -chunk)
        signal[-chunk:] = num_data
        fftspec = fft(signal)
        specItem.plot(abs(fftspec[1:fftLen // 2 + 1] * signal_scale), clear=True)
        QtWidgets.QApplication.processEvents()

if __name__ == "__main__":
    spectrumAnalyzer()

fftLen = 2048
shift = 100
signal_scale = 1. / 2000

capture_setting = { 
    "ch": 1,
    "fs": 16000,
    "chunk": shift
}

def spectrumAnalyzer():
    global fftLen, capture_setting, signal_scale

    ch = capture_setting["ch"]
    fs = capture_setting["fs"]
    chunk = capture_setting["chunk"]

    p = pyaudio.PyAudio()
    inStream = p.open(format=pyaudio.paInt16,
                      channels=ch,
                      rate=fs,
                      input=True,
                      frames_per_buffer=chunk)

    signal = np.zeros(fftLen, dtype=float)

    app = QtWidgets.QApplication([])
    app.quitOnLastWindowClosed()
    mainWindow = QtGui.QMainWindow()
    mainWindow.setWindowTitle("Spectrum Analyzer")
    mainWindow.resize(800, 300)
    centralWid = QtWidgets.QWidget()
    mainWindow.setCentralWidget(centralWid)
    lay = QtGui.QVBoxLayout()
    centralWid.setLayout(lay)

    specWid = pg.PlotWidget(name="spectrum")
    specItem = specWid.getPlotItem()
    specItem.setMouseEnabled(y=False)
    specItem.setYRange(0, 1000)
    specItem.setXRange(0, fftLen / 2, padding=0)
    specAxis = specItem.getAxis("bottom")
    specAxis.setLabel("Frequency [Hz]")
    specAxis.setScale(fs / 2. / (fftLen / 2 + 1))
    hz_interval = 500
    newXAxis = (np.arange(int(fs / 2 / hz_interval)) + 1) * hz_interval
    oriXAxis = newXAxis / (fs / 2. / (fftLen / 2 + 1))
    specAxis.setTicks([zip(oriXAxis, newXAxis)])
    lay.addWidget(specWid)

    mainWindow.show()

    while True:
        data = inStream.read(chunk, exception_on_overflow = False)
        num_data = np.frombuffer(data, dtype="int16")
        signal = np.roll(signal, -chunk)
        signal[-chunk:] = num_data
        fftspec = fft(signal)
        specItem.plot(abs(fftspec[1:fftLen // 2 + 1] * signal_scale), clear=True)
        QtGui.QApplication.processEvents()

if __name__ == "__main__":
    spectrumAnalyzer()

fftLen = 2048
shift = 100
signal_scale = 1. / 2000

capture_setting = {
    "ch": 1,
    "fs": 16000,
    "chunk": shift
}

def spectrumAnalyzer():
    global fftLen, capture_setting, signal_scale

    ch = capture_setting["ch"]
    fs = capture_setting["fs"]
    chunk = capture_setting["chunk"]

    p = pyaudio.PyAudio()
    inStream = p.open(format=pyaudio.paInt16,
                      channels=ch,
                      rate=fs,
                      input=True,
                      frames_per_buffer=chunk)

    signal = np.zeros(fftLen, dtype=float)

    app = QtWidgets.QApplication # -*- coding: utf-8 -*-
import sys
import pyaudio
import numpy as np
from scipy.fft import fft
import pyqtgraph as pg
from pyqtgraph.Qt import QtGui, QtCore

fftLen = 2048
shift = 100
signal_scale = 1. / 2000

capture_setting = {
    "ch": 1,
    "fs": 16000,
    "chunk": shift
}

def spectrumAnalyzer():
    global fftLen, capture_setting, signal_scale

    ch = capture_setting["ch"]
    fs = capture_setting["fs"]
    chunk = capture_setting["chunk"]

    p = pyaudio.PyAudio()
    inStream = p.open(format=pyaudio.paInt16,
                      channels=ch,
                      rate=fs,
                      input=True,
                      frames_per_buffer=chunk)

    signal = np.zeros(fftLen, dtype=float)

    app = QtGui.QApplication([])
    app.quitOnLastWindowClosed()
    mainWindow = QtWidgets.QMainWindow()
    mainWindow.setWindowTitle("Spectrum Analyzer")
    mainWindow.resize(800, 300)
    centralWid = QtWidgets.QWidget()
    mainWindow.setCentralWidget(centralWid)
    lay = QtGui.QVBoxLayout()
    centralWid.setLayout(lay)

    specWid = pg.PlotWidget(name="spectrum")
    specItem = specWid.getPlotItem()
    specItem.setMouseEnabled(y=False)
    specItem.setYRange(0, 1000)
    specItem.setXRange(0, fftLen / 2, padding=0)
    specAxis = specItem.getAxis("bottom")
    specAxis.setLabel("Frequency [Hz]")
    specAxis.setScale(fs / 2. / (fftLen / 2 + 1))
    hz_interval = 500
    newXAxis = (np.arange(int(fs / 2 / hz_interval)) + 1) * hz_interval
    oriXAxis = newXAxis / (fs / 2. / (fftLen / 2 + 1))
    specAxis.setTicks([zip(oriXAxis, newXAxis)])
    lay.addWidget(specWid)

    mainWindow.show()

    while True:
        data = inStream.read(chunk, exception_on_overflow = False)
        num_data = np.frombuffer(data, dtype="int16")
        signal = np.roll(signal, -chunk)
        signal[-chunk:] = num_data
        fftspec = fft(signal)
        specItem.plot(abs(fftspec[1:fftLen // 2 + 1] * signal_scale), clear=True)
        QtGui.QApplication.processEvents()

if __name__ == "__main__":
    spectrumAnalyzer().QApplication([])
    app.quitOnLastWindowClosed()
    mainWindow = QtWidgets.QMainWindow()
    mainWindow.setWindowTitle("Spectrum Analyzer")
    mainWindow.resize(800, 300)
    centralWid = QtGui.QWidget()
    mainWindow.setCentralWidget(centralWid)
    lay = QtGui.QVBoxLayout()
    centralWid.setLayout(lay)

    specWid = pg.PlotWidget(name="spectrum")
    specItem = specWid.getPlotItem()
    specItem.setMouseEnabled(y=False)
    specItem.setYRange(0, 1000)
    specItem.setXRange(0, fftLen / 2, padding=0)
    specAxis = specItem.getAxis("bottom")
    specAxis.setLabel("Frequency [Hz]")
    specAxis.setScale(fs / 2. / (fftLen / 2 + 1))
    hz_interval = 500
    newXAxis = (np.arange(int(fs / 2 / hz_interval)) + 1) * hz_interval
    oriXAxis = newXAxis / (fs / 2. / (fftLen / 2 + 1))
    specAxis.setTicks([zip(oriXAxis, newXAxis)])
    lay.addWidget(specWid)

    mainWindow.show()

    while True:
        data = inStream.read(chunk, exception_on_overflow = False)
        num_data = np.frombuffer(data, dtype="int16")
        signal = np.roll(signal, -chunk)
        signal[-chunk:] = num_data
        fftspec = fft(signal)
        specItem.plot(abs(fftspec[1:fftLen // 2 + 1] * signal_scale), clear=True)
        QtGui.QApplication.processEvents()

if __name__ == "__main__":
    spectrumAnalyzer()
