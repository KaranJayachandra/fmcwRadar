from numpy import tile, max, linspace, abs, pad, zeros, copy, multiply, roll, exp
from math import floor, pi
from numpy.random import uniform
from numpy.fft import fft
from scipy.signal import chirp, spectrogram
from scipy.constants import c
import matplotlib.pyplot as plot
from scipy.signal.filter_design import normalize
from transmitter import chirpGenerator

def sequenceGenerator(radar, chirpSignal):
    # Returns the same input chirp signal repeated multiples times
    return tile(chirpSignal, radar["Number of Chirps"])

def mixSingal(trasmitSequence, receiveSequence):
    return multiply(trasmitSequence, receiveSequence)

def rangeDopplerMap(radar, mixSequence):
    radarFrame = copy(mixSequence).reshape((radar["Time Samples in Chirp"], \
        radar["Number of Chirps"]))
    rangeDopplerMap = fft(radarFrame, axis=0)
    rangeDopplerMap = fft(rangeDopplerMap, axis=1)
    return roll(rangeDopplerMap, floor(radar["Number of Chirps"] / 2))

# Configuration variables
radar = {
    "Chirp Bandwidth" : 1e6,
    "Chirp Time" : 25.6e-6,
    "Time Samples in Chirp" : 256,
    "Number of Chirps" : 256
}

environment = {
    "Total Targets": 2,
    "Target 1" : 1000,
    "Target 2" : 3000
}

# Main code for the creation
transmitChirp = chirpGenerator(radar, False)
transmitSequence = sequenceGenerator(radar, transmitChirp.real)
receiveSequence = radarChannel(radar, environment, transmitSequence)
mixerOutput = mixSingal(transmitSequence, receiveSequence)
rdMap = rangeDopplerMap(radar, mixerOutput)

# Plotting the results below
fig = plot.figure()
title = "Radar Processing Chain"
fig.suptitle(title, fontsize=20, weight=50)

transmitPlot = plot.subplot(221)
transmitPlot.plot(transmitChirp.real)
transmitPlot.title.set_text('Transmit Chirp')
transmitPlot.grid()

receivePlot = plot.subplot(222)
receivePlot.plot(receiveSequence[0:radar["Time Samples in Chirp"]])
receivePlot.title.set_text('Received Chirp')
receivePlot.grid('both')

mixPlot = plot.subplot(223)
mixPlot.plot(mixerOutput[0:256])
mixPlot.title.set_text('Mixer Output')
mixPlot.grid('both')

rdPlot = plot.subplot(224)
rdPlot.plot(abs(roll(fft(mixerOutput[0:radar["Time Samples in Chirp"]]), 128)))
rdPlot.title.set_text('Range FFT')
rdPlot.grid('both')
# rdPlot.imshow(abs(rdMap), cmap='jet', vmin=0, vmax=1000)

plot.show()