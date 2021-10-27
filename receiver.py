# FMCW Receiver Module
# This is made of the mixer and the signal processing to extract information

from scipy.constants import c
import matplotlib.pyplot as plot
from numpy import multiply, linspace, conj
from numpy.fft import fft, fftshift
from test_config import RADAR, ENVIRONMENT
from common import simpleSpectrum
from transmitter import chirpGenerator, sequenceGenerator
from environment import radarChannel

def signalMixer(signal1, signal2):
    """
    This function mixes two signals in by multiplying them
    :param signal1: numpy.array
    :param signal2: numpy.array
    :return numpy.array
    """
    return multiply(signal1, signal2)

def test_signalMixer():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])

    # Generating the frequency axis for plotting
    frequency = linspace(- 0.5 / time[1], 0.5 / time[1], \
        RADAR["Time Samples in Chirp"])

    # Creating the range axis to check for the target
    range = linspace(-time[1] * c / 2, \
        time[1] * c / 2, RADAR["Time Samples in Chirp"])

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Creating the targets and the reflections
    receiveSequence = radarChannel(RADAR, ENVIRONMENT, transmitSequence)

    # Mixing the signals to get the beat signal
    beatSignal = signalMixer(transmitSequence, receiveSequence)

    # Calculating the transmit frequency spectrum
    transmitSpectrum = simpleSpectrum(transmitSequence)

    # Calculating the receive frequency spectrum
    receiveSpectrum = simpleSpectrum(receiveSequence)

    # Calculating the transmit frequency spectrum
    beatSpectrum = simpleSpectrum(beatSignal)

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Mixer Output (Two Targets at 1000m and 1500m)"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(221)
    timePlot.plot(time, beatSignal.real)
    timePlot.title.set_text('Mixer: Time Domain')
    timePlot.grid()

    frequencyPlot = plot.subplot(222)
    frequencyPlot.plot(range, abs(beatSpectrum))
    frequencyPlot.title.set_text('Mixer: Frequency Domain')
    frequencyPlot.grid()

    transmitPlot = plot.subplot(223)
    transmitPlot.plot(frequency / 1e6, abs(transmitSpectrum))
    transmitPlot.title.set_text('Transmit: Frequency Domain')
    transmitPlot.grid()

    receivePlot = plot.subplot(224)
    receivePlot.plot(frequency / 1e6, abs(receiveSpectrum))
    receivePlot.title.set_text('Receive: Frequency Domain')
    receivePlot.grid()

    plot.show()

if __name__ == '__main__':
    test_signalMixer()