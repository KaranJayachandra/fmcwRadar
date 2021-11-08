# FMCW Receiver Module
# This is made of the mixer and the signal processing to extract information

from scipy.constants import c
import matplotlib.pyplot as plot
from matplotlib import cm
from numpy import multiply, linspace, copy, log, meshgrid, transpose
from numpy.fft import fft, fftshift
from test_config import RADAR, ENVIRONMENT
from common import powerSpectrum, addNoise
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
    rMax = (RADAR["Chirp Time"] * c) / (4 * RADAR["Chirp Bandwidth"] * time[1])

    range = linspace(-rMax, rMax, RADAR["Time Samples in Chirp"])

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Creating the targets and the reflections
    receiveSequence = radarChannel(RADAR, ENVIRONMENT, transmitSequence)

    # Mixing the signals to get the beat signal
    beatSignal = signalMixer(transmitSequence, receiveSequence)

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp (Targets at: " + str(ENVIRONMENT["Target 1"]) + \
        "m and " + str(ENVIRONMENT["Target 2"]) + "m)"
    fig.suptitle(title, fontsize=20, weight=50)

    frequencyPlot = plot.subplot(211)
    frequencyPlot.plot(range, powerSpectrum(beatSignal))
    frequencyPlot.title.set_text('Mixer: Frequency Domain')
    frequencyPlot.grid()

    transmitPlot = plot.subplot(223)
    transmitPlot.plot(frequency / 1e6, powerSpectrum(transmitSequence))
    transmitPlot.title.set_text('Transmit: Frequency Domain')
    transmitPlot.grid()

    receivePlot = plot.subplot(224)
    receivePlot.plot(frequency / 1e6, powerSpectrum(receiveSequence))
    receivePlot.title.set_text('Receive: Frequency Domain')
    receivePlot.grid()

    plot.show()

def rangeDopplerProcessing(radar, mixerOutput):
    # Reshape the output for FFT processing
    radarCube = copy(mixerOutput).reshape(( \
            radar["Number of Chirps"], radar["Time Samples in Chirp"]))

    # Calculate the FFT for range
    radarCube = fft(radarCube, axis=0)
    radarCube = fft(radarCube, axis=1)
    radarCube = fftshift(radarCube)

    # Return the processed cube
    return radarCube

def test_rangeDopplerProcessing():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])

    # Generating the frequency axis for plotting
    frequency = linspace(- 0.5 / time[1], 0.5 / time[1], \
        RADAR["Time Samples in Chirp"])

    # Creating the range axis to check for the target
    rMax = (RADAR["Chirp Time"] * c) / (4 * RADAR["Chirp Bandwidth"] * time[1])

    range = linspace(0, rMax, int(RADAR["Time Samples in Chirp"] / 2))
    velocity = linspace(1, 256, 256)

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Creating the targets and the reflections
    receiveSequence = radarChannel(RADAR, ENVIRONMENT, transmitSequence)

    # Adding noise
    receiveSequence = addNoise(RADAR, receiveSequence)

    # Mixing the signals to get the beat signal
    beatSignal = signalMixer(transmitSequence, receiveSequence)

    # Calculate the Radar Cube
    radarCube = rangeDopplerProcessing(RADAR, beatSignal)

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Range Doppler Map (Targets at: " + str(ENVIRONMENT["Target 1"]) + \
        "m and " + str(ENVIRONMENT["Target 2"]) + "m)"
    fig.suptitle(title, fontsize=20, weight=50)

    # Neglecting the negative range data
    visualData = transpose(abs(radarCube[:, 0:256]))
    # visualData = 20*log(visualData + 1)

    # Plotting the Range Doppler Map
    plot.imshow(visualData, extent=[-128, 128, 0, rMax])
    plot.show()

if __name__ == '__main__':
    test_signalMixer()
    test_rangeDopplerProcessing()