# FMCW Environment Module
# It is made of one class: RadarTarget() and one function: radarChannel()
# They come with their own test/debug functions that help you visualize the
# waveforms returned and their frequency spectrums.

from math import pi
from scipy.constants import c
import matplotlib.pyplot as plot
from numpy import linspace, copy, pad, zeros, abs, angle
from numpy.fft import fft, fftshift
from test_config import RADAR, ENVIRONMENT
from transmitter import chirpGenerator, sequenceGenerator

class RadarTarget():
    """
    RadarTarget class creates a target for the FMCW radar system.
    """

    # The class is initialized with the range of the target
    # TODO: Add the target velocity as a parameters as well
    def __init__(self, range):
        """
        RadarTarget object contructor that requires the range of the target as 
        input
        :param range: integer, must be positive
        """
        self.range = range
        # Calculate the delay based on the target distance
        self.delay = (self.range * 2) / c
        # Calculate the attentuation due to propagation
        self.attenuation = 1 / self.range ** 4
    # The reflection currently only contains range information
    def reflect(self, radar, chirpSequence):
        """
        Thus function takes as input an FMCW chirp sequence and returns the 
        sequence with a delay corresponding to the target range
        :param radar: dict
        :param chirpSequence: numpy.array
        """
        # Find how many zeros need to be padded in the start
        time = linspace(0, radar["Chirp Time"], radar["Time Samples in Chirp"])
        closestIndex = (abs(time - self.delay)).argmin()
        # Seperating the chirps for individual processing
        chirpBlock = copy(chirpSequence).reshape(( \
            radar["Time Samples in Chirp"], radar["Number of Chirps"]))
        # Processing chirp by chirp
        for iSlow in range(radar["Number of Chirps"]):
            chirpSignal = chirpBlock[iSlow, :]
            # Pad the zeros and return the chirp signal back after cutting
            delayChirp = pad(chirpSignal, (closestIndex, 0))
            chirpBlock[iSlow, :] = delayChirp[0:chirpSignal.size]
        # Return the sequence back to the receiver
        return self.attenuation * chirpBlock.flatten()

def test_radarTarget():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])
    # Generating the frequency axis for plotting
    frequency = linspace(- 0.5 / time[1], 0.5 / time[1], \
        RADAR["Time Samples in Chirp"])

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Create a radarTarget at range 1500 m
    target = RadarTarget(2000)

    # Transmit the chirp sequence against the RadarTarget
    receiveSequence = target.reflect(RADAR, transmitSequence)

    # Calculating the frequency spectrum
    frequencySpectrum = fftshift(fft(receiveSequence, \
        n=RADAR["Time Samples in Chirp"]))

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp (Single Target at 2000m)"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, abs(receiveSequence))
    timePlot.title.set_text('Time Domain')
    timePlot.grid()

    frequencyPlot = plot.subplot(223)
    frequencyPlot.plot(frequency / 1e6, abs(frequencySpectrum))
    frequencyPlot.title.set_text('Frequency Domain: Amplitude')
    frequencyPlot.grid()

    anglePlot = plot.subplot(224)
    anglePlot.plot(frequency / 1e6, (180 / pi) * angle(frequencySpectrum))
    anglePlot.title.set_text('Frequency Domain: Phase')
    anglePlot.grid()

    plot.show()

def radarChannel(radar, environment, chirpSequence):
    """
    This function creates a list of targets based on the environment and the 
    radar sensor and reflects the chirp sequence
    :param radar: dict
    :param environment: dict
    :param chirpSequence: numpy.array
    :return numpy.array
    """

    # Creating an empty array where the return sequence is stored
    receivedSequence = zeros(chirpSequence.size)
    # Creating the targets based on the class radarTarget
    for iTarget in range(environment["Total Targets"]):
        # targetDistance = 3000 * uniform(0, 1)
        targetDistance = environment["Target " + str(iTarget + 1)]
        target = RadarTarget(targetDistance)
        targetResponse = target.reflect(radar, chirpSequence)
        receivedSequence = receivedSequence + targetResponse
    # Return back the sequence to the Receiver
    return receivedSequence

def test_radarChannel():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])
    # Generating the frequency axis for plotting
    frequency = linspace(- 0.5 / time[1], 0.5 / time[1], \
        RADAR["Time Samples in Chirp"])

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Creating the targets and the reflections
    receiveSequence = radarChannel(RADAR, ENVIRONMENT, transmitSequence)

    # Calculating the frequency spectrum
    frequencySpectrum = fftshift(fft(receiveSequence, \
        n=RADAR["Time Samples in Chirp"]))

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp (Two Targets at 1000m and 1500m)"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, abs(receiveSequence))
    timePlot.title.set_text('Time Domain')
    timePlot.grid()

    frequencyPlot = plot.subplot(223)
    frequencyPlot.plot(frequency / 1e6, abs(frequencySpectrum))
    frequencyPlot.title.set_text('Frequency Domain: Amplitude')
    frequencyPlot.grid()

    anglePlot = plot.subplot(224)
    anglePlot.plot(frequency / 1e6, (180 / pi) * angle(frequencySpectrum))
    anglePlot.title.set_text('Frequency Domain: Phase')
    anglePlot.grid()

    plot.show()

# Run this file to test the functions by examining the time and frequency 
# domain representations of the received chirp sequence
if __name__ == '__main__':
    test_radarTarget()
    test_radarChannel()