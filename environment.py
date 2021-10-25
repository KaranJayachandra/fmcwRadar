from numpy import linspace, copy, pad, zeros, abs, angle
from scipy.constants import c
from math import pi
from numpy.fft import fft, fftshift
from transmitter import chirpGenerator, sequenceGenerator
import matplotlib.pyplot as plot
from test_config import RADAR

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
        self.attenuation = 1 / self.range ** 4
    # The reflection currently only contains range information
    def reflect(self, radar, chirpSequence):
        """
        Thus function takes as input an FMCW chirp sequence and returns the 
        sequence with a delay corresponding to the target range
        :param RADAR: dict
        :param chirpSequence: numpy.array
        """
        # Calculate the delay based on the target distance
        timeDelay = (self.range * 2) / c
        # Find how many zeros need to be padded in the start
        time = linspace(0, radar["Chirp Time"], radar["Time Samples in Chirp"])
        closestIndex = (abs(time - timeDelay)).argmin()
        # Seperating the chirps for individual processing
        chirpBlock = copy(chirpSequence).reshape(( \
            radar["Time Samples in Chirp"], radar["Number of Chirps"]))
        # Processing chirp by chirp
        for iSlow in range(radar["Number of Chirps"]):
            chirpSignal = chirpBlock[iSlow, :]
            # Pad the zeros and return the chirp signal back after cutting
            delayChirp = pad(chirpSignal, closestIndex)
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
    target = RadarTarget(1500)

    # Transmit the chirp sequence against the RadarTarget
    receiveSequence = target.reflect(RADAR, transmitSequence)

    # Calculating the frequency spectrum
    frequencySpectrum = fftshift(fft(receiveSequence, \
        n=RADAR["Time Samples in Chirp"]))

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, receiveSequence)
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

def radarChannel(radar, environment, chirpSequence):
    # Creating an empty array where the return sequence is stored
    returnSequence = zeros(chirpSequence.size)
    # Creating the targets based on the class radarTarget
    targets = []
    for iTarget in range(environment["Total Targets"]):
        # targetDistance = 3000 * uniform(0, 1)
        targetDistance = environment["Target " + str(iTarget + 1)]
        print(targetDistance)
        targets.append(RadarTarget(targetDistance))
        returnSequence += targets[iTarget].reflect(radar, chirpSequence)
    # Return back the sequence to the Receiver
    return returnSequence

# Run this file to test the functions by examining the time and frequency 
# domain representations of the received chirp sequence
if __name__ == '__main__':
    test_radarTarget()