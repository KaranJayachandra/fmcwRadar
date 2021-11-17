# oooo    oooo                                          
# `888   .8P'                                          Karan Jayachandra
#  888  d8'     .oooo.   oooo d8b  .oooo.   ooo. .oo.  mail@karanjayachandra.com
#  88888[      `P  )88b  `888""8P `P  )88b  `888P"Y88b karanjayachandra.com
#  888`88b.     .oP"888   888      .oP"888   888   888 
#  888  `88b.  d8(  888   888     d8(  888   888   888 
# o888o  o888o `Y888""8o d888b    `Y888""8o o888o o888o 

# FMCW Environment Module
# It is made of one class: RadarTarget() and one function: radarChannel()
# They come with their own test/debug functions that help you visualize the
# waveforms returned and their frequency spectrums.

from math import pi, sin, radians
from scipy.constants import c
import matplotlib.pyplot as plot
from numpy import linspace, copy, pad, zeros, abs, exp, multiply
from common import phaseSpectrum, powerSpectrum, addNoise
from test_config import RADAR, ENVIRONMENT
from transmitter import chirpGenerator, sequenceGenerator

class RadarTarget():
    """
    RadarTarget class creates a target for the FMCW radar system.
    """

    # The class is initialized with the range of the target
    # TODO: Add the target velocity as a parameters as well
    def __init__(self, range, velocity):
        """
        RadarTarget object contructor that requires the range of the target as 
        input
        :param range: integer, must be positive
        """
        # Range and Doppler of the target based on initialization
        self.range = range
        self.velocity = velocity
        # Calculate the delay based on the target distance
        self.delay = (self.range * 2) / c
        # Adding a constant so that the distant targets are still seen
        powerConstant = 10
        # Calculate the attentuation due to propagation
        self.attenuation = powerConstant / (self.range ** 4)
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
        chirpBlock = copy(chirpSequence.astype(complex)).reshape(( \
            radar["Number of Chirps"], radar["Time Samples in Chirp"]))
        # Processing chirp by chirp
        for iSlow in range(radar["Number of Chirps"]):
            chirpSignal = chirpBlock[iSlow, :]
            # Pad the zeros and return the chirp signal back after cutting
            delayChirp = pad(chirpSignal, (closestIndex, 0))
            delayChirp = delayChirp[0:chirpSignal.size]
            # Multiple with the constant phase denoting doppler
            phase = 4 * pi * (iSlow * self.velocity * radar["Chirp Time"]) \
                / radar["Carrier Wavelength"]
            phase = exp(1j * phase)
            # Multiply with the phase and return
            chirpBlock[iSlow, :] = multiply(phase, delayChirp)
        
        # Add noise to the returned signal
        chirpBlock = addNoise(radar, chirpBlock)

        # Return the sequence back to the receiver
        return self.attenuation * chirpBlock.flatten()
        # return chirpBlock.flatten()

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

    # Using test_config to access target distance
    targetDistance = ENVIRONMENT["Target 1"][0]
    targetVelocity = ENVIRONMENT["Target 1"][1]

    # Create a radarTarget at range 100 m
    target = RadarTarget(targetDistance, targetVelocity)

    # Transmit the chirp sequence against the RadarTarget
    receiveSequence = target.reflect(RADAR, transmitSequence)

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp (Single Target at" + str(targetDistance) + "m)"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, receiveSequence)
    timePlot.title.set_text('Time Domain')
    timePlot.grid()

    frequencyPlot = plot.subplot(223)
    frequencyPlot.plot(frequency / 1e6, powerSpectrum(receiveSequence))
    frequencyPlot.title.set_text('Frequency Domain: Amplitude')
    frequencyPlot.grid()

    anglePlot = plot.subplot(224)
    anglePlot.plot(frequency / 1e6, phaseSpectrum(receiveSequence))
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
    receivedSequence = zeros((radar["Array Size"], chirpSequence.size), \
        dtype=complex)

    # Loop through the channels
    for iChannel in range(radar["Array Size"]):
        # Creating the targets based on the class radarTarget
        for iTarget in range(environment["Total Targets"]):
            # Storing the target properties for easy access
            targetDistance = environment["Target " + str(iTarget + 1)][0]
            targetVelocity = environment["Target " + str(iTarget + 1)][1]
            targetAngle = environment["Target " + str(iTarget + 1)][2]
            # Calculating the phase difference due to array geometry
            arrayPhase = 2 * pi * radar["Array Spacing"] * iChannel * \
                sin(radians(targetAngle))
            arrayFactor = exp(1j * arrayPhase)
            # Calculate the target reflection and multiply array phase factor
            target = RadarTarget(targetDistance, targetVelocity)
            targetResponse = arrayFactor * target.reflect(radar, chirpSequence)
            # Store in the received sequence array
            receivedSequence[iChannel, :] = receivedSequence[iChannel, :] + \
                targetResponse
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

    # Processing just one channel for testing
    channelSequence = receiveSequence[0, :]

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp (Targets at: " + str(ENVIRONMENT["Target 1"][0]) + \
        "m and " + str(ENVIRONMENT["Target 2"][0]) + "m)"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, abs(channelSequence.real))
    timePlot.title.set_text('Time Domain')
    timePlot.grid()

    frequencyPlot = plot.subplot(223)
    frequencyPlot.plot(frequency / 1e6, powerSpectrum(channelSequence))
    frequencyPlot.title.set_text('Frequency Domain: Amplitude')
    frequencyPlot.grid()

    anglePlot = plot.subplot(224)
    anglePlot.plot(frequency / 1e6, phaseSpectrum(channelSequence))
    anglePlot.title.set_text('Frequency Domain: Phase')
    anglePlot.grid()

    plot.show()

# Run this file to test the functions by examining the time and frequency 
# domain representations of the received chirp sequence
if __name__ == '__main__':
    test_radarTarget()
    test_radarChannel()