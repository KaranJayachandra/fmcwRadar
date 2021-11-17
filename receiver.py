# oooo    oooo                                          
# `888   .8P'                                          Karan Jayachandra
#  888  d8'     .oooo.   oooo d8b  .oooo.   ooo. .oo.  mail@karanjayachandra.com
#  88888[      `P  )88b  `888""8P `P  )88b  `888P"Y88b karanjayachandra.com
#  888`88b.     .oP"888   888      .oP"888   888   888 
#  888  `88b.  d8(  888   888     d8(  888   888   888 
# o888o  o888o `Y888""8o d888b    `Y888""8o o888o o888o 

# FMCW Receiver Module
# This is made of the mixer and the signal processing to extract information

from scipy.constants import c
from math import pi, sin, radians
import matplotlib.pyplot as plot
from matplotlib import cm
from numpy import multiply, linspace, copy, zeros, transpose, mean, log
from numpy.fft import fft, fftshift
from test_config import RADAR, ENVIRONMENT
from common import powerSpectrum, findTargets, estimateAngle
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

    # Processing just one channel for testing
    channelSequence = transpose(receiveSequence[0, :])

    # Mixing the signals to get the beat signal
    beatSignal = signalMixer(transmitSequence, channelSequence)

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Receive Chirp (Targets at: " + str(ENVIRONMENT["Target 1"][0]) + \
        "m and " + str(ENVIRONMENT["Target 2"][0]) + "m)"
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
    receivePlot.plot(frequency / 1e6, powerSpectrum(channelSequence))
    receivePlot.title.set_text('Receive: Frequency Domain')
    receivePlot.grid()

    plot.show()

def rangeDopplerProcessing(radar, transmitSequence, receivedSequence):
    # Variable for output range doppler map
    coherentCube = zeros((radar["Array Size"], radar["Number of Chirps"], \
        radar["Time Samples in Chirp"]), dtype=complex)

    # Process the received signal per channel
    for iChannel in range(radar["Array Size"]):
        # Signal received at the current channel
        channelSequence = receivedSequence[iChannel, :]

        # Calculate the mixer output for the channel
        beatSignal = signalMixer(transmitSequence, channelSequence)

        # Reshape the output for FFT processing
        radarCube = copy(beatSignal).reshape(( \
                radar["Number of Chirps"], radar["Time Samples in Chirp"]))

        # Calculate the FFT for range
        radarCube = fft(radarCube, axis=0)
        radarCube = fft(radarCube, axis=1)
        radarCube = fftshift(radarCube)

        # Sum the rangle doppler maps
        coherentCube[iChannel, :, :] = radarCube

    # Return the processed cube
    return coherentCube

def test_rangeDopplerProcessing():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])

    # Creating the range axis to check for the target
    rMax = (RADAR["Chirp Time"] * c) / (4 * RADAR["Chirp Bandwidth"] * time[1])

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Creating the targets and the reflections
    receiveSequence = radarChannel(RADAR, ENVIRONMENT, transmitSequence)

    # Calculate the Radar Cube
    radarCube = rangeDopplerProcessing(RADAR, transmitSequence, receiveSequence)

    # Calculate the average across the receivers incoherently
    rangeDopplerMap = mean(abs(radarCube), axis=0)

    # Plotting the received signal to check for delay
    fig = plot.figure()
    title = "Range Doppler Map (Targets at: " + str(ENVIRONMENT["Target 1"][0]) + \
        "m and " + str(ENVIRONMENT["Target 2"][0]) + "m)"
    fig.suptitle(title, fontsize=20, weight=50)

    # Neglecting the negative range data
    visualData = 20 * log(transpose(abs(rangeDopplerMap[:, \
        0:int(RADAR["Time Samples in Chirp"]/2)])))

    # Plotting the Range Doppler Map
    plot.imshow(visualData, extent=[-128, 128, 0, rMax])
    plot.show()

def angleEstimation(radar, radarCube):
    angleData = zeros((radar["Number of Chirps"], radar["Time Samples in Chirp"]))
    
    for iDoppler in range(radar["Number of Chirps"]):
        for iRange in range(radar["Time Samples in Chirp"]):
            snapShot = radarCube[:, iDoppler, iRange]
            angleData[iDoppler, iRange] = estimateAngle(snapShot)
    return angleData

def test_angleEstimation():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])

    # Creating the range axis to check for the target
    rMax = (RADAR["Chirp Time"] * c) / (4 * RADAR["Chirp Bandwidth"] * time[1])

    # Generate a chirp signal
    chirpSignal = chirpGenerator(RADAR, False)

    # Generate Chirp Sequence
    transmitSequence = sequenceGenerator(RADAR, chirpSignal, False)

    # Creating the targets and the reflections
    receiveSequence = radarChannel(RADAR, ENVIRONMENT, transmitSequence)

    # Calculate the Radar Cube
    radarCube = rangeDopplerProcessing(RADAR, transmitSequence, receiveSequence)

    # Calculate the average across the receivers incoherently
    rangeDopplerMap = mean(abs(radarCube), axis=0)

    # Find the range doppler bins of the target
    targetIndices = findTargets(rangeDopplerMap, ENVIRONMENT["Total Targets"])

    # Extract the angle information based on array 
    targetAngles = angleEstimation(radarCube, targetIndices)

    # Print target details
    print(targetAngles)

if __name__ == '__main__':
    # test_signalMixer()
    # test_rangeDopplerProcessing()
    test_angleEstimation()