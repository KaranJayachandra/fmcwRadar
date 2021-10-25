# FMCW Transmitter Module
# It is made of two functions: chirpGenerator() and sequenceGenerator()
# They come with their own test/debug functions that help you visualize the
# waveforms generated and their frequency spectrums.

from math import pi
import matplotlib.pyplot as plot
from numpy import tile, linspace, abs, exp, real, angle
from numpy.fft import fft, fftshift
from test_config import RADAR

def chirpGenerator(RADAR, log):
    """ 
    This function generates an LFM chirp based on the give RADAR configuration 
    file. 
    :param RADAR: dict
    :param log: boolean
    :return: numpy.array
    """

    # Calculate the sampling frequency required
    maxSamplingTime = 1 / (2 * RADAR["Chirp Bandwidth"])
    # Create the time axis for the calculation of the signal
    time = linspace(0, RADAR["Chirp Time"], RADAR["Time Samples in Chirp"])
    # Check if nyguist criteria is met using the given time samples
    if time[1] > maxSamplingTime:
        raise ValueError("For the given chirp the smallest sampling time is " \
            + maxSamplingTime + ". Please configure the Chirp correctly")
    # Print the log if requested
    elif (log):
        print('Nyguist Sample time: {:.2e}'.format(maxSamplingTime))
        print('Radar Bandwidth: {:.2e}'.format(RADAR["Chirp Bandwidth"]))
        print('Current Sample Time: {:.2e}'.format(time[1]))
    # Calculate the instantaneous chirp frequency
    frequency = (time * RADAR["Chirp Bandwidth"]) / (2 * RADAR["Chirp Time"])
    # Creating the complex signal which can then be transformed
    chirpSignal = exp(-1j * 2 * pi * frequency * time)
    return chirpSignal

def test_chirpGenerator():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"], RADAR["Time Samples in Chirp"])
    # Generating the frequency axis for plotting
    frequency = linspace(- 0.5 / time[1], 0.5 / time[1], time.size)

    # Generate the signal from the chirpGenerator function
    transmitChirp = real(chirpGenerator(RADAR, True))
    
    # Calculating the frequency spectrum
    frequencySpectrum = fftshift(fft(transmitChirp))

    # Plotting the results
    fig = plot.figure()
    title = "Transmit Chirp"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, transmitChirp)
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

def sequenceGenerator(radar, chirpSignal, log):
    """ 
    This function repeats in the input chirpbased on the give RADAR 
    configuration file to create a sequence. 
    :param  RADAR: dict
    :param  chirpSignal: numpy.array
    :param  log: boolean
    :return: numpy.array
    """

    # Returns the same input chirp signal repeated multiples times
    if log:
        print('Creating a chirp sequence of length: {}\n'\
            .format(radar["Number of Chirps"]))
    return tile(chirpSignal, radar["Number of Chirps"])

def test_sequenceGenerator():
    # Generate the time axis for plotting the signal
    time = linspace(0, RADAR["Chirp Time"] * RADAR["Number of Chirps"], \
        RADAR["Time Samples in Chirp"] * RADAR["Number of Chirps"])
    # Generating the frequency axis for plotting
    frequency = linspace(- 0.5 / time[1], 0.5 / time[1], \
        RADAR["Time Samples in Chirp"])

    # Generate the signal from the chirpGenerator function
    transmitChirp = real(chirpGenerator(RADAR, True))
    
    # Generate the chirp sequence from the sequenceGenerator
    transmitSequence = sequenceGenerator(RADAR, transmitChirp, True)

    # Calculating the frequency spectrum
    frequencySpectrum = fftshift(fft(transmitSequence, \
        n=RADAR["Time Samples in Chirp"]))

    # Plotting the results
    fig = plot.figure()
    title = "Transmit Sequence"
    fig.suptitle(title, fontsize=20, weight=50)

    timePlot = plot.subplot(211)
    timePlot.plot(time, transmitSequence)
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
# domain representations of the chirp and the chirp sequence
if __name__ == '__main__':
    test_chirpGenerator()
    test_sequenceGenerator()