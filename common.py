from test_config import RADAR
from math import pi
from scipy.constants import k
from numpy import log, abs, angle
from numpy.fft import fftshift, fft
from numpy.random import normal


def powerSpectrum(input):
    return 20 * log(abs(fftshift(fft(input, \
        n=RADAR["Time Samples in Chirp"]))))

def phaseSpectrum(input):
    return (180 / pi) * angle(fftshift(fft(input, \
        n=RADAR["Time Samples in Chirp"])))

def addNoise(RADAR, data):
    bandwidth = RADAR["Time Samples in Chirp"] / RADAR["Chirp Time"]
    variance = k * RADAR["Operating Temperature"] * bandwidth * RADAR["Noise Figure"]
    return data + normal(loc=0.0, scale=variance, size=data.shape)