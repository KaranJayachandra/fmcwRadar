from test_config import RADAR
from numpy import log, abs
from numpy.fft import fftshift, fft

def simpleSpectrum(input):
    return 20 * log(abs(fftshift(fft(input, \
        n=RADAR["Time Samples in Chirp"]))))