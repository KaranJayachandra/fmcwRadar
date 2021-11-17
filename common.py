# oooo    oooo                                          
# `888   .8P'                                          Karan Jayachandra
#  888  d8'     .oooo.   oooo d8b  .oooo.   ooo. .oo.  mail@karanjayachandra.com
#  88888[      `P  )88b  `888""8P `P  )88b  `888P"Y88b karanjayachandra.com
#  888`88b.     .oP"888   888      .oP"888   888   888 
#  888  `88b.  d8(  888   888     d8(  888   888   888 
# o888o  o888o `Y888""8o d888b    `Y888""8o o888o o888o 

# FMCW Common Module

from test_config import RADAR
from math import pi
from scipy.constants import k
from numpy import log, abs, angle, argmax, unravel_index
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

def findTargets(rangeDopplerMap, totalTargets):
    maxIndices = []
    for iTarget in range(totalTargets * 4):
        index = unravel_index(argmax(rangeDopplerMap), rangeDopplerMap.shape)
        maxIndices.append(index)
        print(index)
        rangeDopplerMap[index[0]-5:index[0]+5, :] = -float("inf")
    return maxIndices

def estimateAngle(snapShot):
    angle = 0
    return angle