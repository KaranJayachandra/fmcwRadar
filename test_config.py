# Calculating these parameters requires Radar background knowledge. Please
# refer to the the pdf document talking about the basics of Automotive Radar
# systems.

from scipy.constants import c

RADAR = {
    "Carrier Wavelength" : c / 77e9,
    "Range Resolution" : 1,
    "Chirp Bandwidth" : 150e6,
    "Chirp Time" : 25.6e-6,
    "Time Samples in Chirp" : 512,
    "Number of Chirps" : 256,
    "Operating Temperature" : 300,
    "Antenna Gain" : 1,
    "Noise Figure": 100,
    "Array Size": 10,
    "Array Spacing": 0.5
}

ENVIRONMENT = {
    "Total Targets": 2,
    "Target 1" : [100, 10, 30],
    "Target 2" : [150, -20, -45]
}