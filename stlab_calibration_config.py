#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 25 09:17:10 2023

@author: bbsrc
"""

import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Minimum and maximum bit depth
MINTENSITY = 0
MAXTENSITY = 4095

# Multicast addresses used for playing video files. These should
# already be configured, but if not, they can be set manually with
# d.set_multicast_address(...). The broadcast address targets all
# luminaires.
STLAB_1, STLAB_2 = 1021, 1022
STLAB_BROADCAST = 1023

# Background spectra for each device
with open('./calibration/STLAB_1_background.pkl', 'rb') as fp:
    STLAB_1_BACKGROUND = pickle.load(fp)
with open('./calibration/STLAB_2_background.pkl', 'rb') as fp:
    STLAB_2_BACKGROUND = pickle.load(fp)

# Calibration ratio
STLAB_1_2_CALIBRATION_RATIO = pd.read_csv(
    './calibration/S1_S2_calibration_ratio.csv', index_col='Primary'
    ).squeeze()
    

def sample_spectra(d):
    s1 = d.get_spectrometer_spectrum(STLAB_1)
    s2 = d.get_spectrometer_spectrum(STLAB_2)
    return s1[0]*s1[1], s2[0]*s2[1]
    
def plot_spectra(d):
    s1, s2 = sample_spectra(d)
    wls = list(range(380, 785, 5))
    plt.plot(wls, s1, label='STLAB_1')
    plt.plot(wls, s2, label='STLAB_1')
    plt.xlabel('Wavelength (nm)')
    plt.ylabel('Flux (mW)')
    plt.legend()