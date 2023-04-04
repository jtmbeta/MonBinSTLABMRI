#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 11 16:37:42 2022

@author: jtm545

Script to make the stimuli for the MonBin2 experiment. 
"""
import os
import os.path as op
from pprint import pprint
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import pandas as pd

from pysilsub.problems import SilentSubstitutionProblem as SSP

# %% ~~~ PLOT STYLE ~~~

plt.style.use('seaborn')
plt.rcParams['font.size'] = 14
plt.rcParams['font.family'] = 'Helvetica'

# %% Functions


def check_exists(folder: str) -> str:
    if not op.exists(folder):
        os.makedirs(folder)
    return folder


# %% ~~~ CONSTANTS ~~~


MINTENSITY = 0
MAXTENSITY = 4095
BACKGROUND = MAXTENSITY/2
Fs = 100  # STLAB switching time
MAX_S_CONE_CONTRAST = .45


# %% ~~~ MAIN SCRIPT ~~~

def main():

    gamma_folder = check_exists('./gamma/')

    # Load predictive models for each device and plug in the observer
    S1 = SSP.from_json('./STLAB_1_York.json')
    S2 = SSP.from_json('./STLAB_2_York.json')

    # We know that the two devices differ slightly in output. Here we obtain
    # a calibration ratio for each LED that *may* be used to perform a simple
    # correction later.
    S1_S2_calibration_ratio = (
        S1.calibration.groupby(level=0).sum().sum(axis=1)
        / S2.calibration.groupby(level=0).sum().sum(axis=1)
        )
    print('> S1/S2 LED calibration ratio')
    print(S1_S2_calibration_ratio)
    S1_S2_calibration_ratio.to_csv('./S1_S2_calibration_ratio.csv')

    # To scale Y-axis for calibration plots
    max_counts = max(S1.calibration.max().max(), S2.calibration.max().max())

    # Plot the calibration spds, do the gamma corrections, save output, etc.
    for device in [S1, S2]:
        # Plot spds
        fig, ax = plt.subplots(figsize=(12, 4))
        device.plot_calibration_spds(ax=ax)
        ax.set_ylim(0, max_counts*1.05)
        fig.savefig(
            f'./{device.config["json_name"]}_calibration_spds.svg')

        # Keep a log of which device / calibration was used to prepare the stims
        # and at what time
        with open(f'./{device.config["json_name"]}_device_log.txt', 'w') as fh:
            pprint(S1.config, stream=fh)
            print(f'\n> Time created: {datetime.now()}', file=fh)

        # Perform gamma correction
        device.do_gamma(fit='polynomial')
        device.gamma[device.gamma < MINTENSITY] = MINTENSITY
        device.gamma[device.gamma > MAXTENSITY] = MAXTENSITY
        device.gamma.to_csv(
            op.join(gamma_folder, f'./{device.config["json_name"]}_gamma_table.csv'))
        device.plot_gamma(save_plots_to=gamma_folder, show_corrected=True)

    # Match backgrounds and pickle
    S1.background = pd.Series([.5] * S1.nprimaries)
    S2.background = S1.background * S1_S2_calibration_ratio

    # Pickle backgrounds so they can be loaded at start of experiment script
    with open('./STLAB_1_background.pkl', 'wb') as fh:
        pickle.dump(S1.w2s(S1.background), fh)
    with open('./STLAB_2_background.pkl', 'wb') as fh:
        pickle.dump(S2.w2s(S2.background), fh)
        
    # Save plot of the background spectra
    fig, ax = plt.subplots()
    s1_bg = S1.predict_multiprimary_spd(S1.background)
    s2_bg = S2.predict_multiprimary_spd(S2.background)
    ax.plot(s1_bg, label='STLAB_1 background')
    ax.plot(s2_bg, label='STLAB_2 background')
    ax.legend()
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(S1.config['calibration_units'])
    ax.set_title('Background spectra')
    fig.savefig('./Background_spectra.svg')
        
if __name__ == '__main__':
    main()
