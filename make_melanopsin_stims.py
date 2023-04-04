#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:27:27 2023

@author: bbsrc
"""

import os
import os.path as op
from itertools import product
from pprint import pprint
from datetime import datetime
import pickle
import json
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysilsub.CIE import get_CIE_1924_photopic_vl
from pysilsub.problems import SilentSubstitutionProblem
from pysilsub.observers import ColorimetricObserver
from pyplr import stlabhelp


# %% ~~~ CONSTANTS ~~~
MINTENSITY = 0
MAXTENSITY = 4095
Fs = 50  # STLAB switching time
VL = get_CIE_1924_photopic_vl()
FREQUENCY = 2
ATTENTION_EVENT = 300  # ms
PEAK_CONTRAST = .15

# Background spectra for each device
with open('./calibration/STLAB_1_background.pkl', 'rb') as fp:
    STLAB_1_BACKGROUND = pickle.load(fp)
with open('./calibration/STLAB_2_background.pkl', 'rb') as fp:
    STLAB_2_BACKGROUND = pickle.load(fp)


# %%
def main(out, age=32):

    print(f'> Making melanopsin stimuli for {out}')


    # %% Conditions
    conditions = ['MonL', 'MonR', 'Bin', 'BinA']
    nstims = 16
    block_length = 384000  # ms

    # %% Make waveforms
    duration = 12
    frequency = 2

    # Times
    sampling_interval = 1.0 / Fs
    wavetime = np.arange(0, duration, sampling_interval)
    ramptime = np.arange(0, 3, sampling_interval)
    offtime = np.arange(duration, duration*2, sampling_interval)

    # Waves
    ramp = np.cos(2 * np.pi * 1/duration * ramptime)
    wave = np.sin(2 * np.pi * frequency * wavetime + 0)
    antiwave = np.cos(2 * np.pi * frequency * wavetime + np.pi/2)
    off = np.zeros(len(offtime))
    fulloff = np.hstack([off, off])

    # Apply half cosine on/off ramp
    wave[450:] = wave[450:] * ramp
    wave[:150] = wave[:150] * ramp[::-1]
    antiwave[450:] = antiwave[450:] * ramp
    antiwave[:150] = antiwave[:150] * ramp[::-1]

    # Apply off period
    wave = np.hstack([wave, off])
    antiwave = np.hstack([antiwave, off])
    fulloff = fulloff

    fulltime = np.hstack([wavetime, offtime])

    # Plotz
    plt.plot(fulltime, wave, label='Wave')
    plt.plot(fulltime, antiwave, label='Antiwave')
    plt.xlabel('Time (s)')
    plt.ylabel('Contrast')
    plt.legend()

    # %% spectral switching times
    times = []
    for i in range(nstims):
        times.append(np.arange(0, 24000, 1000/Fs).astype('int') + (24000 * i))

    event_onsets = [t[0] for t in times]
    video_times = np.hstack(times + [384000])

    # %%
    
    obs = ColorimetricObserver(age=age, field_size=10)
    # Load predictive models for each device and plug in the observer
    S1 = SilentSubstitutionProblem.from_json('./calibration/STLAB_1_York.json')
    S2 = SilentSubstitutionProblem.from_json('./calibration/STLAB_2_York.json')
    S1.observer, S2.observer = obs, obs
    
    # We know that the two devices differ slightly in output. Here we obtain
    # a calibration ratio for each LED that *may* be used to perform a simple
    # correction later.
    S1_S2_calibration_ratio = pd.read_csv(
        './calibration/S1_S2_calibration_ratio.csv',
        index_col='Primary'
        ).squeeze()
    
    # Plot the calibration spds, do the gamma corrections, save output, etc.
    for device in [S1, S2]:
        # Keep a log of which device / calibration was used to prepare the stims
        # and at what time
        with open(f'./{out}/{device.config["json_name"]}_device_log.txt', 'w') as fh:
            pprint(device.config, stream=fh)
            print(f'\n> Time created: {datetime.now()}', file=fh)

        # Perform gamma correction
        device.do_gamma(fit='polynomial', force_origin=True)
        device.gamma[device.gamma < MINTENSITY] = MINTENSITY
        device.gamma[device.gamma > MAXTENSITY] = MAXTENSITY
    # %%
   
    # Adjust intensity amplitudes to native resolution
    wave_contrasts = wave * PEAK_CONTRAST
    antiwave_contrasts = wave * PEAK_CONTRAST

    
    # Define the problems
    for problem in [S1, S2]:
        problem.ignore = ['rh']
        problem.target = ['mel']
        problem.silence = ['sc', 'mc', 'lc']                
        problem.target_contrast = 0.
    S1.background = STLAB_1_BACKGROUND
    S2.background = STLAB_2_BACKGROUND 

        
    # Find the solutions
    s1_wave_solutions = []
    s2_wave_solutions = []
    for target_contrast in wave_contrasts:
        S1.target_contrast = target_contrast
        s1_wave_solutions.append(S1.linalg_solve())
        S2.target_contrast = target_contrast
        s2_wave_solutions.append(S2.linalg_solve())
        
    # Find the solutions
    s1_antiwave_solutions = []
    s2_antiwave_solutions = []
    for target_contrast in antiwave_contrasts:
        S1.target_contrast = target_contrast
        s1_antiwave_solutions.append(S1.linalg_solve())
        S2.target_contrast = target_contrast
        s2_antiwave_solutions.append(S2.linalg_solve())
        
    #%%
    
    # Left eye
    S1_primary_waves = (
        pd.DataFrame([S1.w2s(s) for s in s1_wave_solutions])
    ).astype('int')
    
    S1_primary_antiwaves = (
        pd.DataFrame([S1.w2s(s) for s in s1_antiwave_solutions])
    ).astype('int')
    
    # Right eye
    S2_primary_waves = (
        pd.DataFrame([S2.w2s(s) for s in s2_wave_solutions])
    ).astype('int')
    
    S2_primary_antiwaves = (
        pd.DataFrame([S2.w2s(s) for s in s2_antiwave_solutions])
    ).astype('int')
    
    # Background
    S1_primary_background = (
        pd.DataFrame([STLAB_1_BACKGROUND for s in range(len(S1_primary_waves))])
    ).astype('int')
    
    S2_primary_background = (
        pd.DataFrame([STLAB_2_BACKGROUND for s in range(len(S2_primary_waves))])
    ).astype('int')

    # Get trial list
    trials = (conditions * 4)
    random.shuffle(trials)

    # Make the trials
    left_stims = []
    right_stims = []
    attention_idxs = []
    BinA_counter = 0
    for i, trial in enumerate(trials):
        if trial == 'MonL':
            left = S1_primary_waves.copy(deep=True)
            right = S2_primary_background.copy(deep=True)

        if trial == 'MonR':
            left = S1_primary_background.copy(deep=True)
            right = S2_primary_waves.copy(deep=True)

        if trial == 'Bin':
            left = S1_primary_waves.copy(deep=True)
            right = S2_primary_waves.copy(deep=True)

        if trial == 'BinA':
            if BinA_counter < 2:
                # Binocular in counter phase
                left = S1_primary_waves.copy(deep=True)
                right = S2_primary_antiwaves.copy(deep=True)
            else:
                # Binocular in phase
                left = S1_primary_antiwaves.copy(deep=True)
                right = S2_primary_waves.copy(deep=True)

            # Keep track of antiphase stims
            BinA_counter += 1

        # Add attention event
        start = random.choice(range(0, len(S1_primary_waves), 1))
        end = start + 15  # 300 ms
        attention_idxs.append(start)

        left.iloc[start:end] = left.iloc[start:end].mul(.5).astype('int')
        right.iloc[start:end] = right.iloc[start:end].mul(.5).astype('int')

        # Save the stims
        left_stims.append(left)
        right_stims.append(right)

    # Join up the left
    left_video = pd.concat(left_stims, axis=0)
    left_video = pd.concat(
        [left_video, left_video.tail(1)]).reset_index(drop=True)
    left_video.index = video_times
    left_video.index.name = 'time'

    # Join up the right
    right_video = pd.concat(right_stims, axis=0)
    right_video = pd.concat(
        [right_video, right_video.tail(1)]).reset_index(drop=True)
    right_video.index = video_times
    right_video.index.name = 'time'

    # Plotz
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(24, 4))
    left_video.plot(ax=ax1, legend=False)
    right_video.plot(ax=ax2, legend=False)
    ax1.set_ylabel('Left eye')
    ax2.set_ylabel('Right eye')
    ax1.set_xlabel('')


    for x, t in zip(event_onsets, trials):
        plt.text(x+6000, 5000, t, horizontalalignment='center', fontsize=14)
    plt.tight_layout()
    fig.savefig(op.join(out, 'stimuli.png'))

    # Event info
    off_times = ['off' for trial in trials]
    onsets = list(range(0, 384000, 12000))
    new_events = [val for pair in zip(trials, off_times) for val in pair]
    meta = {
        'onsets': onsets,
        'events': new_events
        }
    new = pd.DataFrame(zip(onsets, new_events))
    attention_times = [idx*20+t for idx, t in zip(attention_idxs, onsets[::2])]
    attention_times = pd.DataFrame([attention_times, ['Attention']*len(attention_times)])
    run_events = pd.concat([new, attention_times.T]).sort_values(by=0).reset_index(drop=True)
    run_events.columns = ['time', 'event']
    run_events.to_csv(op.join(out, 'events.csv'), header=None, index=None)
    
    # Make the left video files!        
    left_video = left_video.reset_index()
    left_video.columns = stlabhelp.get_video_cols()
    vf_left_name = 'melanopsin_left.json'
    stlabhelp.make_video_file(left_video, op.join(out, vf_left_name), metadata=meta)

    # Make the left video files!
    right_video = right_video.reset_index()
    right_video.columns = stlabhelp.get_video_cols()
    vf_right_name = 'melanopsin_right.json'
    stlabhelp.make_video_file(right_video, op.join(out, vf_right_name), metadata=meta)
    
    return vf_left_name, vf_right_name, run_events

#%%

if __name__ == '__main__':
    events = main('.')