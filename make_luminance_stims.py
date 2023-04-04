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
from pyplr import stlabhelp


# %% ~~~ CONSTANTS ~~~
MINTENSITY = 0
MAXTENSITY = 4095
Fs = 50  # STLAB switching time
VL = get_CIE_1924_photopic_vl()
FREQUENCY = 2
ATTENTION_EVENT = 300  # ms

# Background spectra for each device
with open('./calibration/STLAB_1_background.pkl', 'rb') as fp:
    STLAB_1_BACKGROUND = pickle.load(fp)
with open('./calibration/STLAB_2_background.pkl', 'rb') as fp:
    STLAB_2_BACKGROUND = pickle.load(fp)


# %%
def main(out):

    print(f'> Making luminance stimuli for {out}')


    # %% Conditions
    block = ''
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
    wave = (np.hstack([wave, off]) + 1) / 2
    antiwave = (np.hstack([antiwave, off]) + 1) / 2
    fulloff = (fulloff + 1) / 2

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
    primary_waves = (
        pd.DataFrame([wave for primary in range(10)]).T
        * MAXTENSITY
    ).astype('int')

    primary_antiwaves = (
        pd.DataFrame([antiwave for primary in range(10)]).T
        * MAXTENSITY
    ).astype('int')

    primary_background = (
        pd.DataFrame([fulloff for primary in range(10)]).T
        * MAXTENSITY
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
            left = primary_waves.copy(deep=True)
            right = primary_background.copy(deep=True)

        if trial == 'MonR':
            left = primary_background.copy(deep=True)
            right = primary_waves.copy(deep=True)

        if trial == 'Bin':
            left = primary_waves.copy(deep=True)
            right = primary_waves.copy(deep=True)

        if trial == 'BinA':
            if BinA_counter < 2:
                # Binocular in counter phase
                left = primary_waves.copy(deep=True)
                right = primary_antiwaves.copy(deep=True)
            else:
                # Binocular in phase
                left = primary_antiwaves.copy(deep=True)
                right = primary_waves.copy(deep=True)

            # Keep track of antiphase stims
            BinA_counter += 1

        # Add attention event
        start = random.choice(range(0, len(primary_waves), 1))
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
    #for ax in [ax1, ax2]:
    #    ax.spines[['right', 'top']].set_visible(False)

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
    vf_left_name = 'luminance_left.json'
    stlabhelp.make_video_file(left_video, op.join(out, vf_left_name), metadata=meta)

    # Make the left video files!
    right_video = right_video.reset_index()
    right_video.columns = stlabhelp.get_video_cols()
    vf_right_name = 'luminance_right.json'
    stlabhelp.make_video_file(right_video, op.join(out, vf_right_name), metadata=meta)
    
    return vf_left_name, vf_right_name, run_events

#%%

if __name__ == '__main__':
    events = main('.')
