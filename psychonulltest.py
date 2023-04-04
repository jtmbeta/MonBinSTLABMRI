#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 11:06:23 2023

@author: bbsrc
"""

from pysilsub.observers import ColorimetricObserver
from pysilsub.problems import SilentSubstitutionProblem
from pyplr import stlabscene
from pyplr import stlabhelp
from pynput import keyboard
import pandas as pd
import numpy as np
import pickle
import re
import pprint
import glob
import os
import sys
sys.path.append("/home/j/jtm545/MonBinMRI/Code/PyPlr")
sys.path.append("/home/j/jtm545/MonBinMRI/Code/PySilSub")


# Configuration
TARGET = ['mc', 'lc']
SILENCE = ['sc', 'mel']
IGNORE = ['rh']

global PREVIOUS_NULLING_CONTRAST, NULLING_CONTRAST, FLICKER_FREQUENCY, STEP, TARGET_CONTRAST
FLICKER_FREQUENCY = 4  # Hz
PREVIOUS_NULLING_CONTRAST = .0
NULLING_CONTRAST = .0
STEP = .005
TARGET_CONTRAST = [0.0, -.05]
OBSERVER_AGE = 41
MINTENSITY = 0
MAXTENSITY = 4095
STLAB_1, STLAB_2 = 1021, 1022
STLAB_BROADCAST = 1023

global alpha
alpha = 3*np.pi/2

# Time step for making video files
time_step = int((
    int(FLICKER_FREQUENCY*1000)
    if FLICKER_FREQUENCY < 1.
    else int((1 / FLICKER_FREQUENCY) * 1000)
)/2)

# Background spectra for each device
with open('./calibration/STLAB_1_background.pkl', 'rb') as fp:
    STLAB_1_BACKGROUND = pickle.load(fp)
with open('./calibration/STLAB_2_background.pkl', 'rb') as fp:
    STLAB_2_BACKGROUND = pickle.load(fp)

# Load predictive models for each device and plug in the observer
global s1, s2
s1 = SilentSubstitutionProblem.from_json('./calibration/STLAB_1_York.json')
s2 = SilentSubstitutionProblem.from_json('./calibration/STLAB_2_York.json')
s1.observer = ColorimetricObserver(age=OBSERVER_AGE)
s2.observer = ColorimetricObserver(age=OBSERVER_AGE)

for problem in [s1, s2]:
    problem.target = TARGET
    problem.silence = SILENCE
    problem.ignore = IGNORE
    problem.target_contrast = TARGET_CONTRAST

# Set the background spectra
s1.background = STLAB_1_BACKGROUND
s2.background = STLAB_2_BACKGROUND

# # modulations
# mod1 = s1.linalg_solve()
# mod2 = s2.linalg_solve()

# Backgrounds
global bg1, bg2
bg1 = pd.Series(s1.background)
bg2 = pd.Series(s2.background)


def make_video_df(bg, mod):
    time = [0, time_step, time_step*2]
    vf_df = pd.concat([mod, bg, mod], axis=1).T.mul(MAXTENSITY).astype(int)
    vf_df.index = time
    vf_df = vf_df.reset_index()
    vf_df.columns = stlabhelp.get_video_cols()
    return vf_df


def make_next_video_files(video_file_1_df, video_file_2_df):
    stlabhelp.make_video_file(video_file_1_df, 'video1.json', repeats=9999)
    stlabhelp.make_video_file(video_file_2_df, 'video2.json', repeats=9999)


def upload_video_files(video_left, video_right):
    for video_file in [video_left, video_right]:
        if video_file is not None:
            d.upload_video(video_file)
            _ = d.get_video_file_metadata(video_file)


def update_stimulus(step):
    TARGET_CONTRAST[0] = .12 * np.cos(alpha)
    TARGET_CONTRAST[1] = .12 * np.sin(alpha)
    for problem in [s1, s2]:
        problem.target_contrast = TARGET_CONTRAST
    mod1 = s1.linalg_solve()
    mod2 = s2.linalg_solve()
    return mod1, mod2


def present_next_flicker(step):
    mod1, mod2 = update_stimulus(step)
    video_file_1_df = make_video_df(bg1-(mod1-bg1), mod1)
    video_file_2_df = make_video_df(bg2-(mod2-bg2), mod2)
    make_next_video_files(video_file_1_df, video_file_2_df)
    upload_video_files('video1.json', 'video2.json')
    d.scene(STLAB_1, STLAB_2, 'video1.json', 'video2.json')
    print(PREVIOUS_NULLING_CONTRAST, NULLING_CONTRAST,
          FLICKER_FREQUENCY, STEP, TARGET_CONTRAST, alpha*180/np.pi)


def on_press(key):
    global alpha
    """Report and log button press events."""
    try:
        if key.char == '1':
            print('Decreasing contrast')
            alpha -= .1
            present_next_flicker(-STEP)

        if key.char == '2':
            print('Increasing contrast')
            alpha += .1
            present_next_flicker(STEP)

        if key.char == '3':
            print('Trial accepted')

        if key.char == '9':
            return False
    except:
        pass


def main():

    try:
        print('Perceptual nulling protocol.')
        global d
        d = stlabscene.SpectraTuneLabScene.from_config()

        # Start listening for button responses
        with keyboard.Listener(on_press=on_press) as listener:
            listener.join()

    except Exception as e:
        print(e)

    finally:
        print('\n')
        print(f"{'='*80 : ^80}")
        print(f"{'END OF PROTOCOL': ^80}")
        print(f"{'='*80 : ^80}")
        print('\n')
        print('Cleaning up...')

        # Stop videos
        d.stop_video(STLAB_1)
        d.stop_video(STLAB_2)

        # Stop listening for key responses
        listener.stop()

        # Just in case there are any video files left in the cwd
        for f in glob.glob('./*.json'):
            os.remove(f)
            print(f'> Removed {f} from the current working directory.')


if __name__ == '__main__':
    main()
