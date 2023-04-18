#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 11:14:04 2023

@author: bbsrc

This script implements a psychophysical nulling procedure for STLAB.

"""

from pysilsub.observers import ColorimetricObserver
from pysilsub.problems import SilentSubstitutionProblem
from pyplr import stlabscene
from pyplr import stlabhelp
from pynput import mouse  
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import pickle
import glob
import os
#sys.path.append("/home/j/jtm545/MonBinMRI/Code/PyPlr")
#sys.path.append("/home/j/jtm545/MonBinMRI/Code/PySilSub")


# Upper and lower limites, -90 to 0 or 30
# Random starting point within range on each trial
# Right click to start trial
# Background spectrum between trials
# Optimal flicker... ?
# Display diagram after protocol
# Include standard settings in case of bad data

# Configuration
TARGET = ['mc', 'lc']
SILENCE = ['sc', 'mel']
IGNORE = ['rh']



# Use global variables. This is bad practice because it makes code hard to test 
# and diagnose, but in this case they make sense (to me).
global FLICKER_FREQUENCY, TARGET_CONTRAST, RESPONSES, LOWER_LIMIT, UPPER_LIMIT, ALPHA, TRIALNUM
FLICKER_FREQUENCY = 4  # Hz
TARGET_CONTRAST = [0.0, 0.0]
RESPONSES = []
LOWER_LIMIT, UPPER_LIMIT = -np.pi/2., np.pi/5  # -90 to 36 degrees
TRIALNUM = 0

OBSERVER_AGE = 41
MAXTENSITY = 4095
STLAB_1, STLAB_2 = 1021, 1022
STLAB_BROADCAST = 1023



def new_alpha():
    """New random alpha value between -90 and 36 degrees."""
    return np.random.uniform(-np.pi/2., np.pi/5)

ALPHA = new_alpha()

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
global S1, S2
S1 = SilentSubstitutionProblem.from_json('./calibration/STLAB_1_York.json')
S2 = SilentSubstitutionProblem.from_json('./calibration/STLAB_2_York.json')
S1.observer = ColorimetricObserver(age=OBSERVER_AGE)
S2.observer = ColorimetricObserver(age=OBSERVER_AGE)

for problem in [S1, S2]:
    problem.target = TARGET
    problem.silence = SILENCE
    problem.ignore = IGNORE
    problem.target_contrast = TARGET_CONTRAST

# Set the background spectra
S1.background = STLAB_1_BACKGROUND
S2.background = STLAB_2_BACKGROUND

# Backgrounds
global bg1, bg2
bg1 = pd.Series(S1.background)
bg2 = pd.Series(S2.background)

def plot_responses(responses):
    """Plot average and individual responses."""
    a = np.linspace(0, np.pi*2, 100)
    x = np.cos(a)
    y = np.sin(a)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_aspect('equal', 'box')
    ax.hlines(0, -1, 1, color='k', ls='--')    
    ax.vlines(0, -1, 1, color='k', ls='--')
    arc = np.linspace(LOWER_LIMIT, UPPER_LIMIT, 100)
    ax.plot(np.sin(arc+np.pi)/2, np.cos(arc+np.pi)/2, color='k')
    ax.plot(np.sin(arc)/2, np.cos(arc)/2, color='k')
    for r in responses:
        ax.plot([-np.cos(r), np.cos(r)], [-np.sin(r), np.sin(r)], 
                lw=.5, ls=':', c='k')
    average_r = np.array(responses).mean()
    ax.plot([-np.cos(average_r), np.cos(average_r)], 
            [-np.sin(average_r), np.sin(average_r)], 
            lw=2, ls='-', c='k')
    ax.set_xlabel('L')
    ax.set_ylabel('M')
    return fig

def make_video_df(bg, mod):
    """Generate DataFrame for video file."""
    time = [0, time_step, time_step*2]
    vf_df = pd.concat([mod, bg, mod], axis=1).T.mul(MAXTENSITY).astype(int)
    vf_df.index = time
    vf_df = vf_df.reset_index()
    vf_df.columns = stlabhelp.get_video_cols()
    return vf_df


def make_next_video_files(video_file_1_df, video_file_2_df):
    """Make video files from DataFrames."""
    stlabhelp.make_video_file(video_file_1_df, 'video1.json', repeats=9999)
    stlabhelp.make_video_file(video_file_2_df, 'video2.json', repeats=9999)


def upload_video_files(video_left, video_right):
    """Upload and cache video files."""
    for video_file in [video_left, video_right]:
        if video_file is not None:
            d.upload_video(video_file)
            _ = d.get_video_file_metadata(video_file)


def update_stimulus():
    """Generate new modulation spectra for each device."""
    TARGET_CONTRAST[0] = .1 * np.sin(ALPHA)
    TARGET_CONTRAST[1] = .1 * np.cos(ALPHA)
    for problem in [S1, S2]:
        problem.target_contrast = TARGET_CONTRAST
    mod1 = S1.linalg_solve()
    mod2 = S2.linalg_solve()
    return mod1, mod2


def present_next_flicker():
    """Prepare and present next video files."""
    mod1, mod2 = update_stimulus()
    video_file_1_df = make_video_df(bg1-(mod1-bg1), mod1)
    video_file_2_df = make_video_df(bg2-(mod2-bg2), mod2)
    make_next_video_files(video_file_1_df, video_file_2_df)
    upload_video_files('video1.json', 'video2.json')
    d.scene(STLAB_1, STLAB_2, 'video1.json', 'video2.json')
    print(f'Target contrasts (mc, lc): {TARGET_CONTRAST}, alpha: {ALPHA} ({ALPHA*180/np.pi} degrees)')


def on_scroll(x, y, dx, dy):
    """Present the next flicker after mouse scroll."""
    global ALPHA
    print('Mouse scrolled at ({0}, {1})({2}, {3})'.format(x, y, dx, dy))
    if dy == 1:
        if ALPHA < UPPER_LIMIT:
            ALPHA += .05
            present_next_flicker()
            print('Up')

    if dy == -1:
        if ALPHA > LOWER_LIMIT:
            ALPHA -= .05
            present_next_flicker()
            print('Down')

    
def on_click(x, y, button, pressed):
    """Report and log button press events."""
    global ALPHA, TRIALNUM
    try:
        if button == mouse.Button.left and pressed == True:
            print('Accepting trial')
            RESPONSES.append(ALPHA)
            # Stop videos and apply background spectra
            d.stop_video(STLAB_1)
            d.stop_video(STLAB_2)
            d.set_spectrum_a(STLAB_1_BACKGROUND, STLAB_1)
            d.set_spectrum_a(STLAB_2_BACKGROUND, STLAB_2)
            ALPHA = new_alpha()
            TRIALNUM += 1

    except:
        pass
    
    if TRIALNUM == 10:
        return False


def main():

    try:
        print('Perceptual nulling protocol.')
        global d
        d = stlabscene.SpectraTuneLabScene.from_config()
        
        # Apply background spectra
        d.set_spectrum_a(STLAB_1_BACKGROUND, STLAB_1)
        d.set_spectrum_a(STLAB_2_BACKGROUND, STLAB_2)

        # Start listening for button responses
        with mouse.Listener(on_click=on_click, on_scroll=on_scroll) as listener:
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
        
        # Responses
        print(RESPONSES)
        with open('./test/result.pkl', 'wb') as fp:
            pickle.dump(RESPONSES, fp)
        fig = plot_responses(RESPONSES)
        fig.savefig('./test/result.png')
        
        # Stop videos aqnd turn off
        d.stop_video(STLAB_1)
        d.stop_video(STLAB_2)
        d.turn_off(STLAB_1)
        d.turn_off(STLAB_2)

        # Stop listening for key responses
        listener.stop()

        # Just in case there are any video files left in the cwd
        for f in glob.glob('./*.json'):
            os.remove(f)
            print(f'> Removed {f} from the current working directory.')


if __name__ == '__main__':
    main()
