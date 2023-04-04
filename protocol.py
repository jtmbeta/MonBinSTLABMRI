#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:25:13 2023

@author: bbsrc
"""

import os
import os.path as op
import sys
sys.path.append("/home/j/jtm545/MonBinMRI/Code/PyPlr")
sys.path.append("/home/j/jtm545/MonBinMRI/Code/PySilSub")
import re
import json
import pickle
import glob
import shutil
from time import sleep, time


import pandas as pd
import matplotlib.pyplot as plt
from pynput import keyboard
from pyplr import stlabscene
import make_luminance_stims
import make_melanopsin_stims


global PROTOCOL_START_TIME
PROTOCOL_START_TIME = time()
# Whether to use STLAB. If set to False, the protocol will be simulated
# without making a connection to the LightHub.
USE_STLAB = True

# Required for STLAB
if USE_STLAB:
    STLAB_1, STLAB_2 = 1021, 1022
    STLAB_BROADCAST = 1023
    # Background spectra for each device
    with open('./calibration/STLAB_1_background.pkl', 'rb') as fp:
        STLAB_1_BACKGROUND = pickle.load(fp)
    with open('./calibration/STLAB_2_background.pkl', 'rb') as fp:
        STLAB_2_BACKGROUND = pickle.load(fp)

# Experiment conditions
CONDITIONS = ['lum', 'mel']
PROTOCOL_DURATION = 384  # s
RESPONSES = []


# Helper functions
def input_condition():
    condition = ''
    while condition not in CONDITIONS:
        condition = input(f'> Enter condition ({CONDITIONS}): ')
    return condition


def input_rnumber():
    """Input R-number number"""
    rnumber = ''
    while not re.search('^[Rr]\d\d\d\d$', rnumber):
        rnumber = input('> Enter subject R-number: ')
    return rnumber.upper()


def input_block():
    """Input Block number"""
    block = ''
    while not block.isdigit():
        block = input('> Enter block: ')
    return block.zfill(3)


def input_subject_age():
    """Input subject age"""
    age = ''
    while not age.isdigit():
        age = input('> Enter subject age: ')
    return age


def receive_mri_trigger(key):
    """Stop the listener thread and start experiment when a '5' is received."""
    try:
        if key.char == ('5'):
            return False
    except:
        pass


def wait_for_mri_trigger():
    """Wait for a '5' from the MRI scanner."""
    print('Waiting for MRI trigger...')
    with keyboard.Listener(
            on_press=receive_mri_trigger) as listener:
        listener.join()


def receive_button_press(key):
    """Report and log button press events."""
    try:
        if key.char in ('1', '2', '3', '4'):
            t = int((time() - PROTOCOL_START_TIME) * 1000)
            print(f'* Button (press) at {t}')
            RESPONSES.append((t, key.char, 'press'))
    except:
        pass


def receive_button_release(key):
    """Report and log button release events."""
    try:
        if key.char in ('1', '2', '3', '4'):
            t = int((time() - PROTOCOL_START_TIME) * 1000)
            print(f'* Button (release) at {t}')
            RESPONSES.append((t, key.char, 'release'))
    except:
        pass


def protocol_timer(events, message="Waiting..."):
    """Block script, provide stimulus feedback.

    Feedback on event timings is not accurate.

    Parameters
    ----------
    events : pd.DataFrame
        Events to keep track of.
    message : str, optional
        Feedback message. The default is 'Waiting...'.

    Returns
    -------
    None.

    """
    print(message)
    seconds = time() - PROTOCOL_START_TIME
    event_index = 0
    while seconds < PROTOCOL_DURATION:
        try:
            if seconds*1000 >= events.loc[event_index, 'time']:
                e = events.loc[event_index, 'event']
                t = events.loc[event_index, 'time']
                print(f"! Event: {repr(e)} at {t}")
                event_index += 1
        except:
            pass

        print(f"> Time: {round(seconds)} s")
        sleep(1)
        seconds += 1


# Experiment function
def main():
    try:
        print('\n')
        print(f"{'='*80 : ^80}")
        print(f"{'MONBIN MRI PROTOCOL': ^80}")
        print(f"{'='*80 : ^80}")
        print('\n')
        sleep(.1)

        # Subjects folder
        subjects_dir = './Subjects/'
        if not op.exists(subjects_dir):
            os.makedirs(subjects_dir)
            print(f'> Created subjects folder: {subjects_dir}')

        # Ask for subject RNUM and block number and condition
        rnumber = input_rnumber()
        block = input_block()
        condition = input_condition()

        # Create RNUM directory
        r_dir = f'./Subjects/{rnumber}'
        if not op.exists(r_dir):
            os.makedirs(r_dir)
            print(f'* Created folder: {r_dir}')

        # Create subject stims directory
        block_dir = f'./Subjects/{rnumber}/{block}/'
        if not op.exists(block_dir):
            os.makedirs(block_dir)
            print(f'* Created folder: {block}')

        # Get subject age if not already saved
        if not op.exists(op.join(r_dir, 'rinfo.json')):
            subject_age = input_subject_age()
            with open(op.join(r_dir, 'rinfo.json'), 'w') as fh:
                json.dump({'Age': subject_age}, fh)
        else:
            with open(op.join(r_dir, 'rinfo.json'), 'r') as fh:
                subject_age = json.load(fh)['Age']

        # Make stimuli
        if condition == 'lum':
            vf_left, vf_right, events = make_luminance_stims.main(block_dir)
        if condition == 'mel':
            vf_left, vf_right, events = make_melanopsin_stims.main(
                block_dir, age=int(subject_age))

        # Connect to STLAB and authenticate
        if USE_STLAB:
            d = stlabscene.SpectraTuneLabScene.from_config()
            print('\n')

            # Multicast addresses used for playing video files. These should
            # already be configured, but if not, they can be set manually with
            # d.set_multicast_address(...). The broadcast address targets all
            # luminaires
            STLAB_1, STLAB_2 = 1021, 1022
            STLAB_BROADCAST = 1023

            # Half max background for adaptation
            d.set_spectrum_a(STLAB_1_BACKGROUND, STLAB_1)
            print('> Setting background spectrum on STLAB:')
            print(f'\t{STLAB_1_BACKGROUND}')

            d.set_spectrum_a(STLAB_2_BACKGROUND, STLAB_2)
            print('> Setting background spectrum on STLAB:')
            print(f'\t{STLAB_2_BACKGROUND}')

            # NOTE:
            # Video files are kept in the stims folder with informative names.
            # When required, we copy them into the current working directory
            # and change the names to video1.json and video2.json, which is the
            # required format for multiple uploads.
            # Let's always follow the convention below:
            # video1.json plays on STLAB_1, which stimulates the left eye
            # video2.json plays on STLAB_2, which stimulates the right eye

            # First, get rid of any old video files in the current working dir.
            for f in glob.glob('./*.json'):
                os.remove(f)
                print(f'> Removed {f} from the current working directory.')

            # Now copy accross the required files and rename.
            try:
                video_1 = shutil.copyfile(
                    op.join(block_dir, vf_left), './video1.json')
            except FileNotFoundError as e:
                raise(e)
            try:
                video_2 = shutil.copyfile(
                    op.join(block_dir, vf_right), './video2.json')
            except FileNotFoundError as e:
                raise(e)

            # Upload and cache the video files.
            for video_file in [video_1, video_2]:
                if video_file is not None:
                    d.upload_video(video_file)
                    _ = d.get_video_file_metadata(video_file)
            print('> Video file(s) uploaded and cached on LIGHT HUB.')

        else:
            print('> Simulating protocol without STLAB...')

        # Start listening for button responses
        response_listener = keyboard.Listener(
            on_press=receive_button_press,
            on_release=receive_button_release
        )
        response_listener.start()

        # Wait for a '5'
        _ = wait_for_mri_trigger()

        # Clock time after receiving trigger
        PROTOCOL_START_TIME = time()

        # Launch the video files
        if USE_STLAB:
            # Launch the video files and time how long the script is blocked
            pre_scene_launch = time()
            d.scene(STLAB_1, STLAB_2, video_1, video_2)
            post_scene_launch = time()

            # Calculate and display blocking time
            blocking_time = post_scene_launch - pre_scene_launch
            print(f'LIGHT HUB blocking time: {blocking_time} s')

        else:
            print('> Scene command would be used to launch video'
                  + 'files here...')

        # Keep track of what's going on
        protocol_timer(events, '\n> Starting stimulus protocol\n')

    except Exception as e:
        print(e)
        print('Experiment not working. Find a better postdoc.')

    finally:
        print('\n')
        print(f"{'='*80 : ^80}")
        print(f"{'END OF PROTOCOL': ^80}")
        print(f"{'='*80 : ^80}")
        print('\n')
        print('Cleaning up...')

        # Get time at end
        PROTOCOL_END_TIME = time()

        # Show duration
        print(f'Ellapsed time: {PROTOCOL_END_TIME-PROTOCOL_START_TIME}')

        # Stop the response listener
        response_listener.stop()

        # Display and save button responses
        response_df = pd.DataFrame(
            RESPONSES, columns=['time', 'button', 'event']
        )
        print('Subject made the following responses:\n')
        print(response_df)
        print('\n')
        response_df.to_csv(op.join(block_dir, 'responses.csv'), index=False)

        # Just in case there are any video files left in the cwd
        for f in glob.glob('./*.json'):
            os.remove(f)
            print(f'> Removed {f} from the current working directory.')

        if USE_STLAB:
            d.stop_video(STLAB_1)
            d.stop_video(STLAB_2)
            print(f'LIGHT HUB blocking time: {blocking_time} s')
            # d.turn_off(STLAB_BROADCAST)  # Probably don't want to do this
            d.logout()
            print('> Logged out of the LIGHT HUB.')


# Run the protocol
if __name__ == '__main__':
    main()
