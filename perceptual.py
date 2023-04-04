#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 24 10:59:30 2023

@author: bbsrc
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 21 07:25:13 2023

@author: bbsrc
"""

import os
import os.path as op
import re
import json
import pickle
import glob
import shutil
from time import sleep, time

import pandas as pd
from pynput import keyboard
from pyplr import stlabscene
import make_luminance_stims, make_melanopsin_stims


# 
USE_STLAB = True
CONDITIONS = ['S', 'L-M']


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
        subjects_dir = f'./Subjects/'
        if not op.exists(subjects_dir):
            os.makedirs(subjects_dir)
            print(f'> Created subjects folder: {subjects_dir}')

        # Ask for subject RNUM and block number and condition
        rnumber = input_rnumber()
        condition = input_condition()

        # Create RNUM directory
        r_dir = f'./Subjects/{rnumber}'
        if not op.exists(r_dir):
            os.makedirs(r_dir)
            print(f'* Created folder: {r_dir}')

        # Create subject stims directory
        out_dir = f'./Subjects/{rnumber}/{condition}/'
        if not op.exists(out_dir):
            os.makedirs(out_dir)
            print(f'* Created folder: {out_dir}')

        # Get subject age if not already saved
        if not op.exists(op.join(r_dir, 'rinfo.json')):
            subject_age = input_subject_age()
            with open(op.join(r_dir, 'rinfo.json'), 'w') as fh:
                json.dump({'Age': subject_age}, fh)
        else:
            with open(op.join(r_dir, 'rinfo.json'), 'r') as fh:
                subject_age = json.load(fh)['Age']

        # Make stimuli
        if condition == 'L-M':
            events = make_luminance_stims.main(out_dir)


        # Background spectra for each device
        with open('./calibration/STLAB_1_background.pkl', 'rb') as fp:
            STLAB_1_BACKGROUND = pickle.load(fp)
        with open('./calibration/STLAB_2_background.pkl', 'rb') as fp:
            STLAB_2_BACKGROUND = pickle.load(fp)

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

            # Get the correct video file for each eye, for this block
            video_left = 'lum_left.json'
            video_right = 'lum_right.json'

            # First, get rid of any old video files in the current working dir.
            for f in glob.glob('./*.json'):
                os.remove(f)
                print(f'> Removed {f} from the current working directory.')

            # Now copy accross the required files and rename. Note that in
            # MonL/MonR conditions only one video file is played, so we catch
            # the error and set the value to None, which means that a video
            # will not be played for that eye when we issue the scene command.
            try:
                video_1 = shutil.copyfile(
                    op.join(out_dir, video_left), './video1.json')

            except Exception:
                video_1 = None

            try:
                video_2 = shutil.copyfile(
                    op.join(out_dir, video_right), './video2.json')

            except Exception:
                video_2 = None

        if USE_STLAB:
            # Upload and cache the video files.
            for video_file in [video_1, video_2]:
                if video_file is not None:
                    d.upload_video(video_file)
                    _ = d.get_video_file_metadata(video_file)
            print('> Video file(s) uploaded and cached on LIGHT HUB.')

        else:
            print('> Video files would be uploaded here...')
        
        # Start listening for button responses
        response_listener = keyboard.Listener(
            on_press=receive_button_press,
            on_release=receive_button_release
            )
        response_listener.start()
        
        # Wait for a '5'
        _ = wait_for_mri_trigger()

        # Time before starting the experiment
        global PROTOCOL_START_TIME 
        PROTOCOL_START_TIME = time()

        # Launch the vide files
        if USE_STLAB:
            # Launch the video files and time how long the script is blocked
            pre_scene_launch = time()
            d.scene(STLAB_1, STLAB_2, video_1, video_2)
            post_scene_launch = time()

            # Calculate and display blocking time
            dead_time = post_scene_launch - pre_scene_launch
            print(f'Dead time: {dead_time} s')

        else:
            print('> Scene command would be used to launch video'
                  + 'files here...')

        # Keep track of what's going on
        protocol_timer(events, '> \nLaunched stimulus protocol\n')

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
        print(f'LIGHT HUB blocking time: {dead_time} s')
        print(f'Ellapsed time: {PROTOCOL_END_TIME-PROTOCOL_START_TIME}')
        
        # Stop the response listener
        response_listener.stop()
        
        # Display and save button responses
        response_df = pd.DataFrame(
            RESPONSES, columns=['time', 'button', 'event']
            )
        print('The following button responses were made:\n')
        print(response_df)
        print('\n')
        response_df.to_csv(op.join(block_dir, 'responses.csv'), index=False)
            
        for f in glob.glob('./*.json'):
            os.remove(f)
            print(f'> Removed {f} from the current working directory.')

        if USE_STLAB:
            d.stop_video(STLAB_1)
            d.stop_video(STLAB_2)
            d.turn_off(STLAB_BROADCAST)  # Probably don't want to do this
            d.logout()
            print('> Turned off STLAB and logged out of the LIGHT HUB.')


if __name__ == '__main__':
    main()
