#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 29 14:49:48 2023

@author: jtm545
"""

import os
import os.path as op
import sys
sys.path.append("/home/j/jtm545/MonBinMRI/Code/PyPlr")
sys.path.append("/home/j/jtm545/MonBinMRI/Code/PySilSub")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pysilsub.problems import SilentSubstitutionProblem
from pysilsub.binocular import BinocularStimulationDevice

S1 = SilentSubstitutionProblem.from_json('./calibration/STLAB_1_York.json')
S2 = SilentSubstitutionProblem.from_json('./calibration/STLAB_2_York.json')

S1_S2_calibration_ratio = pd.read_csv(
    './calibration/S1_S2_calibration_ratio.csv',
    index_col='Primary'
    ).squeeze()
    
# Define the problems
for problem in [S1, S2]:
    problem.ignore = ['rh']
    problem.target = ['mel']
    problem.silence = ['sc', 'mc', 'lc']                
    problem.target_contrast = 0.

S1.target_contrast = 'max'

solution_1 = S1.optim_solve() 
S1.plot_solution(solution_1.x)

Sbin = BinocularStimulationDevice(S1, S2)
Sbin.anchor = 'right'
Sbin.optim = 'left'


settings = [solution_1.x[0:10], solution_1.x[10:]]
new = Sbin.optimise_settings(settings)

# Assign the backgrounds
S1.background = solution_1.x[0:10]
S2.background = new[0][0:10]

# Plot the solutions
S1.plot_solution((solution_1.x[10:] / S1_S2_calibration_ratio))
S2.plot_solution(new[1])


