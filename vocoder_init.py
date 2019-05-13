# -*- coding: utf-8 -*-
"""
Created on Wed May  1 15:17:46 2019

@author: Jbeim
"""

    

import numpy as np
import scipy as sp
from scipy import io as sio
import pyaudio as pa
from NeurToBinMatrix import NeurToBinMatrix
from matplotlib import pyplot as plt




fileName = 'TestData.mat'
elecFreqs = None
spread = None
neuralLocsOct = None
nNeuralLocs = 300
MCLmuA = None
TmuA = None
tAvg = .005
audioFs = 44100
tPlay = None
tauEnvMS = 10
playAudio = False
nCarriers = 20



plt.figure(1)
plt.pcolormesh(np.arange(600),fftFreqs,np.abs(spectHolder))
plt.figure(2)
plt.pcolormesh(np.arange(600),toneFreqs,np.abs(interpSpect))