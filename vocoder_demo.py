# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:05:45 2019

@author: Jbeim
"""

from vocoder import vocoder
from matplotlib.pyplot import plot
import scipy.io as sio
import numpy as np
#import cProfile

Fs = 44100
#%% Compare to matlab data

#audioOut = cProfile.run('vocoder("TestData.mat",playAudio = False)')
#[audioOut,audioOut2,audioOut3] = vocoder('TestData.mat',playAudio = False)
auioOut = vocoder2('TestData.mat',playAudio = False)
matData = sio.loadmat('vocoded2.mat')
#
matAudioOut = matData['testAudioOut'][:,0]
#
#
matPyDiff = audioOut-matAudioOut
Err = np.sum(np.abs(matPyDiff))
#
tseries1 = np.arange(audioOut.size)/Fs
#tseries2 = np.arange(matAudioOut.size)/Fs
#
##plot(tseries1,audioOut,'k--',tseries2,matAudioOut,'r.')
#plot(tseries2,matAudioOut,'r.',tseries1,audioOut,'k--')

plot(tseries1,audioOut/np.max(audioOut),'k',tseries1,audioOut2/np.max(audioOut2),'c',tseries1,audioOut3,'r')
#%% Extract individual variables
#testAudioOut = vocoder('TestData.mat')

