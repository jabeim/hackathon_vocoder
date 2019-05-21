# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:05:45 2019

@author: Jbeim
"""

#from vocoder import vocoder
from matplotlib.pyplot import plot
import scipy.io as sio
import numpy as np
import pyaudio as pa
from ab_standard import vocoder as standard_voc
from ab_spiral import vocoder as hybrid_voc
#import cProfile

Fs = 44100
#%% Compare to matlab data

#audioOut = cProfile.run('vocoder("TestData.mat",playAudio = False)')
#[audioOut,audioOut2,audioOut3] = vocoder('TestData.mat',playAudio = False)
#auioOut = vocoder2('TestData.mat',playAudio = False)
nlVal = 5

standardOut = standard_voc('TestData.mat',nl=nlVal)
hybridOut = hybrid_voc('TestData.mat',nl=nlVal)
#%%Plotting
matData = sio.loadmat('vocoded2.mat')
##
matAudioOut = matData['testAudioOut'][:,0]
#
#
matPyDiff = standardOut-matAudioOut
Err = np.sum(np.abs(matPyDiff))

tseries1 = np.arange(standardOut.size)/Fs
tseries2 = np.arange(matAudioOut.size)/Fs
#
plot(tseries1,standardOut,'k--',tseries2,matAudioOut,'r:')
#plot(tseries2,matAudioOut,'r.',tseries1,audioOut,'k--')

#plot(tseries1,audioOut/np.max(audioOut),'k',tseries1,audioOut2/np.max(audioOut2),'c',tseries1,audioOut3,'r')

#%% Play Audio Output
output = np.float32(standardOut)/np.max(standardOut)
output2 = np.float32(np.concatenate((np.zeros(Fs),hybridOut)))/np.max(hybridOut)
#        output = np.float32(testAudioOut)/np.max(testAudioOut)
p = pa.PyAudio()
nChan = 1
stream = p.open(format=pa.paFloat32,
                channels=nChan,
                rate=Fs,
                output=True,
                output_device_index = 3
                )
#        inData = audiodata.astype(np.float32).tostring()
outData = output.astype(np.float32).tostring()
outData2 = output2.astype(np.float32).tostring()
stream.write(outData)

stream.write(outData2)
stream.close()