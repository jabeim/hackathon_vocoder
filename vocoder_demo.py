# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:05:45 2019

@author: Jbeim
"""

#from vocoder import vocoder
from matplotlib.pyplot import plot
import numpy as np
import pyaudio as pa
from mat4py import loadmat
from scipy.io.wavfile import write as wavwrite
from ab_standard import vocoder as standard_voc
from ab_spiral import vocoder as hybrid_voc
#import cProfile

Fs = 44100
playAudio = False
#%% Compare to matlab data

#audioOut = cProfile.run('vocoder("TestData.mat",playAudio = False)')
#[audioOut,audioOut2,audioOut3] = vocoder('TestData.mat',playAudio = False)
#auioOut = vocoder2('TestData.mat',playAudio = False)
nlVal = 5

standardOut = standard_voc('TestData.mat',nl=nlVal)
hybridOut = hybrid_voc('TestData.mat',nl=nlVal)
#%%Plotting
#matData = sio.loadmat('vocoded2.mat')
startingData = loadmat('TestData.mat','S')
newdata = loadmat('Hackathon_scope_demo/ScopeData_17-May-2019_1.scope')
matData = loadmat('vocoded2.mat')
##
#matAudioOut = matData['testAudioOut'][:,0]
matAudioOut =np.array(matData['testAudioOut'])
matAudioOut = matAudioOut[:,0]

matPyDiff = standardOut-matAudioOut
Err = np.sum(np.abs(matPyDiff))

tseries1 = np.arange(standardOut.size)/Fs
tseries2 = np.arange(matAudioOut.size)/Fs
#
plot(tseries1,standardOut,'k--',tseries2,matAudioOut,'r:')
#plot(tseries2,matAudioOut,'r.',tseries1,audioOut,'k--')

#plot(tseries1,audioOut/np.max(audioOut),'k',tseries1,audioOut2/np.max(audioOut2),'c',tseries1,audioOut3,'r')

outFileName = 'TestData_voc.wav'
audioFs = 44200
audioNorm = standardOut
wavData = (audioNorm*(2**32-1)).astype(np.int32)
#wavData = (audioNorm*(2**15-1)).astype(np.int16) 
wavwrite(outFileName,audioFs,wavData)

#%% Play Audio Output
if playAudio:
    output = np.float32(standardOut)/np.max(standardOut)
    # add zeros before second output sample
    output2 = np.float32(np.concatenate((np.zeros(Fs),hybridOut)))/np.max(hybridOut)
    p = pa.PyAudio()
    devInfo = p.get_default_output_device_info()
    devIndex = devInfo['index']
    nChan = 1
    stream = p.open(format=pa.paFloat32,
                    channels=nChan,
                    rate=Fs,
                    output=True,
                    output_device_index = devIndex
                    )
    #        inData = audiodata.astype(np.float32).tostring()
    outData = output.astype(np.float32).tostring()
    outData2 = output2.astype(np.float32).tostring()
    stream.write(outData)
    
    stream.write(outData2)
    stream.close()