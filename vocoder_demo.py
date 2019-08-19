# -*- coding: utf-8 -*-
"""
Created on Wed May  1 14:05:45 2019

@author: Jbeim
"""

#from vocoder import vocoder

import numpy as np
import pyaudio as pa
from scipy.signal import resample
from scipy.io.wavfile import read
#from ab_standard import vocoder as standard_voc
from ab_spiral import vocoder as hybrid_voc
#import cProfile

Fs = 44100


#nums = np.arange(1,11)
nums = np.arange(1,2)
playAudio = True
fileLoc = 'Hackathon_scope_demo/'
nlVal = 5
for fNum in nums:
    fn = 'ScopeData_17-May-2019_'+fNum.astype(str)  
    wavN = 'MOD_MP_1_'+fNum.astype(str)
    #%% Process and visualize data
    
    
#    standardOut,outFs = standard_voc(fn,resistorVal=10,nl=nlVal,MCLmuA = 542)
    hybridOut,outFs = hybrid_voc(fn,resistorVal=10,nl=nlVal,MCLmuA = 542)
    wavIn = read(fileLoc+wavN+'.wav')
    wavData = wavIn[1]/(2**15-1)
    wavFs = wavIn[0]
    
    wavResampled = resample(wavData,((outFs/wavFs)*wavData.shape[0]).astype(int))
    
    #%% Play Audio Output
    if playAudio:
        input1 = np.float32(np.concatenate((wavResampled,np.zeros(Fs))))
#        output1 = np.float32(standardOut)
        # add zeros before second output sample
        output2 = np.float32(np.concatenate((np.zeros(Fs),hybridOut)))
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
        inData = input1.astype(np.float32).tostring()
#        outData1 = output1.astype(np.float32).tostring()
        outData2 = output2.astype(np.float32).tostring()
        
        stream.write(inData)
#        stream.write(outData1)
        stream.write(outData2)
        stream.close()