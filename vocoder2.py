# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:29:54 2019

@author: beimx004
"""
import numpy as np
import scipy as sp
from scipy import io as sio
import pyaudio as pa
from vocoder_tools import ActivityToPower, NeurToAudio


from NeurToBinMatrix import generate_cfs, NeurToBinMatrix
#from NeurToBinMatrix import NeurToBinMatrix
#from NeurToBinMatrix import generate_cfs
import time

import scipy.signal as sps

def vocoder(fileName,**kwargs):

    elecFreqs = kwargs.get('elecFreqs',None)
    spread = kwargs.get('spread',None)
    neuralLocsOct = kwargs.get('neuralLocsOct',None)
    nNeuralLocs = kwargs.get('nNeuralLocs',300)
    MCLmuA = kwargs.get('MCLmuA',None)
    TmuA = kwargs.get('TmuA',None)
    tAvg = kwargs.get('tAvg',.005)
    audioFs = kwargs.get('audioFs',44100)
    tPlay = kwargs.get('tPlay',None)
    tauEnvMS = kwargs.get('tauEnvMS',10)
    outFileName = kwargs.get('outFileName',fileName+'_voc.wav')
    showDetails = kwargs.get('showDetails',False)
    playAudio = kwargs.get('playAudio',False)
    nCarriers = kwargs.get('nCarriers',20)
    
#%% Load .matfile of electrode recording and format data    
    matData = sio.loadmat(fileName)
    resistorValue = 2
    nElec = matData['electrodeAmp'].shape[0]-1
    captFs = float(matData['SampleRate'])
    captTs = 1/captFs
    scaletoMuA = 500/resistorValue
    elData = np.flipud(matData['electrodeAmp'][1:,:])*scaletoMuA
# compute electrode locations in terms of frequency 
    if elecFreqs is None:
        elecFreqs = np.logspace(np.log10(381.5),np.log10(5046.4),nElec)
    else:
        if nElec != elecFreqs.size:
            raise ValueError('# of electrode frequencies does not match recorded data!')
        else:
            elecFreqs = elecFreqs
# load electric field spread data
    if spread is None:
        elecPlacement = np.zeros(nElec).astype(int) # change to zeros to reflect python indexing
        spreadFile = 'C:/Users/Jbeim/Vocoder/spread.mat'
        spread = sio.loadmat(spreadFile)
    else: # This seciont may need reindexing if the actual spread mat data is passed through, for now let use the spread.mat data
        elecPlacement = spread['elecPlacement']
        
# Create octave location of neural populations
    if neuralLocsOct is None:
        neuralLocsOct = np.append(
                np.log2(np.linspace(150,850,40)),
                np.linspace(np.log2(870),np.log2(8000),260)
                )
        
        x = np.linspace(1,neuralLocsOct.size,nNeuralLocs)
        xp = np.arange(1,neuralLocsOct.size+1)
        fp = neuralLocsOct
        
    neuralLocsOct = np.interp(
            np.linspace(1,neuralLocsOct.size,nNeuralLocs),
            np.arange(1,neuralLocsOct.size+1),
            neuralLocsOct)
    
# tauEnvMS to remove carrier synthesis effect
    taus = tauEnvMS/1000
    alpha = np.exp(-1/(taus*captFs))
    
# MCT and T levels in micro amp
    if MCLmuA is None:
        MCLmuA = 500*np.ones(nElec)*1.2
    else:
        if MCLmuA.size == nElec:
            MCLmuA = MCLmuA * 1.2
        elif MCLmuA.size == 1:
            MCLmuA = np.ones(nElec)*MCLmuA*1.2
        else:
            raise ValueError('Wrong number of M levels!')
            
    if TmuA is None:
        TmuA = 50*np.ones(nElec)
    else:
        if TmuA.size == nElec:
            TmuA = TmuA            
        elif TmuA.size == 1:
            TmuA = np.ones(nElec)*TmuA
        else:
            raise ValueError('Wrong Number of T levels!')
            
# Time constant for averaging neural activity to relate to frequency
    tAvg = np.ceil(tAvg/captTs)*captTs
    mAvg = np.round(tAvg/captTs)
    blkSize = mAvg.astype(int)  
            
# audio output frequency
    audioFs = np.ceil(tAvg*audioFs)/tAvg
    audioTs = 1/audioFs
    nAvg = np.int(np.round(tAvg/audioTs))
    tWin = 2*tAvg
    nFFT = np.round(tWin/audioTs).astype(int)
# total audio file length
    if tPlay is None:
        tPlay = 10*tWin;
    else:
        tPlay = np.ceil(tPlay/tWin)*tWin
# create output file name
# this section seems matlab specific and may not be needed    
        
# store original audio for comparison
# this section rescales the presumed pulse audio for comparison

# create metrix to convert electrode charge to electric field

    charge2EF = np.zeros((nNeuralLocs,nElec))
    elecFreqOct = np.log2(elecFreqs)
    
    for iEl in range(nElec):
        f = sp.interpolate.interp1d(
                spread['fOct'][:,elecPlacement[iEl]]+elecFreqOct[iEl],
                spread['voltage'][:,elecPlacement[iEl]],
                fill_value = 'extrapolate')       
        steerVec = f(neuralLocsOct)
        steerVec[steerVec < 0] = 0
        charge2EF[:,iEl] = steerVec
#    return charge2EF
# matrix to map neural activity to FFT bin frequencies
#    here we call another function
    mNeurToBin = NeurToBinMatrix(neuralLocsOct,nFFT,audioFs)

# window shape
    win = .5-.5*np.cos(2*np.pi*np.arange(0,nFFT)/(nFFT-1))
    
#%% other auxilliary variables
    
    playOverAvgRatio = np.round(tPlay/tAvg).astype(int)
    shLen = nFFT+(playOverAvgRatio-1)*nAvg
    stateHolder = np.zeros(shLen.astype(int))
    sH2 = np.zeros(shLen.astype(int))
    sH3 = np.zeros((nCarriers,shLen.astype(int)))
    shli = np.int(0)
    
#    random phase
#    phs = 2*np.pi*np.random.rand(np.floor(nFFT/2).astype(int))
    
    phsDat = sio.loadmat('phs.mat')  # load predefined random phase for comparison
    phs = phsDat['phs'][:,0]    
    dphi = 2*np.pi*np.arange(1,np.floor(nFFT/2)+1)*nAvg/nFFT
    
    audioPwr = np.zeros((nNeuralLocs,blkSize+1))
    audioPwr1 = np.zeros((nNeuralLocs,blkSize+1))
    nTaps = 128
    
    M = np.interp(neuralLocsOct,elecFreqOct,MCLmuA)
    M[neuralLocsOct<elecFreqOct[0]] = MCLmuA[0]
    M[neuralLocsOct>elecFreqOct[nElec-1]] = MCLmuA[nElec-1]
    
    
    T = np.interp(neuralLocsOct,elecFreqOct,TmuA)
    T[neuralLocsOct<elecFreqOct[0]] = TmuA[0]
    T[neuralLocsOct>elecFreqOct[nElec-1]] = TmuA[nElec-1]
    
    normRamp = np.multiply(charge2EF.T,1/(M-T)).T
    normOffset = T/(M-T)

    testAudioOut = np.array([])
    testAudioOut2 = np.array([])
    toneEnvsOut = np.empty((nCarriers,1))
    elData [elData < 0 ] = 0
    
#%% Generate tone complex
    nBlocks = (nFFT/2*(np.floor(elData.shape[1]/blkSize+1))).astype(int)-1
    tones = np.zeros((nBlocks,nCarriers))
    toneFreqs = generate_cfs(20,20000,nCarriers)
    t = np.arange(nBlocks)/audioFs 
    
    for toneNum in range(nCarriers):
        tones[:,toneNum] = np.sin(2.*np.pi*toneFreqs[toneNum]*t+phs[toneNum])   # random phase
#        tones[:,toneNum] = np.sin(2.*np.pi*toneFreqs[toneNum]*t)               # sine phase
    
    toneCmplx = np.sum(tones,axis=1)   
    spectHolder = np.zeros(((nFFT/2).astype(int),np.floor(elData.shape[1]/blkSize).astype(int)),dtype=complex)
    interpSpect = np.zeros((nCarriers,np.floor(elData.shape[1]/blkSize).astype(int)),dtype=complex)
#%% electrode data cleaning??
#    for iChan in np.arange(0,elData.shape[0]):
#        for iTime in np.arange(1,elData.shape[1]):
#            if elData[iChan,iTime-1] > 5 and elData[iChan,iTime] > 5:
#                elData[iChan,iTime-1] = max((elData[iChan,iTime-1],elData[iChan,iTime]))
#                elData[iChan,iTime] = 0
                
    fftFreqs = np.arange(1,np.floor(nFFT/2)+1)*audioFs/nFFT
    fftFreqsOct = np.log2(fftFreqs)
#%% Loop TODO: Try to split or optimize double loop for speed
    
audioOut = NeurToAudio(elData,normRamp,normOffset,mNeurToBin,audioPwr,alpha,nAvg,playOverAvgRatio,blkSize,nFFT)
#%% Play audio file
    if playAudio:
        output = np.float32(audioOut)/np.max(audioOut)
#        output = np.float32(testAudioOut)/np.max(testAudioOut)
        p = pa.PyAudio()
        nChan = 1
        stream = p.open(format=pa.paFloat32,
                        channels=nChan,
                        rate=audioFs.astype(int),
                        output=True,
                        output_device_index = 3
                        )
#        inData = audiodata.astype(np.float32).tostring()
        outData = output.astype(np.float32).tostring()
#        stream.write(inData)
        stream.write(outData)
        stream.close()
#%% Return wavdata
     
     
        
    return audioOut