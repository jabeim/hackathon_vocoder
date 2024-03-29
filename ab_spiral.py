# -*- coding: utf-8 -*-
"""
Created on Mon Apr 22 11:29:54 2019

@author: beimx004
"""
import numpy as np
import scipy as sp
import scipy.io as sio
from mat4py import loadmat as loadmatstruct
from scipy.io.wavfile import write as wavwrite
from vocoder_tools import ActivityToPower, NeurToBinMatrix, generate_cfs



def vocoder(fileName,**kwargs):
    
    nCarriers = kwargs.get('nCarriers',20)   
    elecFreqs = kwargs.get('elecFreqs',None)
    spread = kwargs.get('spread',None)
    neuralLocsOct = kwargs.get('neuralLocsOct',None)
    nNeuralLocs = kwargs.get('nNeuralLocs',300)
    MCLmuA = kwargs.get('MCLmuA',None)
    TmuA = kwargs.get('TmuA',None)
    tAvg = kwargs.get('tAvg',.005)
    audioFs = kwargs.get('audioFs',48000)
    tPlay = kwargs.get('tPlay',None)
    tauEnvMS = kwargs.get('tauEnvMS',10)
    nl =kwargs.get('nl',8)
    resistorValue = kwargs.get('resistorVal',2)
    outFileName = kwargs.get('outFileName','Hackathon_scope_demo/'+fileName+'hybrid_voc.wav')
    
    
#%% Load .matfile of electrode recording and format data    
#    matData = sio.loadmat(fileName)
    
    matData = loadmatstruct('Hackathon_scope_demo/'+fileName+'.scope')
    matData = matData['S']
    electrodeAmp = np.array(matData['electrodeAmp'])
    
#    resistorValue = 2
    nElec = electrodeAmp.shape[0]-1
    captFs = float(matData['SampleRate'])
    captTs = 1/captFs
    scaletoMuA = 1000/resistorValue
    elData = np.flipud(electrodeAmp[1:,:])*scaletoMuA
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
        spreadFile = 'spread.mat'
        spread = sio.loadmat(spreadFile)
    else: # This seciont may need reindexing if the actual spread mat data is passed through, for now let use the spread.mat data
        elecPlacement = spread['elecPlacement']
        
# Create octave location of neural populations
    if neuralLocsOct is None:
        neuralLocsOct = np.append(
                np.log2(np.linspace(150,850,40)),
                np.linspace(np.log2(870),np.log2(8000),260)
                )
        
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
        if (type(MCLmuA) == int) or (type(MCLmuA)== float):
            MCLmuA = np.ones(nElec)*MCLmuA*1.2            
        elif (type(MCLmuA) == 'numpy.ndarray') and (MCLmuA.size == nElec):
            MCLmuA = MCLmuA * 1.2
        else:
            raise ValueError('Wrong number of M levels!')
            
    if TmuA is None:
        TmuA = 50*np.ones(nElec)
    else:
        if (type(TmuA) == int) or (type(TmuA)== float):
            TmuA = np.ones(nElec)*TmuA                   
        elif (type(TmuA) == 'numpy.ndarray') and (TmuA.size == nElec):
            TmuA = TmuA
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

# matrix to map neural activity to FFT bin frequencies
# here we call another function
    mNeurToBin = NeurToBinMatrix(neuralLocsOct,nFFT,audioFs)

# window shape
    win = .5-.5*np.cos(2*np.pi*np.arange(0,nFFT)/(nFFT-1))
    
# %% other auxilliary variables
       
#    random phase
    phs = 2*np.pi*np.random.rand(np.floor(nFFT/2).astype(int))
    # prededfined random phase
#    phsDat = sio.loadmat('phs.mat')  # load predefined random phase for comparison
#    phs = phsDat['phs'][:,0]    
    dphi = 2*np.pi*np.arange(1,np.floor(nFFT/2)+1)*nAvg/nFFT
    
    audioPwr = np.zeros((nNeuralLocs,blkSize+1))
    
    M = np.interp(neuralLocsOct,elecFreqOct,MCLmuA)
    M[neuralLocsOct<elecFreqOct[0]] = MCLmuA[0]
    M[neuralLocsOct>elecFreqOct[nElec-1]] = MCLmuA[nElec-1]
    
    
    T = np.interp(neuralLocsOct,elecFreqOct,TmuA)
    T[neuralLocsOct<elecFreqOct[0]] = TmuA[0]
    T[neuralLocsOct>elecFreqOct[nElec-1]] = TmuA[nElec-1]
    
    normRamp = np.multiply(charge2EF.T,1/(M-T)).T
    normOffset = T/(M-T)

    elData [elData < 0 ] = 0
    
# Generate tone complex
    nBlocks = (nFFT/2*(np.floor(elData.shape[1]/blkSize+1))).astype(int)-1
    tones = np.zeros((nBlocks,nCarriers))
    toneFreqs = generate_cfs(20,20000,nCarriers)
    t = np.arange(nBlocks)/audioFs 
    
    for toneNum in range(nCarriers):
        tones[:,toneNum] = np.sin(2.*np.pi*toneFreqs[toneNum]*t+phs[toneNum])   # random phase
#        tones[:,toneNum] = np.sin(2.*np.pi*toneFreqs[toneNum]*t)               # sine phase
      
    interpSpect = np.zeros((nCarriers,np.floor(elData.shape[1]/blkSize).astype(int)),dtype=complex)
# electrode data cleaning??
    for iChan in np.arange(0,elData.shape[0]):
        for iTime in np.arange(1,elData.shape[1]):
            if elData[iChan,iTime-1] > 5 and elData[iChan,iTime] > 5:
                elData[iChan,iTime-1] = max((elData[iChan,iTime-1],elData[iChan,iTime]))
                elData[iChan,iTime] = 0
                
    fftFreqs = np.arange(1,np.floor(nFFT/2)+1)*audioFs/nFFT
#%% Loop TODO: Try to split or optimize double loop for speed
    for blkNumber in range(1,(np.floor(elData.shape[1]/blkSize).astype(int))+1):
        # charge to electric field
        timeIdx = np.arange((blkNumber-1)*blkSize+1,blkNumber*blkSize+1,dtype=int)-1
        efData = np.dot(normRamp,elData[:,timeIdx])
        
        efData = (efData.T-normOffset).T
        electricField = np.maximum(0,efData)
        
        # Normalized EF to neural activity
#        nl = 5    
#        electricField = electricField/ 0.4
        activity = np.maximum(0,np.minimum(np.exp(-nl+nl*electricField),1)-np.exp(-nl))/(1-np.exp(-nl))
        
#        Neural activity to audio power       
        audioPwr = ActivityToPower(alpha,activity,audioPwr,blkSize)  # JIT optimized inner loop
        
        
        # Average energy
        energy = np.sum(audioPwr,axis = 1)/mAvg        
        spect = np.multiply(np.dot(mNeurToBin,energy),np.exp(1j*phs))       # this is normally overlap-add synthesized, to match interpolation try changing timpoints of the main blk loop to match the overlap add times (-nFFT/2)
        
        # interpolate spectrum across frequency 
        fMagInt = sp.interpolate.interp1d(fftFreqs,np.abs(spect),fill_value = 'extrapolate')
        fPhaseInt = sp.interpolate.interp1d(fftFreqs,np.angle(spect),fill_value = 'extrapolate')
        
        #calculate tone 
        toneMags = fMagInt(toneFreqs)
        tonePhases = fPhaseInt(toneFreqs)        
        interpSpect[:,blkNumber-1] = np.multiply(toneMags,np.exp(1j*tonePhases))
        
#%% Method 3 interpolated spectral envelope filtering
    
    specVec = np.arange(blkNumber)*nFFT/2
    newTimeVec = np.arange(nBlocks-(nFFT/2-1))
    interpSpect2 = np.zeros((len(toneFreqs),len(newTimeVec)),dtype=complex)
    modTones = np.zeros(interpSpect2.shape)
    for freq in range(len(toneFreqs)):  
        fEnvMag = sp.interpolate.interp1d(specVec,np.abs(interpSpect[freq,:]),fill_value = 'extrapolate')
        fEnvPhs = sp.interpolate.interp1d(specVec,np.angle(interpSpect[freq,:]),fill_value = 'extrapolate')       
        tEnvMag = fEnvMag(newTimeVec)
        tEnvPhs = fEnvPhs(newTimeVec)       
        interpSpect2[freq,:] = tEnvMag*np.exp(1j*tEnvPhs)
        modTones[freq,:] = tones[:-(nFFT/2-1).astype(int),freq]*np.abs(interpSpect2[freq,:])
       
    audioOut = np.sum(modTones,axis=0)

#    audioNorm = audioOut
#    wavData = (audioNorm*(2**32-1)).astype(np.int32) 
#    wavwrite(outFileName,audioFs.astype(int),wavData)
# Return wavdata
     
           
    return(audioOut,audioFs)