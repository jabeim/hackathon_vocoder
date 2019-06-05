# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 11:54:01 2019

@author: Jbeim
"""
import numpy as np
import scipy as sp


def NeurToBinMatrix(neuralLocsOct,nFFT,Fs):
    
    
#%%    
    fGrid = np.arange(0,np.floor(nFFT/2)+1)*(Fs/nFFT)
    fBinsOct = np.log2(fGrid[1:])
    
    binCountPerOct = np.divide(1,np.diff(fBinsOct))
    x = np.ones((2,np.floor(nFFT/2).astype(int)-1))
    x[1,:] = np.arange(1,np.floor(nFFT/2),1)
##    x = np.append(a1,a2,axis=1)
#    coef = np.linalg.solve(x,binCountPerOct)
#    scl = coef[0]+coef[1]*10**(neuralLocsOct/20)
    
    nNeuralLocs = len(neuralLocsOct)
    mNeurToBin = np.zeros((np.floor(nFFT/2).astype(int),nNeuralLocs))
    
    I = np.zeros(nNeuralLocs)
    for k in range(len(neuralLocsOct)):
        tmp = np.abs(fBinsOct-neuralLocsOct[k])
        I[k] = np.argmin(tmp)
        mNeurToBin[I[k].astype(int),k] = 1
        
    pFN = 'preemph.mat'
    emph = sp.io.loadmat(pFN)
    
    I = np.argmax(emph['emphDb'])
    emph['emphDb'][I+1:] = emph['emphDb'][I]
    emphDb = -emph['emphDb']
    emphDb= emphDb-emphDb[0]
    
    scl = np.interp(
            np.arange(1,np.floor(nFFT/2)+1)*Fs/nFFT,
            np.append(0,emph['emphF']),
            np.append(0,emphDb)
            )
    mNeurToBin = np.multiply(mNeurToBin.T,10**(scl/20))
    mNeurToBin = np.nan_to_num(mNeurToBin).T
    
    return mNeurToBin
#%%

def generate_cfs(lo, hi, n_bands):
    """
    Generates a series of 'bands' frequencies in Hz, linearely distributed
    on an ERB scale between the frequencies 'lo' and 'hi' (in Hz).
    These would are the centre frequencies (on an ERB scale) of the bands
    specifications made by 'generate_bands' with the same arguments
    """
    density = n_bands / (hz2erb(hi) - hz2erb(lo))
    bands = []
    for i in np.arange(1, n_bands + 1):
        bands.append(erb2hz(hz2erb(lo) + (i - 0.5) / density))
    return bands


def erb2hz(erb):
    """
    Convert equivalent rectangular bandwidth (ERB) to Hertz.
    """
    tmp = np.exp((erb - 43.) / 11.17)
    return (0.312 - 14.675 * tmp) / (tmp - 1.0) * 1000.

def hz2erb(hz):
    """
    Convert Hertz to equivalent rectangular bandwidth (ERB).
    """
    return 11.17 * np.log((hz + 312.) / (hz + 14675.)) + 43.