# Hackathon Vocoder
This repo contains the working python code base for both the Advanced Bionics 'gold standard' and 'hybrid' vocoders. vocoder_demo.py will process a batch of audio files using both vocoders. Setting playAudio to True will allow users to listen to the final audio output of both vocoders to make easy comparisons of the differences in strategy.  The vocoder_demo script currently only calls the hybrid vocoder, but by uncommenting a few lines of code it can be used to compare the ab_standard and ab_spiral vocoders. vocoder_demo requires the contents of Hackathon_scope_demo, specifically the files beginning with MOD_MP_1. Hackathon_scope_demo also includes vocoder outputs saved as wavefiles for easy comparison.

Each vocoder returns a processed acoustic output as a numpy array as well as the sampling frequency required for playback. The vocoders can also save a .wav file of this output.

# standard_voc // ab_standard.py
Standard_voc is a python port from matlab of Advanced Bionics' gold standard vocoder. It currently takes a filename corresponding to a matlab .mat data file containing a nChannels x samples array of recorded electrode pulses. Using a simulation of current spread and neural activation a 'neural spectrum' is created and inverted back to the time domain to produce audio. 

This vocoder has been deprecated for the purposes of the contest and will not be part of the final release.

# hybrid_voc // ab_spiral.py
Hybrid_voc is the vocoder intended for use with the Hackathon competition. Like the standard_voc it requires the same electrode pulse data in a .mat (matlab) file format in order to create an output. This vocoder uses the simulated neural spectrum to modulate a series of configurable carrier tones, creating an acoustic output similar to a tone vocoder while having much greater stimulus density than traditional tone vocoders.

# supporting functions // vocoder_tools.py
vocoder_tools contains important subfunctions used in simulating the neural activity used in both the standard and hybrid vocoders. It also includes some important subfunctions used in optimizing the speed of data processing and in generating the output stimulus components of the hybrid vocoder.

# Dependencies
The hackathon_vocoder repo was developed using anaconda python 3.7.3

The combined set of functions is designed to operate with several important third party libraries some of which are included in anaconda python:

  numpy - fast data arrays/matrices
  scipy - some file io and linear interpolation functions

  numba jit - just in time compilation to speed up loops
  mat4py - loading in matlab data (this may end up deprecated once the processing chain is fully ported to python).
  pyaudio - playback of original and processed audio, only used in vocoder_demo

The vocoders currently use predefined preemphasis and current spread functions that are stored in the following files:
  spread.mat preemph.mat

The processing scripts will not run if the files are not in the appropriate directory structure.

