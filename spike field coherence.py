#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 13:27:05 2018

@author: kylie
"""
import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.pyplot import xlabel, ylabel, plot, title, xlim, ylim,legend

f = 10                         # define sinusoid oscillating at freq f
T = 1                           # define the total duration of the data
fs = 1000                        # define sampling freq. in Hz
trials = 100                    # define number of trials
dt = T / fs                     # time step
samples = T / dt                # number of samples
df = 1/T                        # freq resolution
fNQ = dt/2                      # Nyquist freq
time = np.arange(0,T,dt)

E1 = []                         # vector to store data from data 1 
S1 = []

for i in range(0,trials):      # for 100 trials...
    E1sin = np.zeros(fs) 
    spikes = np.random.binomial(1,0.08,1000)       
    for t in range(0,fs):      #for each sample...
        E1sin[t] = np.sin(((2*np.pi*t*f)/fs)) #generate sinusoid
        if E1sin[t] < 0:
            spikes[t] = 0           #LPF-locked spikes
        train = spikes
    E1 = E1 + [E1sin]
    S1 = S1 + [train]
    
"""
plt.figure()
plt.plot(E1sin)
plt.plot(train)
plt.show() 
"""
K = np.shape(E1)[0]
N = np.shape(E1sin)[0]
SYY = np.zeros(int(N/2+1))
SNN = np.zeros(int(N/2+1))
SYN = np.zeros(int(N/2+1), dtype=complex)

for k in np.arange(K):
    yf = np.fft.rfft((E1[k]-np.mean(E1[k])) *np.hanning(N))    # Hanning taper the field,
    nf = np.fft.rfft((S1[k]-np.mean(S1[k])))           # ... but do not taper the spikes.
    SYY = SYY + ( np.real( yf*np.conj(yf) ) )/K                  # Field spectrum
    SNN = SNN + ( np.real( nf*np.conj(nf) ) )/K                  # Spike spectrum
    SYN = SYN + (          yf*np.conj(nf)   )/K                  # Cross spectrum

cohr = np.real(SYN*np.conj(SYN)) / SYY / SNN                     # Coherence
f = np.fft.rfftfreq(N, dt)                                       # Frequency axis for plotting


plt.figure()
plt.subplot(1,3,1)         # Plot the spike spectrum.
plot(f,SNN)
plt.xlim([0, 100])
xlabel('Frequency [Hz]')
ylabel('Power [Hz]')
title('SNN')
plt.show()

plt.figure()
plt.subplot(1,3,2)        # Plot the field spectrum,
#T = t[-1]
plot(f,dt**2/T*SYY)       # ... with the standard scaling.
plt.xlim([0, 100])
xlabel('Frequency [Hz]')
ylabel('Power [Hz]')
title('SYY')
plt.show()

plt.figure()
plt.subplot(1,3,3)        # Plot the coherence
plot(f,cohr)
plt.xlim([0, 100])
plt.ylim([0, 1])
xlabel('Frequency [Hz]')
ylabel('Coherence')
plt.show()