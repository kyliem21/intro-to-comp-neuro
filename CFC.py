#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 12:54:00 2018

@author: kylie
"""

import matplotlib.pyplot as plt
import numpy as np
import random 
from matplotlib.pyplot import xlabel, ylabel, plot, title, xlim, ylim,legend
from scipy.io import loadmat
from scipy import signal

### Generate sinusoid oscillating at freq f: 1 s data sampled at 500 Hz
f1 = 5                          # define sinusoid oscillating at low freq 
f2 = 40                         # define sinusoid oscillating at high freq
ampl = 2                        # define amplitude of low freq signal
T = 1                           # define the total duration of the data
fs = 500                        # define sampling freq. in Hz
dt = T / fs                     # time step
samples = T / dt                # number of samples
df = 1/T                        # freq resolution
fNQ = 1 / dt / 2                     # Nyquist freq

x = np.zeros(fs)
y = np.zeros(fs)
sinusoid = np.zeros(fs)

for t in range(0,fs):
    x[t] = ampl*np.sin((2*np.pi*t*f1)/fs)
    y[t] = np.sin((2*np.pi*t*f2)/fs)
    x = x + x[t]
    y = y + y[t]   
    if x[t] > 0:
        sinusoid[t] = x[t] + y[t] + random.gauss(0,0.1)
    else: 
        sinusoid[t] = x[t] + random.gauss(0,0.1)


Wn = [4,7]                          # Set the passband [4-7] Hz,
n = 100                              # ... and filter order,
b = signal.firwin(n, Wn, nyq=fNQ, pass_zero=False, window='hamming');
Vlo = signal.filtfilt(b, 1, sinusoid)    # ... and apply it to the data.

Wn = [30,70]                      # Set the passband [30-70] Hz,
n = 100                             # ... and filter order,
b = signal.firwin(n, Wn, nyq=fNQ, pass_zero=False, window='hamming');
Vhi = signal.filtfilt(b, 1, sinusoid)    # ... and apply it to the data.


phi = np.angle(signal.hilbert(Vlo))  # Compute phase of low-freq signal
amp = abs(signal.hilbert(Vhi))       # Compute amplitude of high-freq signal

p_bins = np.arange(-np.pi,np.pi,0.1) # To compute CFC, define phase bins,
a_mean = np.zeros(np.size(p_bins)-1) # ... variable to hold the amplitude,
p_mean = np.zeros(np.size(p_bins)-1) # ... and variable to hold the phase.
for k in range(np.size(p_bins)-1):      # For each phase bin,
    pL = p_bins[k]					    #... get lower phase limit,
    pR = p_bins[k+1]				    #... get upper phase limit.
    indices=(phi>=pL) & (phi<pR)	    #Find phases falling in this bin,
    a_mean[k] = np.mean(amp[indices])	#... compute mean amplitude,
    p_mean[k] = np.mean([pL, pR])		#... save center phase.

plt.plot(p_mean, a_mean)                #Plot the phase versus amplitude,
plt.ylabel('High-frequency amplitude')  #... with axes labeled.
plt.xlabel('Low-frequency phase')
plt.title('CFC')
