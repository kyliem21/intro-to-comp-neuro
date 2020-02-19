#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 19:10:52 2018

@author: kylie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 12 14:17:12 2018

@author: kylie
"""
"""
###Exercise 2: Generate 100 trials of 1 s data sampled at 500 Hz.
For each trial, set the initial phase of the sinusoid to a random value 
between 0 and 2 π. Repeat this procedure to create a second dataset, but in 
this case fix the initial phase of the sinusoid to π. Then compute the 
coherence between these two synthetic datasets. 
"""

import matplotlib.pyplot as plt
import numpy as np
import random
from matplotlib.pyplot import xlabel, ylabel, plot, title, xlim, ylim,legend


### Generate sinusoid oscillating at freq f: 1 s data sampled at 500 Hz
f = 10                         # define sinusoid oscillating at freq f
T = 1                           # define the total duration of the data
fs = 500                        # define sampling freq. in Hz
trials = 100                    # define number of trials
dt = T / fs                     # time step
samples = T / dt                # number of samples
df = 1/T                        # freq resolution
fNQ = dt/2                      # Nyquist freq
time = np.arange(0,T,dt)

E1 = []                         # vector to store data from data 1 (P0 of sinusoid is 0:2pi)
E2 = []                         # vector to store data from data 2 (P0 of sinusoid is pi)

"""
# General function for generating sinusoid
def genSine(f, fs, T):
    t = np.arange(0,T,dt)
    sinusoid = np.sin((2*np.pi*t*f)/fs))
    return sinusoid
"""
for i in range(0,trials):      # for 100 trials...
    E1sin = np.zeros(fs)        
    E2sin = np.zeros(fs)
    v = random.uniform(0,2) * np.pi
    shift = np.pi                  
    for t in range(0,fs):      #for each sample...
        E1sin[t] = np.sin(((2*np.pi*t*f)/fs)+v) + random.gauss(0,1) #generate sinusoid with phase shift v + gaussian noise
        E2sin[t] = np.sin(((2*np.pi*t*f)/fs)+shift) + random.gauss(0,1) # generate sinusoid with phase shift pi + gaussian noise
    E1 = E1 + [E1sin]
    E2 = E2 + [E2sin] 

# Option to visualize data  
plt.figure()
plot(time,E1sin,'blue',label='initial phase: random')
plot(time,E2sin,'red', label='initial phase: π')
xlabel('Time [s]')                          # ... and with axes labelled.
ylabel('Theoretical Voltage [mV]')
legend(loc='upper left')
title('Visualizing Generated Sinusoids')


### Computing Coherence
K = len(E1)			        # Define the number of trials. = 100
N = np.shape(E1sin)[0]                     # Define number of points in each trial. = 500
Sxx = np.zeros([K,int(N/2+1)])		    # Create variables to save the spectra,
Syy = np.zeros([K,int(N/2+1)])
Sxy = np.zeros([K,int(N/2+1)], dtype=complex)

for k in range(K):			                      # For each trial,
    x=E1[k]-np.mean(E1[k])                    # Get the data from each vector,
    y=E2[k]-np.mean(E2[k])
    xf  = np.fft.rfft(x-np.mean(x))               # ... compute Fourier transform,
    yf  = np.fft.rfft(y-np.mean(y))
    Sxx[k,:] = 2*dt**2/T *np.real(xf*np.conj(xf)) # ... and compute the spectra.
    Syy[k,:] = 2*dt**2/T *np.real(yf*np.conj(yf))
    Sxy[k,:] = 2*dt**2/T *       (xf*np.conj(yf))

Sxx = np.mean(Sxx,0)		                      # Average the spectra across trials,
Syy = np.mean(Syy,0)
Sxy = np.mean(Sxy,0)

cohr = np.abs(Sxy) / (np.sqrt(Sxx) * np.sqrt(Syy))# ... and compute the coherence.
freq = np.fft.rfftfreq(fs, dt)

###Plotting Coherence
plt.figure()           
plot(freq, cohr)	                              # Plot coherence vs frequency,
xlim([0, 80])			                          # ... in chosen frequency range,
ylim([0, 1])                                      # ... with y-axis scaled,
xlabel('Frequency [Hz]')                          # ... and with axes labelled.
ylabel('Coherence')
title('Example Coherence Between Sinusoids')
