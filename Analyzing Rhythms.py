#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Analyzing rhythms of a single trial: cosine and sine functions
Created on Tue Nov 13 14:38:20 2018

@author: kylie
"""
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.pyplot import xlabel, ylabel, plot, show, title, xlim, ylim
from matplotlib import rcParams

T = 2
dt = 0.001
points = int(T/dt)
t = np.arange(0, T, dt)

x1 = np.cos(2*np.pi*10*t)
y1 = np.sin(2*np.pi*10*t)

Nx = x1.shape[0]  # Define the total number of data points
Ny = y1.shape[0]
df = 1/T #define freq resolution
fNQ = dt/2 #define Nyquist freq
K = np.shape(x1)[0]			        # Define the number of trials.
N = np.shape(x1)[0]                     # Define number of points in each trial.
f = np.fft.rfftfreq(N, dt)             # Define a frequency axis

#Plotting Data
plt.figure()
plt.plot(t,x1)
plt.plot(t,y1)
xlabel('Time [s]')  # Label the time axis
ylabel('Voltage')  # ... and the voltage axis	
title ('Data vs. Time')
plt.show()

#Power for X
mn = x1.mean()  # Compute the mean of the data
vr = x1.var()   # Compute the variance of the data
sd = x1.std()   # Compute the standard deviation of the data

xf = np.fft.fft(x1 - x1.mean())  # Compute Fourier transform of x
Sxx = 2 * dt ** 2 / T * (xf * np.conj(xf))  # Compute spectrum
Sxx = Sxx[:int(len(x1) / 2)]  # Ignore negative frequencies
fNQ = 1 / dt / 2  # Determine Nyquist frequency
faxis = np.arange(0,fNQ,df)  # Construct frequency axis

#Power for Y
mn = y1.mean()  # Compute the mean of the data
vr = y1.var()   # Compute the variance of the data
sd = y1.std()   # Compute the standard deviation of the data

yf = np.fft.fft(y1 - y1.mean())  # Compute Fourier transform of x
Sxy = 2 * dt ** 2 / T * (yf * np.conj(yf))  # Compute spectrum
Sxy = Sxy[:int(len(y1) / 2)]  # Ignore negative frequencies
fNQ = 1 / dt / 2  # Determine Nyquist frequency
faxis = np.arange(0,fNQ,df)  # Construct frequency axis

#Plot Power Spectrum for Both X and Y
plt.figure()
plt.plot(faxis, np.real(Sxx))  # Plot spectrum vs frequency
plt.plot(faxis, np.real(Sxy))  # Plot spectrum vs frequency
plt.xlim([0, 100])  # Select frequency range
xlabel('Frequency [Hz]')  # Label the axes
ylabel('Power [$\mu V^2$/Hz]')
title ('Power Spectrum')
plt.show() 


#Cross Covariance		              # For each trial,
x = x1[:]-np.mean(x1[:])			# ...get data from one electrode,
y = y1[:]-np.mean(y1[:])			# ...and the other electrode,
xc = 1/N*np.correlate(x,y,2)         # ...compute cross covariance.
lags = np.arange(-N+1,N)            # Create a lag axis,

plt.figure()
plot(lags*dt,xc)					# ... and plot the cross covariance vs lags in time.
xlim([-0.2, 0.2])
xlabel('Lag [s]')					#... with axes labelled.
ylabel('Cross covariance')
plt.show() 


#Coherence
Sxx = np.zeros([K,int(N/2+1)])		    # Create variables to save the spectra,
Syy = np.zeros([K,int(N/2+1)])
Sxy = np.zeros([K,int(N/2+1)], dtype=complex)
			                      
xf  = np.fft.rfft(x-np.mean(x))               # ... compute Fourier transform,
yf  = np.fft.rfft(y-np.mean(y))
Sxx = 2*dt**2/T *np.real(xf*np.conj(xf)) # ... and compute the spectra.
Syy = 2*dt**2/T *np.real(yf*np.conj(yf))
Sxy = 2*dt**2/T *       (xf*np.conj(yf))

Sxx = np.mean(Sxx,0)		                      # Average the spectra across trials,
Syy = np.mean(Syy,0)
Sxy = np.mean(Sxy,0)

cohr = np.abs(Sxy) / (np.sqrt(Sxx) * np.sqrt(Syy))# ... and compute the coherence.
cohr = np.full((1001), cohr)
f = np.fft.rfftfreq(N, dt) 

plt.figure()
plot(f, cohr);		                              # Plot coherence vs frequency,
xlim([0, 50])			                          # ... in chosen frequency range,
ylim([0, 1.2])                                      # ... with y-axis scaled,
xlabel('Frequency [Hz]')                          # ... and with axes labelled.
ylabel('Coherence')
plt.show()
