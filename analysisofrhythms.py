#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:54:37 2018

@author: kylie
"""

import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.pyplot import xlabel, ylabel, plot, show, title, xlim, ylim
from matplotlib import rcParams

""" 
random noise
x = np.random.normal(0,1,500)
y = np.random.normal(0,1,500)
"""

T = 2
dt = 0.001
samples = T / dt
t = np.arange(0, T, dt)
time = int(samples)

x = np.zeros(time)
y = np.zeros(time)

for i in range(0,time):
    x[i] = np.cos(2*np.pi*10*i)
    y[i] = np.sin(2*np.pi*10*i)
    x = x + x[i]
    y = y + y[i]
"""
Nx = x.shape[0]  # Define the total number of data points
Ny = y.shape[0]
df = 1/T #define freq resolution
fNQ = dt/2 #define Nyquist freq
"""

#Plotting Data
plt.figure()
plt.plot(x)
plt.plot(y)
xlabel('Time [s]')  # Label the time axis
ylabel('Voltage')  # ... and the voltage axis
title ('Data vs. Time')
plt.show()
"""
#Power for X
mn = x.mean()  # Compute the mean of the data
vr = x.var()   # Compute the variance of the data
sd = x.std()   # Compute the standard deviation of the data

xf = np.fft.fft(x - x.mean())  # Compute Fourier transform of x
Sxx = 2 * dt ** 2 / T * (xf * np.conj(xf))  # Compute spectrum
Sxx = Sxx[:int(len(x) / 2)]  # Ignore negative frequencies
fNQ = 1 / dt / 2  # Determine Nyquist frequency
faxis = np.arange(0,fNQ,df)  # Construct frequency axis

#Power for Y
mn = y.mean()  # Compute the mean of the data
vr = y.var()   # Compute the variance of the data
sd = y.std()   # Compute the standard deviation of the data

yf = np.fft.fft(y - y.mean())  # Compute Fourier transform of x
Sxy = 2 * dt ** 2 / T * (yf * np.conj(yf))  # Compute spectrum
Sxy = Sxy[:int(len(y) / 2)]  # Ignore negative frequencies
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


#Cross Covariance
x1 = x - np.mean(x)		# Define one time series,
y1 = y - np.mean(y)		# ... and another.
xc=1/Nx*np.correlate(x1,y1,2)	        # ... and compute their cross covariance.
lags = np.arange(-Nx+1,Nx)            # Create a lag axis,
plot(lags*dt,xc)					# ... and plot the cross covariance vs lags in time.
xlim([-0.2, 0.2])
xlabel('Lag [s]')					#... with axes labelled.
ylabel('Cross covariance');
      
#Coherence
K = np.shape(x)[0]			        # Define the number of trials.
N = np.shape(x)[0]                     # Define number of points in each trial.
f = np.fft.rfftfreq(N, dt)             # Define a frequency axis
Sxx = np.zeros([K,int(N/2+1)])		    # Create variables to save the spectra,
Syy = np.zeros([K,int(N/2+1)])
Sxy = np.zeros([K,int(N/2+1)], dtype=complex)
for k in range(K):			                      # For each trial,
    x=x[k]-np.mean(x[k])                    # Get the data from each electrode,
    y=y[k]-np.mean(y[k])
    xf  = np.fft.rfft(x-np.mean(x))               # ... compute Fourier transform,
    yf  = np.fft.rfft(y-np.mean(y))
    Sxx[k] = 2*dt**2/T *np.real(xf*np.conj(xf)) # ... and compute the spectra.
    Syy[k] = 2*dt**2/T *np.real(yf*np.conj(yf))
    Sxy[k] = 2*dt**2/T *       (xf*np.conj(yf))

Sxx = np.mean(Sxx,0)		                      # Average the spectra across trials,
Syy = np.mean(Syy,0)
Sxy = np.mean(Sxy,0)

cohr = np.abs(Sxy) / (np.sqrt(Sxx) * np.sqrt(Syy))# ... and compute the coherence.

plot(f, cohr);		                              # Plot coherence vs frequency,
xlim([0, 50])			                          # ... in chosen frequency range,
ylim([0, 1])                                      # ... with y-axis scaled,
xlabel('Frequency [Hz]')                          # ... and with axes labelled.
ylabel('Coherence');
"""