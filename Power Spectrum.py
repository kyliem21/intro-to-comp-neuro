#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 18 13:06:36 2018

@author: kylie
"""

# Prepare the modules and plot settings
import scipy.io as sio
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from matplotlib.pyplot import xlabel, ylabel, plot, show, title
from matplotlib import rcParams
rcParams['figure.figsize'] = (12,3)
data = sio.loadmat('EEG-4.mat')  # Load the EEG data
EEG = data['EEG'].reshape(-1)  # Extract the EEG variable
t = data['t'][0]  # ... and the t variable

x = EEG  # Relabel the data variable
dt = t[1] - t[0]  # Define the sampling interval
N = x.shape[0]  # Define the total number of data points
T = N * dt  # Define the total duration of the data
df = 1/T
FNQ = dt/2

print('dt=',dt,'sec')
print('T=',T,'sec')
print('df=', df, 'Hz')
print('Nyquist=',FNQ,'Hz')        

#Plot Data vs. Time
plot(t, EEG)  # Plot the data versus time
xlabel('Time [s]')  # Label the time axis
ylabel('Voltage [$\mu V$]')  # ... and the voltage axis
title ('Data vs. Time')
plt.autoscale(tight=True)  # Minimize white space
show()

mn = x.mean()  # Compute the mean of the data
vr = x.var()   # Compute the variance of the data
sd = x.std()   # Compute the standard deviation of the data

#Plot Autocovariance          
# Compute the lags for the full autocovariance vector
lags = np.arange(-len(x) + 1, len(x))
# ... and the autocov for L +/- 100 indices
ac = 1 / N * np.correlate(x - x.mean(), x - x.mean(), mode='full')
# Find the lags that are within 100 time steps
inds = np.abs(lags) <= 100
# ... and plot them
plot(lags[inds] * dt, ac[inds])
# ... with axes labelled
xlabel('Lag [s]')
ylabel('Autocovariance')
title ('Autocorrelation Plot')
show()


#Plot Spectrum vs. Frequency
xf = np.fft.fft(x - x.mean())  # Compute Fourier transform of x
Sxx = 2 * dt ** 2 / T * (xf * np.conj(xf))  # Compute spectrum
Sxx = Sxx[:int(len(x) / 2)]  # Ignore negative frequencies
df = 1 / T.max()  # Determine frequency resolution
fNQ = 1 / dt / 2  # Determine Nyquist frequency
faxis = np.arange(0,fNQ,df)  # Construct frequency axis

plt.plot(faxis, np.real(Sxx))  # Plot spectrum vs frequency
plt.xlim([0, 100])  # Select frequency range
xlabel('Frequency [Hz]')  # Label the axes
ylabel('Power [$\mu V^2$/Hz]')
title ('Power Spectrum')
plt.show() 

# Plot the spectrum in decibels.
plot(faxis, 10 * np.log10(Sxx / max(Sxx)))  
plt.xlim([0, 100])  # Select the frequency range.
plt.ylim([-60, 0])  # Select the decibel range.
xlabel('Frequency [Hz]')  # Label the axes.
ylabel('Power [dB]')
title ('Decibel Transform')
show()

"""
# Log-log scale
plt.semilogx(faxis, 10 * np.log10(Sxx / max(Sxx)))
plt.xlim([df, 100])  # Select frequency range
plt.ylim([-60, 0])   # ... and the decibel range.
xlabel('Frequency [Hz]')  # Label the axes.
ylabel('Power [dB]')
title ('Logarithmic Decibel Transform')
show()
"""

# Compute the spectrogram
Fs = 1 / dt               # Define the sampling frequency,
interval = int(Fs)        # ... the interval size,
overlap = int(Fs * 0.95)  # ... and the overlap intervals
f, t, Sxx = signal.spectrogram(
    EEG,               # Provide the signal,
    fs=Fs,             # ... the sampling frequency,
    nperseg=interval,  # ... the length of a segment,
    noverlap=overlap)  # ... the number of samples to overlap,
plt.pcolormesh(t, f, 10 * np.log10(Sxx),
               cmap='jet')  # Plot the result
plt.colorbar()         # ... with a color bar,
plt.ylim([0, 70])      # ... set the frequency range,
xlabel('Time [s]')     # ... and label the axes
ylabel('Frequency [Hz]')
title ('Spectrogram')
show()     