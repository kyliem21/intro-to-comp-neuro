#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 30 12:19:37 2018

@author: kylie
"""

import scipy.io as sio
data = sio.loadmat('ECoG-1.mat')
data.keys()

E1 = data['E1']
E2 = data['E2']
t = data['t'][0]

#visualizing data one trial at a time
from matplotlib.pyplot import plot, xlabel, ylabel, xlim, ylim
plot(t,E1[0,:], 'b')            # Plot the data from the first trial of one electrode,
plot(t,E2[0,:], 'r')            # ... and the first trial of the other electrode.
xlabel('Time [s]');
ylabel('Voltage [mV]');

#visualizing data multiple trials      
import matplotlib.pyplot as plt
import numpy as np
K = np.shape(E1)[0]                              #Get the number of trials,
plt.imshow(E1,                                   #... and show the image,
           extent=[np.min(t), np.max(t), K, 1],  #... with meaningful axes,
           aspect=0.01)                          #... and a nice aspect ratio.
xlabel('Time [s]')
ylabel('Trial #');

#plotting trial averaged autocovariance    
dt = t[1]-t[0]			                # Define the sampling interval.
K = np.shape(E1)[0]			            # Define the number of trials.
N = np.shape(E1)[1]                     # Define number of points in each trial.
ac = np.zeros([2*N-1])                  # Declare empty vector for autocov.

for index,trial in enumerate(E1):		# For each trial,
    x = trial-np.mean(trial)			# ... subtract the mean,
    ac0 =1/N*np.correlate(x,x,2)	    # ... compute autocovar,
    ac += ac0/K;		                # ... and add to total, scaled by 1/K.
    
lags = np.arange(-N+1,N)                # Create a lag axis,
plot(lags*dt,ac)                        # ... and plot the result.
xlim([-0.2, 0.2])
xlabel('Lag [s]')
ylabel('Autocovariance');
  
      
#crosscovariance between two electrodes      
x = E1[0,:] - np.mean(E1[0,:])		# Define one time series,
y = E2[0,:] - np.mean(E2[0,:])		# ... and another.
xc=1/N*np.correlate(x,y,2)	        # ... and compute their cross covariance.
lags = np.arange(-N+1,N)            # Create a lag axis,
plot(lags*dt,xc)					# ... and plot the cross covariance vs lags in time.
xlim([-0.2, 0.2])
xlabel('Lag [s]')					#... with axes labelled.
ylabel('Cross covariance');
      
      
#single trial cross covariance
XC = np.zeros([K,2*N-1])                    # Declare empty vector for cross cov.
for k in range(K):			                # For each trial,
    x = E1[k,:]-np.mean(E1[k,:])			# ...get data from one electrode,
    y = E2[k,:]-np.mean(E2[k,:])			# ...and the other electrode,
    XC[k,:]=1/N*np.correlate(x,y,2)         # ...compute cross covariance.
plt.subplot(2,1,1)
plot(lags*dt,np.mean(XC,0))					# Plot cross covariance vs lags in time.
xlim([-0.2, 0.2])
ylim([-0.6, 0.6])
xlabel('Lag [s]')					        #... with axes labelled.
plt.title('Trial-averaged cross covariance');

plt.subplot(2,1,2)
for k in range(4):
    plot(lags*dt,XC[k,:])                   # Also, plot the single-trial cross-covariance for 4 trials
xlim([-0.2, 0.2])
ylim([-0.6, 0.6])
xlabel('Lag [s]')
plt.title('Example single-trial cross covariance');

plt.subplots_adjust(hspace=1)               # Space out the subplots.

                   
#trial averaged spectrum
T = t[-1]                                         # Get the total duration of the recording.
Sxx = np.zeros([K,int(N/2+1)])		              # Create variable to store each spectrum.
for k,x in enumerate(E1):				          # For each trial,
    xf  = np.fft.rfft(x-np.mean(x)) 	          # ... compute the Fourier transform,
    Sxx[k,:] = 2*dt**2/T *np.real(xf*np.conj(xf)) # ... and compute the spectrum.
    
f = np.fft.rfftfreq(N, dt)                        # Define a frequency axis

plot(f,10*np.log10(np.mean(Sxx,0)))               # Plot average spectrum over trials in decibels vs frequency,
xlim([0, 100])				                      # ... in select frequency range,
ylim([-50, 0])                                    # ... in select power range,
xlabel('Frequency [Hz]')	                      # ... with axes labelled.
ylabel('Power [ mV^2/Hz]')

plot(f,10*np.log10(Sxx[0,:]), 'r');               # Also, for reference, plot spectrum from the first trial.


#computing coherence
Sxx = np.zeros([K,int(N/2+1)])		              # Create variables to save the spectra,
Syy = np.zeros([K,int(N/2+1)])
Sxy = np.zeros([K,int(N/2+1)], dtype=complex)
for k in range(K):			                      # For each trial,
    x=E1[k,:]-np.mean(E1[k,:])                    # Get the data from each electrode,
    y=E2[k,:]-np.mean(E2[k,:])
    xf  = np.fft.rfft(x-np.mean(x))               # ... compute Fourier transform,
    yf  = np.fft.rfft(y-np.mean(y))
    Sxx[k,:] = 2*dt**2/T *np.real(xf*np.conj(xf)) # ... and compute the spectra.
    Syy[k,:] = 2*dt**2/T *np.real(yf*np.conj(yf))
    Sxy[k,:] = 2*dt**2/T *       (xf*np.conj(yf))

Sxx = np.mean(Sxx,0)		                      # Average the spectra across trials,
Syy = np.mean(Syy,0)
Sxy = np.mean(Sxy,0)

cohr = np.abs(Sxy) / (np.sqrt(Sxx) * np.sqrt(Syy))# ... and compute the coherence.

plot(f, cohr);		                              # Plot coherence vs frequency,
xlim([0, 50])			                          # ... in chosen frequency range,
ylim([0, 1])                                      # ... with y-axis scaled,
xlabel('Frequency [Hz]')                          # ... and with axes labelled.
ylabel('Coherence');
      

#visualizing phase difference across trials
j8 = np.where(f==8)[0][0]	         # Determine index j for frequency 8 Hz.
j24= np.where(f==24)[0][0]	         # Determine index j for frequency 24 Hz.

phi8=np.zeros(K)		             # Variables to hold phase differences.
phi24=np.zeros(K)

for k in range(K):			         # For each trial, compute the cross spectrum. 
    x=E1[k,:]-np.mean(E1[k,:])       # Get the data from each electrode,
    y=E2[k,:]-np.mean(E2[k,:])
    xf  = np.fft.rfft(x-np.mean(x))  # ... compute the Fourier transform,
    yf  = np.fft.rfft(y-np.mean(y))
    Sxy = 2*dt**2/T *(xf*np.conj(yf))# ... and the cross-spectrum,
    phi8[k]  = np.angle(Sxy[j8])	 # ... and the phases.
    phi24[k] = np.angle(Sxy[j24])

plt.subplot(1,2,1)                   # Plot the distributions of phases.
plt.hist(phi8, bins=20, range=[-np.pi, np.pi])
ylim([0, 40])
ylabel('Counts')
plt.title('Angles at 8 Hz')
plt.subplot(1,2,2)
plt.hist(phi24, bins=20, range=[-np.pi, np.pi])
ylim([0, 40])
plt.title('Angles at 24 Hz')
ylabel('Counts')
xlabel('Phase');