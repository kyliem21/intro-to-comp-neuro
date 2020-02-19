#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 12:43:59 2018

@author: kylie
"""

# Building a perceptron
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xlabel, ylabel, plot, title, xlim, ylim,legend

slope     = 2                      # Define the line with slope, 
intercept = 1                      # ... and intercept.

wx,wy,wb = 3*[0.5]                 # Choose initial values for the perceptron's weights

learning_constant = 0.01           # And, set the learning constant.

estimated_slope = np.zeros(2000)    # Variables to hold the perceptron estimates.
estimated_intercept = np.zeros(2000)


### Provide input and known answer
def known_answer(slope, intercept, x, y):       #perceptron function returning known answer

  yline = slope*x + intercept  # Compute y-value on the line.
  if y > yline:                # If the input y value is above the line,
      return 1                 # ... indicate this with output = 1;
  else:                        # Otherwise, the input y is below the line,
      return 0

### Ask perceptron to guess answer
def feedforward(x, y, wx, wy, wb):              #perceptron guessing answer

  # Fix the bias.
  bias = 1

  # Define the activity of the neuron, activity.
  activity = x*wx + y*wy + wb*bias
  
  # Apply the binary threshold,
  if activity > 0:
      return 1
  else:
      return 0
  
for k in np.arange(2000):           # For 2000 iteractions,
    x = np.random.randn(1)         # Choose a random (x,y) point in the plane
    y = np.random.randn(1)
                                    # Step 1: Calculate known answer.
    correct_answer = known_answer(slope, intercept, x, y)
    
                                    # Step 2. Ask perceptron to guess an answer.
    perceptron_guess = feedforward(x, y, wx, wy, wb)
    
                                    # Step 3. Compute the error.
    error = correct_answer - perceptron_guess
    
                                    # Step 4. Adjust weights according to error.
    wx = wx + error*x*learning_constant
    wy = wy + error*y*learning_constant
    wb = wb + error*1*learning_constant
     
    estimated_slope[k] = -wx/wy       # Compute estimated slope from perceptron.
    estimated_intercept[k] = -wb/wy   # Compute estimated intercept from perceptron.

# Display the results! ------------------------------------------------------------------------
x_range = np.linspace(-2,2,100)                  # For a range of x-values,
fig, ax = plt.subplots()
ax.plot(x_range, slope*x_range+intercept, 'k')    # ... plot the true line,

for k in range(1,2000,100):                       # ... and plot some intermediate perceptron guess
    ax.plot(x_range, estimated_slope[k]*x_range+estimated_intercept[k], 'r')
                                                  # ... and plot the last perceptron guess
ax.plot(x_range, estimated_slope[-1]*x_range+estimated_intercept[-1], 'b')
plt.xlabel('x')
plt.ylabel('y')
plt.title('Known answer (black), Perceptron final guess (blue)')
