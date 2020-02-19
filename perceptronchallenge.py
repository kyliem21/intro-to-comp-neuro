#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec  4 13:09:28 2018

@author: kylie
"""

# Building a perceptron
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import xlabel, ylabel, plot, title, xlim, ylim,legend
import scipy.io
mat = scipy.io.loadmat('training_and_testing_values.mat')

xset = mat['x_training']
yset = mat['y_training']
xtest = mat['x_testing']
ytest = mat['y_testing']
z = mat['correct_answer']
points = len(xset)

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

### Train perceptron
wx,wy,wb = 3*[0.5]                 # Choose initial values for the perceptron's weights

learning_constant = 0.01           # And, set the learning constant.

estimated_slope = np.zeros(points)    # Variables to hold the perceptron estimates.
estimated_intercept = np.zeros(points)
errorList = np.zeros(points)

for k in np.arange(2000):           # For 2000 iteractions,
    x = xset[k][0]         
    y = yset[k][0]
                                    # Step 1: Calculate known answer from dataset
    correct_answer = z[k][0]
    
                                    # Step 2. Ask perceptron to guess an answer.
    perceptron_guess = feedforward(x, y, wx, wy, wb)
    
                                    # Step 3. Compute the error.
    error = correct_answer - perceptron_guess
    if k > 0:
        errorList[k] = errorList[k-1] + abs(error)
                                    # Step 4. Adjust weights according to error.
    wx = wx + error*x*learning_constant
    wy = wy + error*y*learning_constant
    wb = wb + error*1*learning_constant
     
    estimated_slope[k] = -wx/wy       # Compute estimated slope from perceptron.
    estimated_intercept[k] = -wb/wy   # Compute estimated intercept from perceptron.

# Display the results! ------------------------------------------------------------------------
x_range = np.linspace(-2,4,200)                  # For a range of x-values,
fig, ax = plt.subplots()

for k in range(1,2000,100):                       # ... and plot some intermediate perceptron guess
    ax.plot(x_range, estimated_slope[k]*x_range+estimated_intercept[k], 'r')
                                                  # ... and plot the last perceptron guess
ax.plot(x_range, estimated_slope[-1]*x_range+estimated_intercept[-1], 'b')
plt.xlabel('x')
plt.ylabel('y')
plt.scatter(xtest,ytest)
plt.title('Perceptron final guess (blue), Test set (blue dots)')

plt.figure()
plt.plot(errorList)
plt.title('Cumulative error over training')
plt.xlabel('Training trial number')
plt.ylabel('Cumulative Error')
plt.show()