#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
from numpy import linalg 
from matplotlib import cm
from matplotlib import pyplot as plt
from scipy.optimize import minimize
from scipy.linalg import qr
from scipy import io
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from keras.layers import Dense, Conv2D, Flatten, MaxPool2D


y = np.linspace(-2,2,401) # spatial coordinate
Ny = np.size(y)

amp1 = 1
y01 = 0.5
sigmay1 = 0.6

amp2 = 1.2
y02 = -0.5
sigmay2 = 0.3

dt = 0.1
Nt = 101
tend = dt*(Nt-1)
t = np.linspace(0,tend,Nt) # time

omega1 = 1.3
omega2 = 4.1

v1 = amp1*np.exp(-((y-y01)**2)/(2*sigmay1**2))
v2 = amp2*np.exp(-((y-y02)**2)/(2*sigmay2**2))

X = np.zeros([Ny,Nt],dtype=complex)
for tt in range(Nt):
    X[:,tt] = v1*np.cos(omega1*t[tt])+v2*np.cos(omega2*t[tt]) 


# In[2]:


np.shape(X)


# In[2]:


# Compute SVD and plot against spatial values
u, s, vh = np.linalg.svd(X)
plt.plot(y, u[1,:])


# In[1]:


# QR Pivoting Method

r = 2
# Truncate u to have same dimensions
ur = u[:,:r]
Q,R,P = qr(np.transpose(ur),pivoting=True)
nspace, ntime = X.shape
timeIndex = 0

DataMean = np.mean(X,axis=1)
DataSub = X - np.tile(DataMean[:,np.newaxis],(1,ntime))

optimalSampleLocations = P[:r]
print(optimalSampleLocations)

# Do reconstruction with these measurements
OptimalMeasurements = DataSub[optimalSampleLocations,timeIndex]
# estimate coefficients of singular vectors from these measurements
cOpt = np.linalg.pinv(ur[optimalSampleLocations,:])@OptimalMeasurements

np.shape(DataSub)


# In[ ]:


ny, nx = X.shape
inputs = keras.Input(shape =(nspace,))

encodedDim = 4

# add a hidden layer to encode our data

HiddenLayer = layers.Dense(encodedDim,activation = 'linear')
encoding = HiddenLayer(inputs)

encoder = keras.Model(inputs,encoding, name="encoder")
print(encoder.summary())


# In[ ]:


# decoder
decoder_input = layers.Input(shape = (encodedDim,))
# add output layer
outputlayer = layers.Dense(nspace,activation = 'linear')
outputs = outputlayer(decoder_input)
decoder = keras.Model(decoder_input,outputs, name="decoder")
print(decoder.summary())


# In[ ]:


output_auto = decoder(encoder(inputs))
model = models.Model(inputs,output_auto)
print(model.summary())


# In[ ]:


# Training
TrainingInputs = np.transpose(DataSub) #mean-subtracted data
TrainingOutputs = np.transpose(DataSub)

model.compile(optimizer = 'adam', loss = 'mse')
history = model.fit(TrainingInputs,TrainingOutputs,batch_size=62,epochs= 200)


# In[ ]:


# Pass all of our data through the Autoencoder

TestData = np.transpose(DataSub)
ReconstructedData = model.predict(TestData)
ReconstructedDataT = np.transpose(ReconstructedData)


plt.plot(y, ReconstructedDataT)
plt.xlabel("Spatial Coordinate")
plt.ylabel("Data Values")
plt.title("Autoencoder (All Timesteps on single plot) ")


# In[ ]:


# Look at SVD

U,Sigma,VT = np.linalg.svd(DataSub,full_matrices=0)
# truncate SVD to the same dimension as the autoencoder
Ur = U[:,:encodedDim] 
Sigmar = np.diag(Sigma[:encodedDim])
VTr = VT[:encodedDim,:]


# In[ ]:


# Compare the total error from Autoencoder to SVD

ErrorSVD = np.linalg.norm(DataSub-Ur@Sigmar@VTr)
print(ErrorSVD/(nspace*ntime))

ErrorAE = np.linalg.norm(DataSub-ReconstructedDataT)
print(ErrorAE/(nspace*ntime))

# The different between the two is seemingly minimal


# In[ ]:


# Visualize SVD

plt.plot(y, u)
plt.xlabel("Spatial Coordinate")
plt.ylabel("Left Singular Values")
plt.title("SVD (All Timesteps on single plot) ")


# In[ ]:




