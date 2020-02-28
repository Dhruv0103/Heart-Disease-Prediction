#!/usr/bin/env python
# coding: utf-8

# In[7]:


import math
import pandas as pd
import numpy as np
import h5py
import matplotlib.pyplot as plt
from tensorflow.python.framework import ops
import pickle
import keras
from keras.models import Sequential
from keras.layers import Dense


# In[9]:


data = pd.read_csv('C:\\Users\\dhruv\\Desktop\\AI\\heart disease\\heart.csv', header=None)
print(data.shape)
print(data.shape)
data.head()


# In[11]:


Data = np.array(data)
x = np.array(Data[:,0:-1])
y = np.array(Data[:,-1])
print(x.shape[0])


# In[ ]:





# In[13]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=4)
#X_train, X_test, Y_train, Y_test=X_train.T, X_test.T, y_train.reshape(1,y_train.shape[0]), y_test.reshape(1,y_test.shape[0])

print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)


# In[15]:


def predict(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    Z1 = np.dot(W1, X) + b1
    A1 = np.maximum(0,Z1)
    Z2 = np.dot(W2, A1) + b2
    return Z2


# In[17]:


def score_model(probs, threshold):
    return np.array([1 if x > threshold else 0 for x in probs[:,0]])


# In[27]:


def create_model():
    model = Sequential()
    model.add(Dense(10,input_dim=13,activation='relu'))
    model.add(Dense(1, activation='relu'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

model = create_model()
model.summary()


# In[28]:


model.fit(X_train, y_train, validation_data=(X_test,y_test), epochs=2000)


# In[32]:


#regressor = LinearRegression()

#Fitting model with trainig data
#regressor.fit(X, y)

# Saving model to disk
pickle.dump(model, open('model.pkl','wb'))

# Loading model to compare the results
model = pickle.load(open('model.pkl','rb'))





