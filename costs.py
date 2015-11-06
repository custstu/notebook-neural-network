
# coding: utf-8

# In[1]:

import imp
import numpy as np
import utility
imp.reload(utility)


plot = utility.Plot(['prediction', 'target', 'cost'], (0, 1), (0, 1), 500)
plot.view(elev=0, azim=120)
prediction, target = plot.coords()


# In[2]:

cost = (prediction - target) ** 2 / 2
plot('Squared Error', cost)


# In[3]:

epsilon = 1e-2
clipped = np.clip(prediction, epsilon, 1 - epsilon)
cost = target * np.log(clipped) + (1 - target) * np.log(1 - clipped)
cost = -cost
plot('Cross Entropy', cost)

