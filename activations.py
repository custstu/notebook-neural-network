
# coding: utf-8

# In[1]:

import imp
import numpy as np
import utility
imp.reload(utility)


plot = utility.Plot(['incoming', 'other', 'outgoing'], (-5, 5), (-5, 5), 500)
plot.view(elev=0, azim=210)
incoming = plot.coords()


# In[2]:

outgoing = incoming
plot('Linear', outgoing[0])


# In[3]:

outgoing = 1 / (1 + np.exp(-incoming))
plot('Sigmoid', outgoing[0])


# In[4]:

outgoing = np.maximum(0, incoming)
plot('Relu', outgoing[0])


# In[5]:

exps = np.exp(incoming)
outgoing = exps / exps.sum(axis=0)
assert np.allclose(outgoing.sum(axis=0), 1)
plot('Softmax', outgoing[0])

