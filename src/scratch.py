# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 15:04:04 2013

@author: andy
"""


print 'Script start.'
import os
import sys
path = os.environ['BNNPATH']
sys.path.append(path)
from network import BNN
from Sampler import HMC_sampler
import numpy as np
import time as t
print 'Libraries imported successfully.'

dat = np.genfromtxt("/home/andy/Downloads/ForAndy_MethodComparison/DataFiles/Compare5.2L.OR3-5.4-6.6.txt",delimiter=2)
X = dat[:,1:90]
Y0 = dat[:,0]*-1 + 1
Y = np.transpose(np.vstack((Y0,dat[:,0])))


layer_sizes = np.array([X.shape[1],50,Y.shape[1]])
layer_types = ["sig","softmax"]

prior_structure = ["ARD","normal_unit"]
prior_params = np.array([1,1])
net = BNN(layer_sizes,X,Y,init_sd=0.01,layer_types=layer_types,prior_params=prior_params)
net.feed_forward()
net.updateAllGradients(True)

hmc = HMC_sampler(net,50,1e-4,scale=False)
hmc.simple_annealing_sim(200,200,eta=0.5,T0=10,epsFinal=1e-4,epsDecay=False,verbose=True)
hmc.getARDSummary()



preds = net.getPosteriorPredictions(X,Y)
mean_pred = np.zeros(Y.shape)
for i in range(0,len(preds)):
    mean_pred += preds[i]

mean_pred = mean_pred/len(preds)
errors = 0.0
for i in range(0,len(Y)):
    if (np.abs(Y - np.round(mean_pred[i])).sum()) > 0:
        errors += 1.0

1.0 - errors/len(Y)