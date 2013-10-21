# -*- coding: utf-8 -*-
"""
Created on Tue Jun 25 12:52:32 2013

@author: andy
"""
import os
path = os.environ['BNNPATH']
import sys
sys.path.append(path)

from Layer import Softmax_Layer,Gaussian_Layer,Sigmoid_Layer,Tanh_Layer

import numpy as np
import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
import pycuda.cumath as cumath

import scikits.cuda.linalg as linalg
import scikits.cuda.misc as cumisc
linalg.init()

class BNN:
    def __init__(self,X,Y,layers,init_sd=1.0,precision=np.float32):
        self.X = gpuarray.to_gpu(X.astype(precision).copy())
        self.Y = gpuarray.to_gpu(Y.astype(precision).copy())
        self.precision = precision
                
        #Compile kernels 
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())
        
        self.layers = layers
        self.n_classes = Y.shape[1]
        self.N = len(X)
        
        ID = 0
        #Create the network, layer by layer
        for i in range(0,len(layers)):
            layers[i].ID = ID
            ID += 1
        self.num_layers = len(self.layers)
    
    ## MUST CALL FEED_FORWARD BEFORE USING THIS FUNCTION
    def updateAllGradients(self,print_timing=False):
        ##Compute gradient and do back-prop
        top_layer = self.layers[-1]
        bp = top_layer.updateGradient(self.Y,self.layers[-2].outputs,print_timing)
        layer_ids = np.linspace(len(self.layers)-2,0, len(self.layers)-1)
        layer_ids = layer_ids.astype(np.int)
        for i in layer_ids:
            l = self.layers[i]
            if i > 0:
                bp = l.updateGradient(bp,self.layers[i-1].outputs,print_timing)
            else:
                bp = l.updateGradient(bp,self.X,print_timing)
    
    def init_all_momentum(self):       
        for i in range(0,(len(self.layers))):
            self.layers[i].initializeMomentum()
    
    def get_total_k(self):    
        k = 0.0
        for i in range(0,len(self.layers)):
            k += self.layers[i].getTotalKineticEnergy()
        return k
    
    def feed_forward(self):
        for i in range(0,self.num_layers):
            layer = self.layers[i]
            if i == 0:
                inputs = self.X
            else:
                inputs = self.layers[i-1].outputs
            layer.updateOutputs(inputs)
    
    def updateAllHyperParams(self):
        for i in range(0,self.num_layers):
            layer = self.layers[i]
            layer.prior.updatePriorVals(layer.weights,layer.biases)
    
    ## MUST CALL FEED_FORWARD BEFORE USING THIS FUNCTION
    def log_like_val(self):
        return self.layers[-1].get_log_like_val(self.Y)
    
    ### THIS IS WRONG ROUNDING DOES NOT DO THE RIGHT THING
    def predict(self,_newX,_newY,predict_class=True):
        X_copy = self.X.get()
        Y_copy = self.Y.get()
        self.X = gpuarray.to_gpu(_newX.astype(self.precision))
        self.Y = gpuarray.to_gpu(_newY.astype(self.precision))
        self.feed_forward()
        preds = self.layers[-1].outputs.get()
        if predict_class:
            #FIX THIS PART
            preds = np.round(preds)
        #restore previous X and Y variables
        self.X = gpuarray.to_gpu(X_copy.astype(self.precision))
        self.Y = gpuarray.to_gpu(Y_copy.astype(self.precision))
        return preds
    
    def getPosteriorPredictions(self,_newX,_newY,predict_class=False):
        X_copy = self.X.get()
        Y_copy = self.Y.get()
        self.X = gpuarray.to_gpu(_newX.astype(self.precision))
        self.feed_forward()
        post_preds = list()
        for i in range(0,len(self.layers[0].posterior_weights)):        
            for j in range(0,self.num_layers):
                layer = self.layers[j]
                layer.setWeights(gpuarray.to_gpu(layer.posterior_weights[i].astype(self.precision)))
                layer.setBiases(gpuarray.to_gpu(layer.posterior_biases[i].astype(self.precision)))
            self.feed_forward()
            preds = self.layers[-1].outputs.get()
            if predict_class:
                preds = np.round(preds)
            post_preds.append(preds)
        #restore previous X and Y variables
        self.X = gpuarray.to_gpu(X_copy.astype(self.precision))
        self.Y = gpuarray.to_gpu(Y_copy.astype(self.precision))
        return post_preds
    
    def getTrainAccuracy(self):
        self.feed_forward()
        accuracy = 0.0
        #if self.layer_types[-1] == 'softmax':
        preds = (self.layers[-1].outputs.get())
        Y_cpu = self.Y.get()
        errors = 0.0
        for i in range(0,len(preds)):
            errors += 1.0-Y_cpu[i,preds[i].argmax()]    
        accuracy = 1.0 - errors/len(preds)
        
        return accuracy
    
    ## MUST CALL FEED_FORWARD BEFORE USING THIS FUNCTION
    def posterior_kernel_val(self):
        val = 0.0
        for i in range(0,len(self.layers)):
            l = self.layers[i]            
            val += l.prior.getPriorDensityValue(l.weights,l.biases)
        val += self.log_like_val()
        return val            
    
    def getMemoryStatus(self):
        (free,total)=cuda.mem_get_info()
        print("Global memory occupancy:%f%% free"%(free*100/total))    
    
