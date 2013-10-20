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
    def __init__(self,layer_sizes,X,Y,prior_params,layer_types,prior_structure='default',init_sd=1.0,magic_numbers=True,precision=np.float32):
        self.layer_sizes = layer_sizes
        self.X = gpuarray.to_gpu(X.astype(precision).copy())
        self.Y = gpuarray.to_gpu(Y.astype(precision).copy())
        self.precision = precision
        self.layer_types = layer_types       
        #Compile kernels 
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())
        
        #Setup default prior structure
        if prior_structure == 'default':
            prior_structure = list()
            for i in range(0,len(layer_sizes)):
                if i == 0:                
                    prior_structure.append('ARD')
                else:
                    prior_structure.append('normal_unit')
        self.layers = list()
        self.n_classes = layer_sizes[-1]
        self.N = len(X)
        self.prior_params = prior_params.astype(self.precision)
        prior_type = prior_structure[-1]
        #Set up the top layer
        if layer_types[-1] == 'softmax':
            top_layer = Softmax_Layer(self.n_classes,layer_sizes[-2],self.N,prior_type,self.prior_params[-1],init_sd=init_sd)
        elif layer_types[-1] == 'gaussian':
            top_layer = Gaussian_Layer(self.n_classes,layer_sizes[-2],self.N,prior_type,self.prior_params[-1],init_sd=init_sd)
        else:
            print 'Layer type ' + layer_types[-1] + ' is not currently implemented.'
            raise NotImplementedError
        ID = 0
        #Create the network, layer by layer
        for i in range(1,(len(layer_sizes)-1)):
            prior_type = prior_structure[i-1]
            if layer_types[i-1] == 'tanh':
                layer = Tanh_Layer(layer_sizes[i],layer_sizes[i-1],self.N,prior_type,self.prior_params[i-1],ID,init_sd=init_sd,magic_numbers=magic_numbers)
            elif layer_types[i-1] == 'sig':
                layer = Sigmoid_Layer(layer_sizes[i],layer_sizes[i-1],self.N,prior_type,self.prior_params[i-1],ID,init_sd=init_sd)
            else:
                print 'Layer type ' + layer_types[i-1] + ' is not currently implemented.'
                raise NotImplementedError
            ID += 1
            self.layers.append(layer)
        self.layers.append(top_layer)
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
        if self.layer_types[-1] == 'softmax':
            preds = (self.layers[-1].outputs.get())
            Y_cpu = self.Y.get()
            errors = 0.0
            for i in range(0,len(preds)):
                errors += 1.0-Y_cpu[i,preds[i].argmax()]
            
            accuracy = 1.0 - errors/len(preds)
        elif self.layer_types[-1] == 'gaussian':
            rmse = 0.0
            diff = (self.Y - self.layers[-1].outputs).get()
            rmse = np.sqrt((diff**2).mean())
            accuracy = rmse
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
    
