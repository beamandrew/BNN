# -*- coding: utf-8 -*-
"""
Created on Wed Jun 26 14:00:53 2013

@author: andy
"""
import sys
path = '/home/albeam/BNN/code/src/OO net'
sys.path.append(path)
from Prior import *

from abc import ABCMeta, abstractmethod

import numpy as np
import time as t

import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule
from pycuda.elementwise import ElementwiseKernel
import pycuda.curandom as curandom
import pycuda.cumath as cumath

import scikits.cuda.linalg as linalg
linalg.init()


class Layer:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def updateOutputs(self,inputs): pass
    
    @abstractmethod
    def updateGradient(self,previous_grad,include_prior): pass
    
    @abstractmethod
    def setWeights(self,new_weights): pass
            
    @abstractmethod
    def setBiases(self,new_biases): pass
    
    @abstractmethod
    def addPosteriorWeightSample(self,new_sample): pass
    
    @abstractmethod
    def addPosteriorBiasSample(self,new_sample): pass
       
    @abstractmethod
    def getNumUnits(self): pass
    
    @abstractmethod
    def getWeights(self): pass
    
    @abstractmethod
    def getBiases(self): pass
    
    @abstractmethod
    def updateMomenta(self,persist=0.0): pass
    
    @abstractmethod
    def getTotalKineticEnergy(self): pass
    
    @abstractmethod 
    def scaleMomentum(self): pass

    @abstractmethod 
    def scaleStepSize(self): pass

    @abstractmethod 
    def setPrior(self): pass
    
    @abstractmethod
    def get_log_like_val(self): pass
    
    def add_bias(self,output,bias):
        grid1 = (output.shape[1]+32-1)/32
        grid2 = (output.shape[0]+32-1)/32
        M = np.int32(output.shape[0])       
        N = np.int32(output.shape[1])
        #Adds bias to output gpuarray object        
        self.add_bias_kernel(output, bias, M, N, block=(32,32,1),grid=( grid1,grid2) )   
     

class Softmax_Layer(Layer):
    def __init__(self,n_classes,n_incoming,N,init_sd=0.1,precision=np.float32):
        self.n_incoming = n_incoming
        self.N = N
        w = np.random.normal(0,init_sd,(self.n_incoming,n_classes))
        b = np.random.normal(0,init_sd,(1,n_classes))
        self.weights = gpuarray.to_gpu(w.copy().astype(precision))
        self.gW = gpuarray.empty_like(self.weights)
                        
        self.biases = gpuarray.to_gpu(b.copy().astype(precision))
        self.gB = gpuarray.empty_like(self.biases)
        
        # Prior and ID are set later        
        self.prior = -1
        self.ID = -1
        
        #Set up momentum variables for HMC sampler
        self.pW = gpuarray.to_gpu(np.random.normal(0,1,self.gW.shape))
        self.pB = gpuarray.to_gpu(np.random.normal(0,1,self.gB.shape))    
        
        #Store stepsizes for each parameter
        self.epsW = gpuarray.zeros(self.weights.shape,precision) + 1.0
        self.epsB = gpuarray.zeros(self.biases.shape,precision) + 1.0
        
        self.n_classes = n_classes
        self.n_incoming = n_incoming
        
        self.N = N
        self.outputs = gpuarray.zeros((self.N,self.n_classes),precision)        
        
        self.precision = precision
                
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())
        self.softmax_kernel = kernels.get_function("softmax")
        self.add_bias_kernel = kernels.get_function("add_bias")
        
        self.rng = curandom.XORWOWRandomNumberGenerator()
        
        ##Initialize posterior weights
        self.posterior_weights = list()
        self.posterior_biases = list()
        
        self.eps_tol = 1e-10
        
    def setPrior(self,prior):
        self.prior = prior
    
    def updateOutputs(self,inputs): 
        self.outputs = linalg.dot(inputs,self.weights)
        #Add bias to gpuarray
        self.add_bias(self.outputs,self.biases)
        #Perform sigmoid transformation        
        self.softmax(self.outputs)
    
    def softmax(self,output):
        grid2 = (output.shape[0]+32-1)/32
        M = np.int32(output.shape[0])       
        N = np.int32(output.shape[1])
        #Perform softmax using GPU      
        self.softmax_kernel(output, M, N, block=(1,32,1),grid=( 1,grid2) )    
    
    def get_log_like_val(self,Y):
        return np.min( ((gpuarray.sum( (cumath.log(self.outputs+self.eps_tol)*Y) )).get(), 10^20 ) )    
    
    ## Updates the gradient information for all of the parameters in this layer.
    ## Returns the back-prop signal to be sent to the next layer
    def updateGradient(self,Y,inputs,print_timing=False,include_prior=True):
        if print_timing:
            t0 =  t.time()
            t_run = t.time()
        diff = Y-self.outputs
        if print_timing:
            t_diff = t.time() - t_run
            t_run = t.time()
        #self.gW = linalg.dot(linalg.transpose(inputs),diff)
        self.gW = linalg.dot(inputs,diff,transa='T')
        if print_timing:
            t_dot = t.time() - t_run
            t_run = t.time()
        ones = gpuarray.to_gpu(np.ones((1,self.N)).astype(self.precision))
        if print_timing:
            t_ones = t.time() - t_run
            t_run = t.time()
        self.gB = linalg.dot(ones,diff)
        if print_timing:
            t_sum_bias = t.time() - t_run
            t_run = t.time()
        if print_timing:
            t1 = t.time()
            t0_prior = t.time()
        if include_prior:
            self.prior.updateWeightGradient(self.weights,self.gW)
            self.prior.updateBiasGradient(self.biases,self.gB)
        if print_timing:
            t1_prior = t.time()
            print 'Total time for gradient update in softmax layer ' + str(t1-t0)
            print 'Time for prior update in softmax layer ' + str(t1_prior - t0_prior)
            print 'Time for Y-outputs ' + str(t_diff)
            print 'Time for inputs-diff dot-prod ' + str(t_dot)
            print 'Time to create ones vector ' + str(t_ones)
            print 'Time for diff-ones dot-prod ' + str(t_sum_bias)
        return linalg.dot(diff,self.weights,transb='T')
        #return linalg.dot(diff,linalg.transpose(self.weights))
    
    def updateMomenta(self,persist=0.0):
        self.rng.fill_normal(self.pW)
        self.rng.fill_normal(self.pB)
    
    ##Compute gradient on CPU
    ## Used to check for possible loss of precision on GPU
    def updateGradient_CPU(self,Y,inputs,print_timing=False):
        if print_timing:
            t0 =  t.time()
        Y_cpu = Y.get().astype(np.float64)
        outputs_cpu = (self.outputs.get()).astype(np.float64)
        inputs_cpu = (inputs.get()).astype(np.float64)
        weights_cpu = (self.weights.get()).astype(np.float64)
 
        diff = Y_cpu - outputs_cpu
        gW_cpu = np.dot(np.transpose(inputs_cpu),diff)
        gB_cpu = diff.sum(axis=0)
        if print_timing:
            t1 = t.time()
            print 'Time for gradient update in softmax layer ' + str(t1-t0)
        bp_signal = np.dot(diff,np.transpose(weights_cpu))               
        grad = list()
        grad.append(gW_cpu)
        grad.append(gB_cpu)
        grad.append(bp_signal)
        return grad              
    
    def setWeights(self,new_weights):
        self.weights = new_weights
        
    def setBiases(self,new_biases):
        self.biases = new_biases
    
    def getNumUnits(self):
        return self.n_classes
            
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases
    
    def addPosteriorWeightSample(self,new_sample):
        self.posterior_weights.append(new_sample)
    
    def addPosteriorBiasSample(self,new_sample):
        self.posterior_biases.append(new_sample)
    
    def getTotalKineticEnergy(self):
        return (self.pW.get()**2).sum() + (self.pB.get()**2).sum()
    
    def scaleMomentum(self):
        self.prior.scaleMomentum(self.pW,self.pB)
    
    def scaleStepSize(self):
        self.prior.scaleStepSize(self.epsW,self.epsB)
    
class Gaussian_Layer(Layer):
    def __init__(self,n_outputs,n_incoming,N,prior,init_sd=0.1,precision=np.float32):
        self.n_outputs = n_outputs
        self.n_incoming = n_incoming
        w = np.random.normal(0,init_sd,(self.n_incoming,self.n_outputs))
        b = np.random.normal(0,init_sd,(1,n_outputs))
        self.weights = gpuarray.to_gpu(w.copy().astype(precision))
        self.gW = gpuarray.empty_like(self.weights)
                        
        self.biases = gpuarray.to_gpu(b.copy().astype(precision))
        self.gB = gpuarray.empty_like(self.biases)
        self.prior = prior
                
        #Set up momentum variables for HMC sampler
        self.pW = gpuarray.to_gpu(np.random.normal(0,1,self.gW.shape))
        self.pB = gpuarray.to_gpu(np.random.normal(0,1,self.gB.shape))    
        
        self.N = N
        self.outputs = gpuarray.zeros((self.N,self.n_outputs),precision)        
        
        self.precision = precision
                
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())
        self.add_bias_kernel = kernels.get_function("add_bias")
        
        self.rng = curandom.XORWOWRandomNumberGenerator()
        
        ##Initialize posterior weights
        self.posterior_weights = list()
        self.posterior_biases = list()
    
    ##Linear output function
    def updateOutputs(self,inputs): 
        self.outputs = linalg.dot(inputs,self.weights)
        #Add bias to gpuarray
        self.add_bias(self.outputs,self.biases)
    
    def get_log_like_val(self,Y):
        prod = Y*self.outputs
        return np.min( (gpuarray.sum(prod).get(), 10^20) )
    
    
    ## Updates the gradient information for all of the parameters in this layer.
    ## Returns the back-prop signal to be sent to the next layer
    def updateGradient(self,Y,inputs,print_timing=False,include_prior=True):
        if print_timing:
            t0 =  t.time()
            t_run = t.time()
        diff = Y-self.outputs
        if print_timing:
            t_diff = t.time() - t_run
            t_run = t.time()
        #self.gW = linalg.dot(linalg.transpose(inputs),diff)
        self.gW = linalg.dot(inputs,diff,transa='T')
        if print_timing:
            t_dot = t.time() - t_run
            t_run = t.time()
        ones = gpuarray.to_gpu(np.ones((1,self.N)).astype(self.precision))
        if print_timing:
            t_ones = t.time() - t_run
            t_run = t.time()
        bias_diff = Y - self.outputs
        self.gB = linalg.dot(ones,bias_diff)
        if print_timing:
            t_sum_bias = t.time() - t_run
            t_run = t.time()
        if print_timing:
            t1 = t.time()
            t0_prior = t.time()
        if include_prior:
            self.prior.updateWeightGradient(self.weights,self.gW)
            if print_timing:
                t_weights = t.time() - t_run
                t_run = t.time()
            self.prior.updateBiasGradient(self.biases,self.gB)
        if print_timing:
            t1_prior = t.time()
            print 'Total time for gradient update in softmax layer ' + str(t1-t0)
            print 'Time for prior update in softmax layer ' + str(t1_prior - t0_prior)
            print 'Time for Y-outputs ' + str(t_diff)
            print 'Time for inputs-diff dot-prod ' + str(t_dot)
            print 'Time to create ones vector ' + str(t_ones)
            print 'Time for diff-ones dot-prod ' + str(t_sum_bias)
        return linalg.dot(diff,self.weights,transb='T')
        #return linalg.dot(diff,linalg.transpose(self.weights))
    
    def updateMomenta(self,persist=0.0):
        loc_pW = self.pW.get()*persist
        loc_pW = (loc_pW + np.sqrt((1-persist**2))*np.random.normal(size=loc_pW.shape)).astype(self.precision)
        
        self.pW = gpuarray.to_gpu(loc_pW)
        #self.rng.fill_normal(self.pW)
        
        loc_pB = self.pB.get()*persist
        loc_pB = (loc_pB + np.sqrt((1-persist**2))*np.random.normal(size=loc_pB.shape)).astype(self.precision)
        self.pB = gpuarray.to_gpu(loc_pB)
              
    
    def setWeights(self,new_weights):
        self.weights = new_weights
        
    def setBiases(self,new_biases):
        self.biases = new_biases
    
    def getNumUnits(self):
        return self.n_outputs
            
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases
    
    def addPosteriorWeightSample(self,new_sample):
        self.posterior_weights.append(new_sample)
    
    def addPosteriorBiasSample(self,new_sample):
        self.posterior_biases.append(new_sample)
    
    def getTotalKineticEnergy(self):
        return (self.pW.get()**2).sum() + (self.pB.get()**2).sum()
    
    def scaleMomentum(self):
        self.prior.scaleMomentum(self.pW,self.pB)
    
    def scaleStepSize(self):
        self.prior.scaleStepSize(self.epsW,self.epsB)

        
class Sigmoid_Layer(Layer):
    def __init__(self,n_units,n_incoming,N,init_sd=1.0,precision=np.float32):
        
        self.n_units = n_units
        self.n_incoming = n_incoming
        self.N = N
        w = np.random.normal(0,init_sd,(self.n_incoming,self.n_units))
        b = np.random.normal(0,init_sd,(1,n_units))
        
        self.weights = gpuarray.to_gpu(w.copy().astype(precision))
        self.gW = gpuarray.empty_like(self.weights)
        
        # Prior and ID must be set after creation
        self.prior = -1
        self.ID = -1
                
        self.biases = gpuarray.to_gpu(b.copy().astype(precision))
        self.gB = gpuarray.empty_like(self.biases)
            
        #Set up momentum variables for HMC sampler
        self.pW = gpuarray.to_gpu(np.random.normal(0,1,self.gW.shape))
        self.pB = gpuarray.to_gpu(np.random.normal(0,1,self.gB.shape))
        
        self.epsW = gpuarray.zeros(self.weights.shape,precision) + 1.0
        self.epsB = gpuarray.zeros(self.biases.shape,precision) + 1.0
    
        self.precision = precision
        self.outputs = gpuarray.zeros((self.N,self.n_units),precision)   
        
        #Define sigmoid function on GPU      
        self.sigmoid = ElementwiseKernel(
            "float *x",
            "x[i] = 1/(1+expf(-1*min(max(-10.0,x[i]),20.0)))",
            "sigmoid",preamble="#include <math.h>")
        #Compile kernels 
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_bias_kernel = kernels.get_function("add_bias")
        
        self.rng = curandom.XORWOWRandomNumberGenerator()
        
        ##Initialize posterior weights
        self.posterior_weights = list()
        self.posterior_biases = list()
    
    def setPrior(self,prior):
        self.prior = prior
    
    def setWeights(self,new_weights):
        self.weights = new_weights
        
    def setBiases(self,new_biases):
        self.biases = new_biases
        
    def updateGradient(self,bp_signal,inputs,print_timing=False,include_prior=True):
        if print_timing:
            print '' 
            t0 =  t.time()
            t_run = t.time()
        #back_prop = linalg.multiply(bp_signal,linalg.multiply(self.outputs,(1-self.outputs)))
        back_prop = bp_signal*self.outputs*(1-self.outputs)
        if print_timing:
            t_bp = t.time() - t_run
            t_run = t.time()
        self.gW = linalg.dot(inputs,back_prop,transa='T')
        if print_timing:
            t_dot = t.time() - t_run
            t_run = t.time()
        #self.gW = linalg.dot(linalg.transpose(inputs),back_prop)
        ones = gpuarray.to_gpu(np.ones((1,self.N)).astype(self.precision))
        if print_timing:
            t_ones = t.time() - t_run
            t_run = t.time()
        self.gB = linalg.dot(ones,back_prop)
        if print_timing:
            t_biases = t.time() - t_run
            t_run = t.time()
        if include_prior:
            self.prior.updateWeightGradient(self.weights,self.gW)
            if print_timing:
                t_weights = t.time() - t_run
                t_run = t.time()
            self.prior.updateBiasGradient(self.biases,self.gB)
        if print_timing:
            t_prior = t.time() - t_run
            print 'Total time for gradient update in hidden layer ' + str(self.ID) + ' '  + str(t.time()-t0)
            print 'Time to calculate backprop in hidden layer ' + str(self.ID) + ' '  + str(t_bp)
            print 'Time to calculate gradient for weights in hidden layer ' + str(self.ID) + ' '  + str(t_dot)
            print 'Time to allocate ones vector in hidden layer ' + str(self.ID) + ' '  + str(t_ones)
            print 'Time to biases gradient in hidden layer ' + str(self.ID) + ' '  + str(t_biases)
            print 'Time for prior update in hidden layer ' + str(self.ID) + ' '  + str(t_prior)
        if self.ID > 0:
            return linalg.dot(back_prop,gpuarray.to_gpu(self.weights.get().T.copy()))
            #return linalg.dot(back_prop,linalg.transpose(self.weights))
        else:
            return -1            
    
    def updateGradient_CPU(self,bp_signal,inputs,print_timing=False):
        
        if print_timing:
            t0 =  t.time()
        back_prop = linalg.multiply(bp_signal,linalg.multiply(self.outputs,(1-self.outputs)))
        self.gW = linalg.dot(linalg.transpose(inputs),back_prop)
        ones = gpuarray.to_gpu(np.ones((1,self.N)).astype(self.precision))
        self.gB = linalg.dot(ones,back_prop)
        if print_timing:
            t1 = t.time()
            t0_prior = t.time()
        self.prior.updateWeightGradient(self.weights,self.gW)
        self.prior.updateBiasGradient(self.biases,self.gB)
        if print_timing:
            t1_prior = t.time()
            print 'Time for gradient update in hidden layer ' + str(self.ID) + ' '  + str(t1-t0)
            print 'Time for prior update in softmax layer ' + str(self.ID) + ' '  + str(t1_prior - t0_prior)
        if self.ID > 0:
            return linalg.dot(back_prop,self.weights,transb='T')
        else:
            return -1            
    
    def updateOutputs(self,inputs):
        self.outputs = linalg.dot(inputs,self.weights)
        #Add bias to gpuarray
        self.add_bias(self.outputs,self.biases)
        #Perform sigmoid transformation        
        self.sigmoid(self.outputs)
    
    def updateMomenta(self,persist=0.0):
        loc_pW = self.pW.get()*persist
        loc_pW = (loc_pW + np.sqrt((1.0-persist**2))*np.random.normal(size=loc_pW.shape)).astype(self.precision)
        
        self.pW = gpuarray.to_gpu(loc_pW)
        #self.rng.fill_normal(self.pW)
        
        loc_pB = self.pB.get()*persist
        loc_pB = (loc_pB + np.sqrt((1.0-persist**2))*np.random.normal(size=loc_pB.shape)).astype(self.precision)
        self.pB = gpuarray.to_gpu(loc_pB)    
        
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases        
    
    def addPosteriorWeightSample(self,new_sample):
        self.posterior_weights.append(new_sample)
    
    def addPosteriorBiasSample(self,new_sample):
        self.posterior_biases.append(new_sample)
    
    def getNumUnits(self):
        return self.n_units
    
    def getTotalKineticEnergy(self):
        return (self.pW.get()**2).sum() + (self.pB.get()**2).sum()

    def scaleMomentum(self):
        self.prior.scaleMomentum(self.pW,self.pB)
    
    def scaleStepSize(self):
        self.prior.scaleStepSize(self.epsW,self.epsB)    
    
    def get_log_like_val(self,Y):
        raise NotImplementedError 

class Tanh_Layer(Layer):
    def __init__(self,n_units,n_incoming,N,init_sd=1.0,precision=np.float32,magic_numbers=False):
        
        self.n_units = n_units
        self.n_incoming = n_incoming
        self.N = N
        w = np.random.normal(0,init_sd,(self.n_incoming,self.n_units))
        b = np.random.normal(0,init_sd,(1,n_units))
        
        self.weights = gpuarray.to_gpu(w.copy().astype(precision))
        self.gW = gpuarray.empty_like(self.weights)
        
        # Prior and ID must be set after creation
        self.prior = -1
        self.ID = -1
                
        self.biases = gpuarray.to_gpu(b.copy().astype(precision))
        self.gB = gpuarray.empty_like(self.biases)
            
        #Set up momentum variables for HMC sampler
        self.pW = gpuarray.to_gpu(np.random.normal(0,1,self.gW.shape))
        self.pB = gpuarray.to_gpu(np.random.normal(0,1,self.gB.shape))
        
        self.epsW = gpuarray.zeros(self.weights.shape,precision) + 1.0
        self.epsB = gpuarray.zeros(self.biases.shape,precision) + 1.0        
        
        self.precision = precision
        self.outputs = gpuarray.zeros((self.N,self.n_units),precision)   
        
        self.magic_numbers = magic_numbers
        #Define tan_h function on GPU   
        if magic_numbers:
            self.tanh = ElementwiseKernel(
                "float *x",
                "x[i] = 1.7159 * tanh(2/3*x[i]);",
                "tan_h",preamble="#include <math.h>")
        else:
            self.tanh = ElementwiseKernel(
            "float *x",
            "x[i] = tanh(min(max(-10.0,x[i]),10.0));",
            "tan_h",preamble="#include <math.h>")
        #Compile kernels 
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_bias_kernel = kernels.get_function("add_bias")
        
        self.rng = curandom.XORWOWRandomNumberGenerator()
        
        ##Initialize posterior weights
        self.posterior_weights = list()
        self.posterior_biases = list()
    
    def setPrior(self,prior):
        self.prior = prior
    
    def setWeights(self,new_weights):
        self.weights = new_weights
        
    def setBiases(self,new_biases):
        self.biases = new_biases
        
    def updateGradient(self,bp_signal,inputs,print_timing=False,include_prior=True):
        if print_timing:
            print '' 
            t0 =  t.time()
            t_run = t.time()
        if self.magic_numbers:
            back_prop = bp_signal* 0.6667/1.7159 * (1.7159 - (self.outputs)*(1.7159 + self.outputs))
        else:
            back_prop = bp_signal*(1.0-(self.outputs*self.outputs))
        
        if print_timing:
            t_bp = t.time() - t_run
            t_run = t.time()
        self.gW = linalg.dot(inputs,back_prop,transa='T')
        if print_timing:
            t_dot = t.time() - t_run
            t_run = t.time()
        #self.gW = linalg.dot(linalg.transpose(inputs),back_prop)
        ones = gpuarray.to_gpu(np.ones((1,self.N)).astype(self.precision))
        if print_timing:
            t_ones = t.time() - t_run
            t_run = t.time()
        self.gB = linalg.dot(ones,back_prop)
        if print_timing:
            t_biases = t.time() - t_run
            t_run = t.time()
        if include_prior:
            self.prior.updateWeightGradient(self.weights,self.gW)
            if print_timing:
                t_weights = t.time() - t_run
                t_run = t.time()
            self.prior.updateBiasGradient(self.biases,self.gB)
        if print_timing:
            t_prior = t.time() - t_run
            print 'Total time for gradient update in hidden layer ' + str(self.ID) + ' '  + str(t.time()-t0)
            print 'Time to calculate backprop in hidden layer ' + str(self.ID) + ' '  + str(t_bp)
            print 'Time to calculate gradient for weights in hidden layer ' + str(self.ID) + ' '  + str(t_dot)
            print 'Time to allocate ones vector in hidden layer ' + str(self.ID) + ' '  + str(t_ones)
            print 'Time to biases gradient in hidden layer ' + str(self.ID) + ' '  + str(t_biases)
            print 'Time for prior update in hidden layer ' + str(self.ID) + ' '  + str(t_prior)
        if self.ID > 0:
            return linalg.dot(back_prop,self.weights,transb='T')
            #return linalg.dot(back_prop,linalg.transpose(self.weights))
        else:
            return -1            
    
    def updateOutputs(self,inputs):
        self.outputs = linalg.dot(inputs,self.weights)
        #Add bias to gpuarray
        self.add_bias(self.outputs,self.biases)
        #Perform sigmoid transformation        
        self.tanh(self.outputs)
    
    def updateMomenta(self,persist=0.0):
        loc_pW = self.pW.get()*persist
        loc_pW = (loc_pW + np.sqrt((1.0-persist**2))*np.random.normal(size=loc_pW.shape)).astype(self.precision)
        
        self.pW = gpuarray.to_gpu(loc_pW)
        #self.rng.fill_normal(self.pW)
        
        loc_pB = self.pB.get()*persist
        loc_pB = (loc_pB + np.sqrt((1.0-persist**2))*np.random.normal(size=loc_pB.shape)).astype(self.precision)
        self.pB = gpuarray.to_gpu(loc_pB) 
    
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases        
    
    def addPosteriorWeightSample(self,new_sample):
        self.posterior_weights.append(new_sample)
    
    def addPosteriorBiasSample(self,new_sample):
        self.posterior_biases.append(new_sample)
    
    def getNumUnits(self):
        return self.n_units
    
    def getTotalKineticEnergy(self):
        return (self.pW.get()**2).sum() + (self.pB.get()**2).sum()
    
    def scaleMomentum(self):
        self.prior.scaleMomentum(self.pW,self.pB)
    
    def get_log_like_val(self,Y):
        raise NotImplementedError 
    
    def scaleStepSize(self):
        self.prior.scaleStepSize(self.epsW,self.epsB)    

class Rectified_Linear_Layer(Layer):
    def __init__(self,n_units,n_incoming,N,init_sd=1.0,precision=np.float32):
        
        self.n_units = n_units
        self.n_incoming = n_incoming
        self.N = N
        w = np.random.normal(0,init_sd,(self.n_incoming,self.n_units))
        b = np.random.uniform(0,1,(1,n_units))
        
        self.weights = gpuarray.to_gpu(w.copy().astype(precision))
        self.gW = gpuarray.empty_like(self.weights)
        
        # Prior and ID must be set after creation
        self.prior = -1
        self.ID = -1
                
        self.biases = gpuarray.to_gpu(b.copy().astype(precision))
        self.gB = gpuarray.empty_like(self.biases)
            
        #Set up momentum variables for HMC sampler
        self.pW = gpuarray.to_gpu(np.random.normal(0,1,self.gW.shape))
        self.pB = gpuarray.to_gpu(np.random.normal(0,1,self.gB.shape))
        
        self.epsW = gpuarray.zeros(self.weights.shape,precision) + 1.0
        self.epsB = gpuarray.zeros(self.biases.shape,precision) + 1.0
    
        self.precision = precision
        self.outputs = gpuarray.zeros((self.N,self.n_units),precision)   
        
        #Define sigmoid function on GPU      
        self.rectifier = ElementwiseKernel(
            "float *x",
            "x[i] = max(0.0,x[i])",
            "rect",preamble="#include <math.h>")
        #Compile kernels 
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_bias_kernel = kernels.get_function("add_bias")
        self.rect_kernel = kernels.get_function("rect_grad")
        
        self.rng = curandom.XORWOWRandomNumberGenerator()
        
        ##Initialize posterior weights
        self.posterior_weights = list()
        self.posterior_biases = list()
    
    def setPrior(self,prior):
        self.prior = prior
    
    def setWeights(self,new_weights):
        self.weights = new_weights
        
    def setBiases(self,new_biases):
        self.biases = new_biases
        
    def updateGradient(self,bp_signal,inputs,print_timing=False,include_prior=True):
        if print_timing:
            print '' 
            t0 =  t.time()
            t_run = t.time()
        
        back_prop = self.rect_grad(bp_signal)
        if print_timing:
            t_bp = t.time() - t_run
            t_run = t.time()
        self.gW = linalg.dot(inputs,back_prop,transa='T')
        if print_timing:
            t_dot = t.time() - t_run
            t_run = t.time()
        #self.gW = linalg.dot(linalg.transpose(inputs),back_prop)
        ones = gpuarray.to_gpu(np.ones((1,self.N)).astype(self.precision))
        if print_timing:
            t_ones = t.time() - t_run
            t_run = t.time()
        self.gB = linalg.dot(ones,back_prop)
        if print_timing:
            t_biases = t.time() - t_run
            t_run = t.time()
        if include_prior:
            self.prior.updateWeightGradient(self.weights,self.gW)
            if print_timing:
                t_weights = t.time() - t_run
                t_run = t.time()
            self.prior.updateBiasGradient(self.biases,self.gB)
        if print_timing:
            t_prior = t.time() - t_run
            print 'Total time for gradient update in hidden layer ' + str(self.ID) + ' '  + str(t.time()-t0)
            print 'Time to calculate backprop in hidden layer ' + str(self.ID) + ' '  + str(t_bp)
            print 'Time to calculate gradient for weights in hidden layer ' + str(self.ID) + ' '  + str(t_dot)
            print 'Time to allocate ones vector in hidden layer ' + str(self.ID) + ' '  + str(t_ones)
            print 'Time to biases gradient in hidden layer ' + str(self.ID) + ' '  + str(t_biases)
            print 'Time for prior update in hidden layer ' + str(self.ID) + ' '  + str(t_prior)
        if self.ID > 0:
            return linalg.dot(back_prop,gpuarray.to_gpu(self.weights.get().T.copy()))
            #return linalg.dot(back_prop,linalg.transpose(self.weights))
        else:
            return -1
    
    def updateOutputs(self,inputs):
        self.outputs = linalg.dot(inputs,self.weights)
        #Add bias to gpuarray
        self.add_bias(self.outputs,self.biases)
        #Perform linear rectifier transformation        
        self.rectifier(self.outputs)
    
    def updateMomenta(self,persist=0.0):
        loc_pW = self.pW.get()*persist
        loc_pW = (loc_pW + np.sqrt((1.0-persist**2))*np.random.normal(size=loc_pW.shape)).astype(self.precision)
        
        self.pW = gpuarray.to_gpu(loc_pW)
        #self.rng.fill_normal(self.pW)
        
        loc_pB = self.pB.get()*persist
        loc_pB = (loc_pB + np.sqrt((1.0-persist**2))*np.random.normal(size=loc_pB.shape)).astype(self.precision)
        self.pB = gpuarray.to_gpu(loc_pB)    
        
    def getWeights(self):
        return self.weights
    
    def getBiases(self):
        return self.biases        
    
    def addPosteriorWeightSample(self,new_sample):
        self.posterior_weights.append(new_sample)
    
    def addPosteriorBiasSample(self,new_sample):
        self.posterior_biases.append(new_sample)
    
    def getNumUnits(self):
        return self.n_units
    
    def getTotalKineticEnergy(self):
        return (self.pW.get()**2).sum() + (self.pB.get()**2).sum()

    def scaleMomentum(self):
        self.prior.scaleMomentum(self.pW,self.pB)
    
    def scaleStepSize(self):
        self.prior.scaleStepSize(self.epsW,self.epsB)    
    
    # Multiply by rectifier gradient mask on GPU #
    def rect_grad(self,back_prop_signal):
        grid1 = (back_prop_signal.shape[1]+32-1)/32
        grid2 = (back_prop_signal.shape[0]+32-1)/32
        M = np.int32(back_prop_signal.shape[0])       
        N = np.int32(back_prop_signal.shape[1])
        #Adds bias to output gpuarray object        
        self.rect_kernel(back_prop_signal, self.outputs, M, N, block=(32,32,1),grid=( grid1,grid2) )   
        return back_prop_signal
    
    def get_log_like_val(self,Y):
        raise NotImplementedError 