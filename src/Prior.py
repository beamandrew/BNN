# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:22:18 2013

@author: andy
"""
from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.stats import invgamma

import pycuda.driver as cuda
from pycuda import gpuarray
import pycuda.autoinit
from pycuda.compiler import SourceModule

import os
path = os.environ['BNNPATH']

class Prior:
    __metaclass__ = ABCMeta
    
    @abstractmethod
    def updateWeightGradient(self,weights,gW): pass
    
    @abstractmethod
    def updateBiasGradient(self,biases,gB): pass
    
    @abstractmethod
    def updatePriorVals(self): pass    
    
    @abstractmethod
    def getPriorDensityValue(self): pass
    
    @abstractmethod
    def scaleMomentum(self,pW,pB): pass
    
    @abstractmethod
    def scaleStepSize(self,epsW,epsB): pass
    

class ARD_Prior(Prior):
    def __init__(self,shape,scale,layer,precision=np.float32,init=100):
        self.precision = precision
        self.shape = shape
        self.scale = scale
        
        ##initialize with random draw
        #init_var = invgamma.rvs(shape,scale=scale,size=(1,layer.weights.shape[0])).astype(precision)
        init_var = (np.tile(init,reps=layer.weights.shape[0]).reshape(1,layer.weights.shape[0])).astype(precision)
        self.sW = gpuarray.to_gpu(init_var)
        
        init_var = invgamma.rvs(1.0,scale=1.0,size=(1,1)).astype(precision)
        self.sB = gpuarray.to_gpu(init_var)
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())
        self.add_prior_kernel = kernels.get_function("add_ARD_grad")
        self.add_prior_b_kernel = kernels.get_function("add_bias_grad")
        self.scale_momentum_kernel = kernels.get_function("scale_momentum_ARD")
        self.scale_stepsize_kernel = kernels.get_function("scale_stepsize_ARD")
        #self.updatePriorVals(layer.weights,layer.biases)
    
    def updateWeightGradient(self,weights,gW):
        grid1 = (gW.shape[1]+32-1)/32
        grid2 = (gW.shape[0]+32-1)/32
        M = np.int32(gW.shape[0])       
        N = np.int32(gW.shape[1])
        #Adds prior contribution to gradient on gpuarray object        
        self.add_prior_kernel(gW,weights, self.sW, M, N, block=(32,32,1),grid=( grid1,grid2) )
    
    def updateBiasGradient(self,biases,gB):
        grid1 = (gB.shape[1]+32-1)/32
        grid2 = (gB.shape[0]+32-1)/32
        M = np.int32(gB.shape[0])       
        N = np.int32(gB.shape[1])
        #Adds prior contribution to gradient on gpuarray object        
        self.add_prior_b_kernel(gB,biases, self.sB, M, N, block=(32,32,1),grid=( grid1,grid2) )
    
    ##Perform a Gibbs update of prior vals
    def updatePriorVals(self,weights,biases): 
        new_sW = np.zeros(self.sW.shape)
        weights_cpu = weights.get()
        n_w = np.float32(weights_cpu.shape[1])
        shape_new = (self.shape + n_w)/2.0
        for i in range(0,len(weights_cpu)):
            scale_new =  self.scale + ((weights_cpu[i])**2).sum()/2.0
            new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
            new_sW[0,i] = np.float32(new_val)
            #print 'New shape for feature ' + str(i+1) + ': ' + str(shape_new)
            #print 'New scale for feature ' + str(i+1) + ': ' + str(1.0/rate_new)
            #print 'New standard deviation for feature ' + str(i+1) + ': ' + str(new_val)
        
        self.sW = gpuarray.to_gpu(new_sW.astype(self.precision))
        
        ## Biases have common variance
        biases_cpu = biases.get()
        n_b = np.float32(biases.shape[1])
        shape_new = (self.shape + n_b)/2
        scale_new =  self.scale  + ((biases_cpu)**2).sum()/2.0
        new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
        new_sB = np.float32(new_val)
        self.sB = gpuarray.to_gpu(new_sB)
    
    def getPriorDensityValue(self,weights,biases):
        w = weights.get()
        b = biases.get()
        sW = np.tile(self.sW.get(),(weights.shape[1],1)).T
        sB = self.sB.get()
        val = -1*(w**2.0/(2.0*sW)).sum()
        val += -1*(b**2.0/(2.0*sB)).sum()
        return val
    
    def scaleMomentum(self,pW,pB):
        grid1 = (pW.shape[1]+32-1)/32
        grid2 = (pW.shape[0]+32-1)/32
        M = np.int32(pW.shape[0])       
        N = np.int32(pW.shape[1])
        self.scale_momentum_kernel(pW,self.sW,M,N,block=(32,32,1),grid=(grid1,grid2)) 
        
        grid1 = (pB.shape[1]+32-1)/32
        grid2 = (pB.shape[0]+32-1)/32
        M = np.int32(pB.shape[0])       
        N = np.int32(pB.shape[1])
        self.scale_momentum_kernel(pB,self.sB,M,N,block=(32,32,1),grid=(grid1,grid2)) 
    
    def scaleStepSize(self,epsW,epsB):
        grid1 = (epsW.shape[1]+32-1)/32
        grid2 = (epsW.shape[0]+32-1)/32
        M = np.int32(epsW.shape[0])       
        N = np.int32(epsW.shape[1])
        self.scale_stepsize_kernel(epsW,self.sW,M,N,block=(32,32,1),grid=(grid1,grid2)) 
        
        grid1 = (epsB.shape[1]+32-1)/32
        grid2 = (epsB.shape[0]+32-1)/32
        M = np.int32(epsB.shape[0])       
        N = np.int32(epsB.shape[1])
        self.scale_momentum_kernel(epsB,self.sB,M,N,block=(32,32,1),grid=(grid1,grid2)) 
    
    def getWeightSigmaMatrix(self,weights):
        sW = np.tile(self.sW.get(),(weights.shape[1],1)).T
        return sW
        
class Gaussian_Unit_Prior(Prior):
    def __init__(self,shape,scale,layer,precision=np.float32):
        self.precision = precision
        self.shape = shape
        self.scale = scale
        
        init_var = invgamma.rvs(1.0,scale=1.0,size=(1,layer.weights.shape[1])).astype(precision)
        self.sW = gpuarray.to_gpu(init_var)
        
        init_var = invgamma.rvs(1.0,scale=1.0,size=(1,1)).astype(precision)
        self.sB = gpuarray.to_gpu(init_var)
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_prior_w_kernel = kernels.get_function("add_gaussian_unit_grad")
        self.add_prior_b_kernel = kernels.get_function("add_bias_grad")
        self.scale_momentum_kernel = kernels.get_function("scale_momentum_normal_unit")
                
        ##initialize with random draw
        #self.updatePriorVals(layer.weights,layer.biases)
        
    def updateWeightGradient(self,weights,gW):
        grid1 = (gW.shape[1]+32-1)/32
        grid2 = (gW.shape[0]+32-1)/32
        M = np.int32(gW.shape[0])       
        N = np.int32(gW.shape[1])
        #Adds prior contribution to gradient on gpuarray object        
        self.add_prior_w_kernel(gW,weights, self.sW, M, N, block=(32,32,1),grid=( grid1,grid2) )
    
    def updateBiasGradient(self,biases,gB):
        grid1 = (gB.shape[1]+32-1)/32
        grid2 = (gB.shape[0]+32-1)/32
        M = np.int32(gB.shape[0])       
        N = np.int32(gB.shape[1])
        #Adds prior contribution to gradient on gpuarray object        
        self.add_prior_b_kernel(gB,biases, self.sB, M, N, block=(32,32,1),grid=( grid1,grid2) )
    
        ##Perform a Gibbs update of prior vals
    def updatePriorVals(self,weights,biases): 
        new_sW = np.zeros(self.sW.shape)
        weights_cpu = weights.get()
        n_w = np.float32(weights_cpu.shape[0])
        shape_new = (self.shape + n_w)/2
        for i in range(0,weights_cpu.shape[1]):
            scale_new =  self.scale + ((weights_cpu[:,i])**2).sum()/2.0
            new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
            new_sW[0,i] = np.float32(new_val)
        self.sW = gpuarray.to_gpu(new_sW.astype(self.precision))
        
         ## Biases have common variance
        biases_cpu = biases.get()
        n_b = np.float32(biases.shape[1])
        shape_new = (self.shape + n_b)/2
        scale_new =  self.scale + ((biases_cpu)**2).sum()/2.0
        new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
        new_sB = np.float32(new_val)
        self.sB = gpuarray.to_gpu(new_sB)
        
    def getPriorDensityValue(self,weights,biases):
        w = weights.get()
        b = biases.get()
        sW = np.tile(self.sW.get(),(w.shape[0],1))
        sB = self.sB.get()
        val = -1*(w**2/(2.0*sW)).sum()
        val += -1*(b**2/(2.0*sB)).sum()
            
    def scaleMomentum(self,pW,pB):
        grid1 = (pW.shape[1]+32-1)/32
        grid2 = (pW.shape[0]+32-1)/32
        M = np.int32(pW.shape[0])       
        N = np.int32(pW.shape[1])
        self.scale_momentum_kernel(pW,self.sW,M,N,block=(32,32,1),grid=(grid1,grid2)) 
        
        grid1 = (pB.shape[1]+32-1)/32
        grid2 = (pB.shape[0]+32-1)/32
        M = np.int32(pB.shape[0])       
        N = np.int32(pB.shape[1])
        self.scale_momentum_kernel(pB,self.sB,M,N,block=(32,32,1),grid=(grid1,grid2)) 

'''
All weights in the layer share a common variance hyper-parameter
'''
class Gaussian_Layer_Prior(Prior):
    def __init__(self,shape,scale,layer,precision=np.float32):
        self.precision = precision
        self.shape = shape
        self.scale = scale
        
        init_var = np.float32(100.0)
        self.sW = gpuarray.to_gpu(init_var)
        
        init_var = invgamma.rvs(1.0,scale=1.0,size=(1,1)).astype(precision)
        self.sB = gpuarray.to_gpu(init_var)
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_prior_w_kernel = kernels.get_function("add_gaussian_layer_grad")
        self.add_prior_b_kernel = kernels.get_function("add_bias_grad")
        self.scale_momentum_kernel = kernels.get_function("scale_momentum_Gaussian_Layer")
        self.scale_stepsize_kernel = kernels.get_function("scale_stepsize_Gaussian_Layer")
                
        ##initialize with random draw
        #self.updatePriorVals(layer.weights,layer.biases)
        
    def updateWeightGradient(self,weights,gW):
        grid1 = (gW.shape[1]+32-1)/32
        grid2 = (gW.shape[0]+32-1)/32
        M = np.int32(gW.shape[0])       
        N = np.int32(gW.shape[1])
        
        #Adds prior contribution to gradient on gpuarray object     
        self.add_prior_w_kernel(gW,weights, self.sW, M, N, block=(32,32,1),grid=( grid1,grid2) )
    
    def updateBiasGradient(self,biases,gB):
        grid1 = (gB.shape[1]+32-1)/32
        grid2 = (gB.shape[0]+32-1)/32
        M = np.int32(gB.shape[0])       
        N = np.int32(gB.shape[1])
        #Adds prior contribution to gradient on gpuarray object        
        self.add_prior_b_kernel(gB,biases, self.sB, M, N, block=(32,32,1),grid=( grid1,grid2) )
    
    ##Perform a Gibbs update of prior vals
    def updatePriorVals(self,weights,biases): 
        new_sW = np.zeros(self.sW.shape)
        weights_cpu = weights.get()
        n_w = np.float32(weights_cpu.shape[0]*weights_cpu.shape[1])
        shape_new = (self.shape + n_w)/2
        scale_new =  self.scale + (weights_cpu**2).sum()/2.0
        new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
        #print 'New standard deviation for feature ' + ': ' + str(new_val)
        new_sW = np.float32(new_val)
        self.sW = gpuarray.to_gpu(new_sW.astype(self.precision))
        
         ## Biases have common variance
        biases_cpu = biases.get()
        n_b = np.float32(biases.shape[1])
        shape_new = (self.shape + n_b)/2
        scale_new =  self.scale + ((biases_cpu)**2).sum()/2.0
        new_val = invgamma.rvs(shape_new,scale=scale_new,size=1)
        new_sB = np.float32(new_val)
        self.sB = gpuarray.to_gpu(new_sB)
        
    def getPriorDensityValue(self,weights,biases):
        w = weights.get()
        b = biases.get()
        sW = np.tile(self.sW.get(),w.shape)
        #print 'sW: ' + str(sW)
        sB = self.sB.get()
        val = -1*(w**2.0/(2.0*sW)).sum()
        val += -1*(b**2.0/(2.0*sB)).sum()
        return np.min((val,10^20))
    
    def scaleMomentum(self,pW,pB):
        grid1 = (pW.shape[1]+32-1)/32
        grid2 = (pW.shape[0]+32-1)/32
        M = np.int32(pW.shape[0])       
        N = np.int32(pW.shape[1])
        self.scale_momentum_kernel(pW,self.sW,M,N,block=(32,32,1),grid=(grid1,grid2)) 
        
        grid1 = (pB.shape[1]+32-1)/32
        grid2 = (pB.shape[0]+32-1)/32
        M = np.int32(pB.shape[0])       
        N = np.int32(pB.shape[1])
        self.scale_momentum_kernel(pB,self.sB,M,N,block=(32,32,1),grid=(grid1,grid2)) 
    
    def scaleStepSize(self,epsW,epsB):
        grid1 = (epsW.shape[1]+32-1)/32
        grid2 = (epsW.shape[0]+32-1)/32
        M = np.int32(epsW.shape[0])       
        N = np.int32(epsW.shape[1])
        self.scale_stepsize_kernel(epsW,self.sW,M,N,block=(32,32,1),grid=(grid1,grid2)) 
        
        grid1 = (epsB.shape[1]+32-1)/32
        grid2 = (epsB.shape[0]+32-1)/32
        M = np.int32(epsB.shape[0])       
        N = np.int32(epsB.shape[1])
        self.scale_momentum_kernel(epsB,self.sB,M,N,block=(32,32,1),grid=(grid1,grid2)) 
        
    

