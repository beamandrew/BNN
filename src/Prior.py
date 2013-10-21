# -*- coding: utf-8 -*-
"""
Created on Sun Jun 30 16:22:18 2013

@author: andy
"""
from abc import ABCMeta, abstractmethod

import numpy as np

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
    

class ARD_Prior(Prior):
    def __init__(self,shape,scale,weights,biases,precision=np.float32):
        self.precision = precision
        self.shape = shape
        self.scale = scale
        self.sW = gpuarray.zeros((1,weights.shape[0]),precision)
        self.sB = gpuarray.zeros((1,1),precision)
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_prior_kernel = kernels.get_function("add_ARD_grad")
        self.add_prior_b_kernel = kernels.get_function("add_bias_grad")
        self.scale_momentum_kernel = kernels.get_function("scale_momentum_ARD")
        
        ##initialize with random draw
        self.updatePriorVals(weights,biases)
    
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
        n_w = weights_cpu.shape[1]
        shape_new = (self.shape + n_w)/2
        for i in range(0,len(weights_cpu)):
            scale_new =  self.scale + 0.5*((weights_cpu[i]**2).sum())
            new_val = np.random.gamma(shape_new,1.0/scale_new)
            new_sW[0,i] = 1.0/np.sqrt(new_val)
            #print 'New scale for feature ' + str(i+1) + ': ' + str(scale_new)
            #print 'New standard deviation for feature ' + str(i+1) + ': ' + str(new_sW[0,i])
        
        self.sW = gpuarray.to_gpu(new_sW.astype(self.precision))
        
        ## Biases have common variance
        biases_cpu = biases.get()
        n_b = biases.shape[1]
        shape_new = (self.shape + n_b)/2
        scale_new =  self.scale + 0.5*((biases_cpu**2.0).sum())
        new_val = np.random.gamma(shape_new,1.0/scale_new)
        new_sB = 1.0/np.sqrt(new_val)
        self.sB = gpuarray.to_gpu(new_sB.astype(self.precision))
    
    def getPriorDensityValue(self,weights,biases):
        w = weights.get()
        b = biases.get()
        sW = np.tile(self.sW.get(),(weights.shape[1],1)).T
        sB = self.sB.get()
        val = (np.log(1/(np.sqrt(2*np.pi)*sW)) - (w**2.0)/(2.0*sW**2.0)).sum()
        val += (np.log(1/(np.sqrt(2*np.pi)*sB)) - b**2/(2*sB**2)).sum()
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
        
class Normal_Unit_Prior(Prior):
    def __init__(self,shape,scale,weights,biases,precision=np.float32):
        self.precision = precision
        self.shape = shape
        self.scale = scale
        self.sW = gpuarray.zeros((1,weights.shape[1]),precision)
        self.sB = gpuarray.zeros((1,1),precision)
        kernels = SourceModule(open(path+'/kernels.cu', "r").read())        
        self.add_prior_w_kernel = kernels.get_function("add_normal_unit_grad")
        self.add_prior_b_kernel = kernels.get_function("add_bias_grad")
        self.scale_momentum_kernel = kernels.get_function("scale_momentum_normal_unit")
                
        ##initialize with random draw
        self.updatePriorVals(weights,biases)
        
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
        n_w = weights_cpu.shape[0]
        shape_new = (self.shape + n_w)/2
        for i in range(0,weights_cpu.shape[1]):
            scale_new =  self.scale + 0.5*((weights_cpu[:,i]**2).sum())
            new_val = np.random.gamma(shape_new,1.0/scale_new)
            new_sW[0,i] = 1.0/np.sqrt(new_val)
        self.sW = gpuarray.to_gpu(new_sW.astype(self.precision))
        
        ## Biases have common variance
        new_sB = np.zeros(self.sB.shape)
        biases_cpu = biases.get()
        n_b = biases.shape[1]
        shape_new = (self.shape + n_b)/2
        scale_new =  self.scale + 0.5*((biases_cpu**2).sum())
        new_val = np.random.gamma(shape_new,1.0/scale_new)
        new_sB = 1.0/np.sqrt(new_val)
        self.sB = gpuarray.to_gpu(new_sB.astype(self.precision))
        
    def getPriorDensityValue(self,weights,biases):
        w = weights.get()
        b = biases.get()
        sW = np.tile(self.sW.get(),(w.shape[0],1))
        sB = self.sB.get()
        val = (np.log(1/(np.sqrt(2*np.pi)*sW)) - w**2/(2*sW**2)).sum()
        val += (np.log(1/(np.sqrt(2*np.pi)*sB)) - b**2/(2*sB**2)).sum()
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

