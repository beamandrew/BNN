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
import matplotlib
import matplotlib.pyplot as plt

import scikits.cuda.linalg as linalg
import scikits.cuda.misc as cumisc
linalg.init()

class BNN:
    def __init__(self,X,Y,layers,init_sd=1.0,precision=np.float32):
        self.X = gpuarray.to_gpu(X.astype(precision).copy())
        self.Y = gpuarray.to_gpu(Y.astype(precision).copy())
        self.precision = precision
        self.num_preds = X.shape[1]
                
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
    
    def updateAllGradients(self,print_timing=False,include_prior=True):
        ##Compute gradient and do back-prop
        self.feed_forward()
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
    
    def update_all_momenta(self,persist=0.0):       
        for i in range(0,(len(self.layers))):
            self.layers[i].updateMomenta(persist)
    
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
        return np.min((self.layers[-1].get_log_like_val(self.Y),10^20))
    
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
        return np.min( (val,10^20) )
    
    #Initialize network with maximum-likelihood estimates
    def initialize(self,iters=100,verbose=True,step_size=1e-3,target_acc=1.0,include_prior=False):
        self.feed_forward()
        done = False
        i = 1
        while not done:
            if include_prior:
                self.updateAllHyperParams()
            
            self.feed_forward()
            self.updateAllGradients(include_prior=include_prior)
            for j in range(0,self.num_layers):
                layer = self.layers[j]
                layer.weights += step_size*layer.gW
                layer.biases += step_size*layer.gB
            
            if np.mod(i,100) == 0:
                if verbose:
                    print 'Iteration: ' + str(i)
                    print 'Log-liklihood value: ' + str(self.log_like_val())
                    print 'Posterior value: ' + str(self.posterior_kernel_val())
                    print 'Current accuracy: ' + str(self.getTrainAccuracy())
            
            if (i > iters) or (self.getTrainAccuracy() >= target_acc):
                done = True
            i += 1
        
        print 'Final Log-liklihood value: ' + str(self.log_like_val())
        print 'Final Posterior value: ' + str(self.posterior_kernel_val())
        print 'Final accuracy: ' + str(self.getTrainAccuracy())
    
    def getMemoryStatus(self):
        (free,total)=cuda.mem_get_info()
        print("Global memory occupancy:%f%% free"%(free*100/total))    
    
    def plotCurrentWeights(self,layerID,absval=False,scale=False):
        l = self.layers[layerID]
        data = l.weights.get().T
        if absval:
            data = np.abs(data)
        if scale:
            prior = l.prior
            data = data/prior.getWeightSigmaMatrix(data)
        
        
        x_labels = np.linspace(1,data.shape[1],data.shape[1]).astype(np.int32)
        y_labels = np.linspace(1,data.shape[0],data.shape[0]).astype(np.int32)
        
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
        
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
        
        # want a more natural, table-like display
        ax.invert_yaxis()
        #ax.xaxis.tick_top()
        
        ax.set_xticklabels(x_labels, minor=False)
        ax.set_yticklabels(y_labels, minor=False)
        
        plt.xticks(rotation=90)
        plt.ion()
        plt.show()
        
    def plotWeightTraceForVariable(self,varID):
        layer = self.layers[0]
        data = np.zeros(shape=(layer.weights.shape[1],len(layer.posterior_weights)))
        plot_rows = np.ceil(data.shape[0]/2).astype(int)
        for unit in range(0,data.shape[0]):
            for j in range(0,len(layer.posterior_weights)):
                data[unit,j] = layer.posterior_weights[j][varID-1,unit]
        
        fig, axarr = plt.subplots(nrows=plot_rows,ncols=2)
        index = 0
        for i in range(0,plot_rows):
            for j in range(0,2):
                axarr[i,j].plot(data[index])
                index += 1
                axarr[i,j].set_title('Unit ' + str(index))
        plt.ion()
        plt.show()
    
    def getSamplesForVariable(self,featureID,layerID=0):
        layer = self.layers[layerID]
        data = np.zeros(shape=(layer.weights.shape[1],len(layer.posterior_weights)))
        for unit in range(0,data.shape[0]):
            for j in range(0,len(layer.posterior_weights)):
                data[unit,j] = layer.posterior_weights[j][featureID-1,unit]
        
        return data
    
    def plotPosteriorWeights(self,layerID,absval=False,scale=False):
        l = self.layers[layerID]
        data = np.zeros(l.weights.get().T.shape)
        for i in range(0,len(l.posterior_weights)):
            data += l.posterior_weights[i].T
        data = data/len(l.posterior_weights)
        if absval:
            data = np.abs(data)
        if scale:
            prior = l.prior
            data = data/prior.getWeightSigmaMatrix(data)
        
        
        x_labels = np.linspace(1,data.shape[1],data.shape[1]).astype(np.int32)
        y_labels = np.linspace(1,data.shape[0],data.shape[0]).astype(np.int32)
        
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
        
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
        
        # want a more natural, table-like display
        ax.invert_yaxis()
        #ax.xaxis.tick_top()
        
        ax.set_xticklabels(x_labels, minor=False)
        ax.set_yticklabels(y_labels, minor=False)
        
        plt.xticks(rotation=90)
        plt.ion()
        plt.show()
    
    
    def plotCurrentGradients(self,layerID,absval=False,scale=False):
        l = self.layers[layerID]
        data = l.gW.get().T
        if absval:
            data = np.abs(data)
        if scale:
            prior = l.prior
            data = data/prior.getWeightSigmaMatrix(data)
        
        
        x_labels = np.linspace(1,data.shape[1],data.shape[1]).astype(np.int32)
        y_labels = np.linspace(1,data.shape[0],data.shape[0]).astype(np.int32)
        
        fig, ax = plt.subplots()
        heatmap = ax.pcolor(data, cmap=plt.cm.Blues)
        
        # put the major ticks at the middle of each cell
        ax.set_xticks(np.arange(data.shape[1])+0.5, minor=False)
        ax.set_yticks(np.arange(data.shape[0])+0.5, minor=False)
        
        # want a more natural, table-like display
        ax.invert_yaxis()
        #ax.xaxis.tick_top()
        
        ax.set_xticklabels(x_labels, minor=False)
        ax.set_yticklabels(y_labels, minor=False)
        
        plt.xticks(rotation=90)
        plt.ion()
        plt.show()
