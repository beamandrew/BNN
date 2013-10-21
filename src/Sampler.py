# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:52:48 2013

@author: andy
"""
import numpy as np
import matplotlib.pyplot as plt

class HMC_sampler:
    def __init__(self,net,L,eps,scale=False):
        self.L = L
        self.eps = eps
        self.net = net
        self.current_weights = list()
        self.current_biases = list()
        self.posterior_weights = list()
        self.posterior_sd = list()
        self.accept = 0.0
        self.sim = 0.0
        self.log_alpha = 0.0
        self.scale = scale
    
    def fixed_step_size_sim(self,n_keep,n_burnin,verbose=False):
        n_sim = n_keep + n_burnin
        self.sim = 0.0
        while self.sim < n_sim:
            self.sim += 1.0
            if verbose:
                print 'Iteration ' + str(self.sim)    
            self.net.updateAllHyperParams()
            self.HMC_sample(self.L,self.eps,verbose=verbose)
            if self.sim > n_burnin:
                ## Get the ARD variance params
                self.posterior_sd.append(self.net.layers[0].prior.sW.get())
                for i in range(0,self.net.num_layers):
                    self.net.layers[i].addPosteriorWeightSample(self.net.layers[i].weights.get())
                    self.net.layers[i].addPosteriorBiasSample(self.net.layers[i].biases.get())
                
    def initialize(self):
        original_eps = self.eps
        original_L = self.L
        eps_vals = [0.1,0.01,0.001,0.0001]
        for eps in eps_vals:
            self.eps = eps
            self.L = 5
            print 'Eps value ' + str(eps)
            self.net.feed_forward()
            current_u = self.net.posterior_kernel_val()
            self.simple_annealing_sim(0,1,eta=1.0,T0=10,verbose=False)
            next_u = self.net.posterior_kernel_val()
            while next_u > current_u:
                current_u = next_u
                self.simple_annealing_sim(0,1,eta=1.0,T0=10,verbose=False)
                next_u = self.net.posterior_kernel_val()
                print 'Current training accuracy ' + str(self.net.getTrainClassificationAccuracy())
        self.eps = original_eps
        self.L = original_L
        self.sim = 0.0
            
    def fixed_L_random_eps_sim(self,n_keep,n_burnin,eps0=0.01,eta=1.0,verbose=False):
        n_sim = n_keep + n_burnin
        self.sim = 0.0
        while self.sim < n_sim:
            self.sim += 1.0
            self.net.updateAllHyperParams()
            eps_rand = np.exp(eta*np.random.standard_cauchy(1))*eps0
            if verbose:
                print 'Iteration ' + str(self.sim)
                print 'eps: ' + str(np.float(eps_rand))
            self.HMC_sample(self.L,np.float(eps_rand),verbose=verbose)
            if self.sim > n_burnin:
                ## Get the ARD variance params
                self.posterior_sd.append(self.net.layers[0].prior.sW.get())
                for i in range(0,self.net.num_layers):
                    self.net.layers[i].addPosteriorWeightSample(self.net.layers[i].weights.get())
                    self.net.layers[i].addPosteriorBiasSample(self.net.layers[i].biases.get())
    
    def simple_annealing_sim(self,n_keep,n_burnin,eta=0.95,eps_eta=0.99,T0=100,epsFinal=0.001,nu=1.0,epsDecay=False,verbose=False):
        n_sim = n_keep + n_burnin
        self.sim = 0.0
        T = T0
        eps = self.eps
        while self.sim < n_sim:
            self.sim += 1.0
            if verbose:
                print 'Iteration ' + str(self.sim)
                print 'T: ' + str(np.float(T))
                print 'eps: ' + str(np.float(eps))                
            self.HMC_sample(self.L,eps,T=T,verbose=verbose)
            self.net.updateAllHyperParams()
            T = np.max([eta*T,1.0])
            if epsDecay:
                eps = float(np.max([eps*eps_eta,epsFinal]))
            if self.sim > n_burnin:
                ## Get the ARD variance params
                self.posterior_sd.append(self.net.layers[0].prior.sW.get())
                for i in range(0,self.net.num_layers):
                    self.net.layers[i].addPosteriorWeightSample(self.net.layers[i].weights.get())
                    self.net.layers[i].addPosteriorBiasSample(self.net.layers[i].biases.get())
    
    def dual_average_sim(self,n_keep,n_burnin,eps0=0.01,delta=0.65,lam=1.0,eps_bar0=1.0,H_bar0=0.0,gamma=0.05,t0=10.0,k=10.0,max_eps=1.0,verbose=False):
        #Set up initial values for params        
        mu = np.log10(10.0*eps0)        
        H_bar = H_bar0
        log_eps_bar_m = np.log(eps_bar0)
        n_sim = n_keep + n_burnin
        
        eps = eps0
        self.sim = 0.0
        while self.sim < n_sim:
            self.sim += 1.0
            self.net.updateAllHyperParams()
            L_m = np.max([1.0,np.round(lam/eps)])
            log_eps_m = np.log(eps)
            if(verbose):
                print 'Iteration ' + str(self.sim)
                print 'L_m: ' + str(L_m) + ' eps: ' + str(eps)
            self.HMC_sample(np.int(L_m),np.float(eps),verbose=verbose)
            if self.sim > n_burnin:
                eps = np.exp(log_eps_bar_m)
                ## Get the ARD variance params
                self.posterior_sd.append(self.net.layers[0].prior.sW.get())
                for i in range(0,self.net.num_layers):
                    self.net.layers[i].addPosteriorWeightSample(self.net.layers[i].weights.get())
                    self.net.layers[i].addPosteriorBiasSample(self.net.layers[i].biases.get())
            else:
                H_bar = (1 - 1/(self.sim+t0))*H_bar + (1/(self.sim+t0)*(delta-np.exp(self.log_alpha)))
                log_eps_m_prev = log_eps_m
                log_eps_m = mu - np.sqrt(self.sim)/gamma*H_bar
                log_eps_bar_m = self.sim**(-k)*log_eps_m + (1-self.sim**(-k))*log_eps_m_prev
                eps = np.min([max_eps,np.exp(log_eps_m)])
    
    def find_starting_eps(self,eps=0.1,verbose=False):
        self.net.feed_forward()
        prev_kernel_val = self.net.posterior_kernel_val()
        print 'Initial kernel val ' + str(prev_kernel_val)
        self.HMC_sample(1,eps,always_accept=True)
        self.net.feed_forward()
        new_kernel_val = self.net.posterior_kernel_val()
        ratio = (new_kernel_val/prev_kernel_val)        
        ratio_ind = 0.0
        if ratio > 0.5:
            ratio_ind = 1.0
        a = 2*ratio_ind - 1
        
        prev_kernel_val = new_kernel_val
        if verbose:        
            print 'Initial Ratio: ' + str(ratio) 
            print 'New kernel val: ' + str(new_kernel_val)
            print 'Initial a: ' + str(a)
        i = 0
        ratio_test = ratio**(a)
        while ratio_test > 2**(-a):
            i += 1
            eps = 2**(a)*eps
            self.HMC_sample(1,eps,always_accept=True)
            self.net.feed_forward()
            new_kernel_val = self.net.posterior_kernel_val()
            ratio = (new_kernel_val/prev_kernel_val)
            prev_kernel_val = new_kernel_val
            ratio_test = ratio**(a)
            if verbose:
                print 'Iteration: ' + str(i)
                print 'Ratio: ' + str(ratio) 
                print 'eps: ' + str(eps)
        return eps
    
    def copy_params(self):
        self.current_weights = list()
        self.current_biases = list()
        for i in range(0,len(self.net.layers)):
            self.current_weights.append(self.net.layers[i].weights.copy())
            self.current_biases.append(self.net.layers[i].biases.copy())
    
    def restore_params(self):
        for i in range(0,len(self.net.layers)):            
            self.net.layers[i].setWeights(self.current_weights[i])
            self.net.layers[i].setBiases(self.current_biases[i])
    
    def getARDSummary(self,plot=0):
        P = self.net.layers[0].prior.sW.shape[1]
        means = np.zeros(P)
        stds = np.zeros(P)
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,i]
                current_var[j] = sample
            means[i] = current_var.mean()
            stds[i] = current_var.std()
        args = means.argsort()
        for i in range(1,len(args)+1):
            index = args[-i]
            print 'Mean ARD for feature ' + str(index+1) + ' is ' +str(means[index])
            if(plot ==1):
                current_var = np.zeros(len(self.posterior_sd))
                for j in range(0,len(current_var)):
                    sample = self.posterior_sd[j][0,index]
                    current_var[j] = sample  
                plt.subplot(211)
                plt.hist(current_var,bins=20,normed=True)
                plt.title('Histogram of posterior samples for feature ' + str(index+1))
                
                x = np.linspace(1,len(current_var),len(current_var))
                plt.subplot(212)
                plt.plot(x,current_var)
                plt.title('Trace of posterior samples for feature ' + str(index+1))
                plt.show()
    
    def getFeatureRank(self,feature_ID):
        P = self.net.layers[0].prior.sW.shape[1]
        means = np.zeros(P)
        stds = np.zeros(P)
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,i]
                current_var[j] = sample
            means[i] = current_var.mean()
            stds[i] = current_var.std()
        args = means.argsort()
        rank = 1
        for i in range(1,len(args)+1):
            index = args[-i]
            if index == (feature_ID-1):
                return rank
            rank = rank + 1
    
    def HMC_sample(self,L,eps,always_accept=False,T=1.0,verbose=False):
        self.net.feed_forward()        
        self.net.init_all_momentum()
        init_ll = self.net.log_like_val()
        current_k = self.net.get_total_k()
        current_u = self.net.posterior_kernel_val()
        self.copy_params()
        
        self.net.updateAllGradients()
        self.net.updateAllHyperParams()
        ##take an initial half-step
        for i in range(0,len(self.net.layers)):
            layer = self.net.layers[i]
            if self.scale:
                layer.scaleMomentum()
            layer.pW += eps*(layer.gW/2.0)
            layer.pB += eps*(layer.gB/2.0)
        
        for step in range(0,L):
            #Update the parameters
            self.net.updateAllHyperParams()
            for i in range(0,len(self.net.layers)):
                layer = self.net.layers[i]
                layer.weights += eps*layer.pW
                layer.biases += eps*layer.pB
           
            if step != L:
                self.net.feed_forward()
                self.net.updateAllGradients()
                for i in range(0,len(self.net.layers)):
                    layer = self.net.layers[i]
                    layer.pW += eps*layer.gW
                    layer.pB += eps*layer.gB
        
        self.net.feed_forward()
        self.net.updateAllGradients()
        self.net.updateAllHyperParams()
        ##take a final half-step
        for i in range(0,len(self.net.layers)):
            layer = self.net.layers[i]
            layer.pW += eps*(layer.gW/2.0)
            layer.pB += eps*(layer.gB/2.0)
        
        
        ##Calculate log_posterior value at current parameter estimates
        self.net.feed_forward()
        proposed_u = self.net.posterior_kernel_val()
        #Divide by T in the case of SA
        proposed_k = self.net.get_total_k()/T
        
        diff = (proposed_u-proposed_k) - (current_u-current_k)
        alpha = np.min([0,diff])
        u = np.log(np.random.random(1)[0])
        self.log_alpha = alpha
        
        if u < diff:
            msg = 'Accept!'
            self.accept += 1.0
        else:
            msg = 'Reject!'
            if not always_accept:
                self.restore_params()        
        if verbose:
                print '----------------------------'
                print 'Current U: ' + str(current_u)
                print 'Proposed U: ' + str(proposed_u)      
                print 'Current K: ' + str(current_k)
                print 'Proposed K: ' + str(proposed_k)      
                print 'Total diff: ' + str(diff)
                print 'Current log-like: ' + str(init_ll)
                print 'Proposed log-like: ' + str(self.net.log_like_val())
                print 'Comparing alpha of: ' + str(alpha) + ' to uniform of: ' + str(u)
                print msg
                print 'Current accuracy on training set: ' + str(self.net.getTrainAccuracy())
                print 'Acceptance rate: ' + str(self.accept/np.float(self.sim))
                print '----------------------------'
            