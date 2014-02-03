# -*- coding: utf-8 -*-
"""
Created on Tue Jul  2 14:52:48 2013

@author: andy
"""
import numpy as np
import matplotlib.pyplot as plt

class HMC_sampler:
    def __init__(self,net,L,eps,scale=False,debug=False,track_vars=list()):
        self.L = L
        self.eps = eps
        self.net = net
        self.current_weights = list()
        self.current_biases = list()
        self.posterior_weights = list()
        self.posterior_sd = list()
        self.posterior_ARDMean = list()
        self.accept = 0.0
        self.sim = 0.0
        self.log_alpha = 0.0
        self.scale = scale   
        self.track_vars = track_vars
        self.track_ranks = list()
        self.track_pvals = list()
        self.debug = debug
        
    def track(self):
        ranks = np.zeros(shape=(len(self.track_vars)))
        pvals = np.zeros(shape=(len(self.track_vars)))
        for i in range(len(self.track_vars)):
            pvals[i] = 1.0-self.testMeanAgainstNull(self.track_vars[i],verbose=False)
            ranks[i] = self.getFeatureRankByARDMean(self.track_vars[i])
        
        self.track_ranks.append(ranks)
        self.track_pvals.append(pvals)
    
    def plot_debug(self):
        
        for i in range(len(self.track_vars)):
            rank = np.zeros(len(self.track_ranks))
            
            for j in range(len(self.track_ranks)):
                rank[j] = self.track_ranks[j][i]
            
            pval = np.zeros(len(self.track_pvals))
            for j in range(len(self.track_pvals)):
                pval[j] = self.track_pvals[j][i]
                
            x = np.linspace(1,len(self.track_ranks),len(self.track_ranks))
            plt.subplot(211)
            plt.plot(x,rank)
            plt.title('Rank for variable ' + str(self.track_vars[i]))
            
            x = np.linspace(1,len(pval),len(pval))
            plt.subplot(212)
            plt.plot(x,pval)
            plt.title('P-value ' + str(self.track_vars[i]))
            plt.show()  
    
    def simple_annealing_sim(self,n_keep,n_burnin,eta=0.95,T0=100,persist=0.0,var_refresh=1,verbose=False):
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
            if np.mod(self.sim,var_refresh) == 0:
                self.net.updateAllHyperParams()
                if verbose:
                    print 'Updating Hyper Parameters'
            self.HMC_sample(self.L,eps,T=T,verbose=verbose,persist=persist)
            T = np.max([eta*T,1.0])
            if self.sim > n_burnin:
                if self.debug and len(self.posterior_ARDMean) > 1:
                    self.track()
                ## Get the ARD variance params
                self.posterior_sd.append(self.net.layers[0].prior.sW.get())
                self.posterior_ARDMean.append(self.net.layers[0].prior.mean.get())
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
        self.current_pW = list()
        self.current_pB = list()
        for i in range(0,len(self.net.layers)):
            self.current_weights.append(self.net.layers[i].weights.copy())
            self.current_biases.append(self.net.layers[i].biases.copy())
            self.current_pW.append(self.net.layers[i].pW.copy())
            self.current_pB.append(self.net.layers[i].pB.copy())
    
    def restore_params(self):
        for i in range(0,len(self.net.layers)):            
            self.net.layers[i].setWeights(self.current_weights[i])
            self.net.layers[i].setBiases(self.current_biases[i])
            
            self.net.layers[i].pW = self.current_pW[i]
            self.net.layers[i].pB = self.current_pB[i]

    def negateMomenta(self):
        for i in range(0,len(self.net.layers)):            
            self.net.layers[i].pW = -1*self.current_pW[i]
            self.net.layers[i].pB = -1*self.current_pB[i]

    def plotARD(self,featureID,useMedian=False):
        P = self.net.layers[0].prior.sW.shape[1]
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,featureID-1]
                current_var[j] = sample
        
        plt.subplot(211)
        plt.hist(current_var,bins=25,normed=True)
        plt.title('Histogram of posterior samples for feature ' + str(featureID+1))
        
        x = np.linspace(1,len(current_var),len(current_var))
        plt.subplot(212)
        plt.plot(x,current_var)
        plt.title('Trace of posterior samples for feature ' + str(featureID+1))
        plt.ion()
        plt.show()  
    
    def getARDSummary(self,plot=0,useMedian=False):
        P = self.net.layers[0].prior.sW.shape[1]
        summary = np.zeros(P)
        stds = np.zeros(P)
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,i]
                current_var[j] = sample
            if useMedian:
                summary[i] = np.median(current_var)
            else:
                summary[i] = current_var.mean()
            stds[i] = current_var.std()
        args = summary.argsort()
        for i in range(1,len(args)+1):
            index = args[-i]
            if useMedian:
                print 'Median ARD for feature ' + str(index+1) + ' is ' +str(summary[index])
            else:
                print 'Mean ARD for feature ' + str(index+1) + ' is ' +str(summary[index])
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
                plt.ioff()
                plt.show()
    
    def getARDPosteriorMeanSummary(self,plot=0,useMedian=False):
        P = self.net.layers[0].prior.sW.shape[1]
        summary = np.zeros(P)
        stds = np.zeros(P)
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_ARDMean))
            for j in range(0,len(current_var)):
                sample = self.posterior_ARDMean[j][0,i]
                current_var[j] = sample
            if useMedian:
                summary[i] = np.median(current_var)
            else:
                summary[i] = current_var.mean()
            stds[i] = current_var.std()
        args = summary.argsort()
        for i in range(1,len(args)+1):
            index = args[-i]
            if useMedian:
                print 'Median of Posterior ARD Mean samples for feature ' + str(index+1) + ' is ' +str(summary[index])
            else:
                print 'Average of Posterior ARD Mean samples for feature ' + str(index+1) + ' is ' +str(summary[index])
            if(plot ==1):
                current_var = np.zeros(len(self.posterior_ARDMean))
                for j in range(0,len(current_var)):
                    sample = self.posterior_ARDMean[j][0,index]
                    current_var[j] = sample  
                plt.subplot(211)
                plt.hist(current_var,bins=20,normed=True)
                plt.title('Histogram of posterior samples for feature ' + str(index+1))
                
                x = np.linspace(1,len(current_var),len(current_var))
                plt.subplot(212)
                plt.plot(x,current_var)
                plt.title('Trace of posterior samples for feature ' + str(index+1))
                plt.ioff()
                plt.show()
    
    def getFeatureRankByARDMean(self,feature_ID,useMedian=True):
        P = self.net.layers[0].prior.sW.shape[1]
        summary = np.zeros(P)
        stds = np.zeros(P)
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_ARDMean))
            for j in range(0,len(current_var)):
                sample = self.posterior_ARDMean[j][0,i]
                current_var[j] = sample
            if useMedian:
                summary[i] = np.median(current_var)
            else:
                summary[i] = current_var.mean()
            stds[i] = current_var.std()
        args = summary.argsort()
        rank = 1
        for i in range(1,len(args)+1):
            index = args[-i]
            if index == (feature_ID-1):
                return rank
            rank = rank + 1
    
    def getFeatureRank(self,feature_ID,useMedian=True):
        P = self.net.layers[0].prior.sW.shape[1]
        summary = np.zeros(P)
        stds = np.zeros(P)
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,i]
                current_var[j] = sample
            if useMedian:
                summary[i] = np.median(current_var)
            else:
                summary[i] = current_var.mean()
            stds[i] = current_var.std()
        args = summary.argsort()
        rank = 1
        for i in range(1,len(args)+1):
            index = args[-i]
            if index == (feature_ID-1):
                return rank
            rank = rank + 1
    
    def getCredibleInterval(self,featureID,level=95):
        P = self.net.layers[0].prior.sW.shape[1]
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,featureID-1]
                current_var[j] = sample
        
        interval = np.array([np.percentile(current_var,q=(100.0-level)/2.0), 
                             np.percentile(current_var,q=100.0-(level/2.0))])   
        return interval   
    
    def testMeanAgainstNull(self,featureID,verbose=True):
        num_units =  self.net.layers[0].n_units        
        
        mean_null = self.net.layers[0].prior.scale/(self.net.layers[0].prior.shape-1)
        
        n_samples = len(self.posterior_ARDMean)
        num_hit = 0.0
        for i in range(0,n_samples):           
            sample = self.posterior_ARDMean[i][0,featureID-1]
            if( sample > mean_null ):
                num_hit += 1.0
        
        p = np.float(num_hit)/np.float(n_samples)
        if verbose:
            print 'Probability the ARD mean for feature ' + str(featureID) + ' is greater than mean null of '+str(mean_null)+':' + str(p)
        return p
        
    
    def getHit(self,featureID,level=95,verbose=False):
        prior = self.net.layers[0].prior
        P = prior.sW.shape[1]
        for i in range(0,P):
            current_var = np.zeros(len(self.posterior_sd))
            for j in range(0,len(current_var)):
                sample = self.posterior_sd[j][0,featureID-1]
                current_var[j] = sample
        quantile = (100.0-level)/2.0
        interval = np.array([np.percentile(current_var,q=quantile), 
                             np.percentile(current_var,q=(level+quantile))])   

        num_units =  self.net.layers[0].n_units
        
        adjustment = num_units*2.0
        shape_null = prior.shape + num_units/2.0
        scale_null = prior.scale + adjustment
        mean_null = scale_null/(shape_null-1)
        hit = False
        if interval[0] > mean_null:
            hit = True
        
        if verbose:
            print 'Feature ' + str(featureID) + ' is a hit? ' + str(hit)
            print 'Mean under null: ' + str(mean_null)
            print str(level) + '% credible interval: [' + str(interval[0]) + ',' + str(interval[1]) + ']'             
        return hit   
    
            
    def HMC_sample(self,L,eps,persist=0.0,T=1.0,verbose=False):        
        
        self.net.feed_forward()        
        self.net.update_all_momenta(persist)
        if self.scale:
            for i in range(0,len(self.net.layers)):
                layer = self.net.layers[i]                
                layer.scaleMomentum()
                layer.scaleStepSize()
        init_ll = self.net.log_like_val()
        current_k = self.net.get_total_k()/2.0
        current_u = self.net.posterior_kernel_val()
        self.copy_params()
        
        for step in range(0,L):
            self.net.updateAllHyperParams()
            for i in range(0,len(self.net.layers)):                
                #Perform 1 leap-frog update
                #Update outputs for each layer
                self.net.feed_forward()
                self.net.updateAllGradients()
                #take a half step for momentum
                layer = self.net.layers[i]      
                
                epsW_component = eps*layer.epsW
                epsB_component = eps*layer.epsB
                
                layer.pW = layer.pW + (epsW_component/2.0)*layer.gW
                layer.pB = layer.pB + (epsB_component/2.0)*layer.gB                
                
                #take a full step for parameters
                layer.weights += epsW_component*layer.pW
                layer.biases += epsB_component*layer.pB
                
                #Update outputs for each layer
                self.net.feed_forward()
                self.net.updateAllGradients()
                #take a final half step for momentum
                layer.pW = layer.pW + (epsW_component/2.0)*layer.gW
                layer.pB = layer.pB + (epsB_component/2.0)*layer.gB
                
        
        self.net.feed_forward()
        self.net.updateAllGradients()        
        ##Calculate log_posterior value at current parameter estimates
        proposed_u = self.net.posterior_kernel_val()
        #Divide by T in the case of SA
        proposed_k = self.net.get_total_k()/2.0
        
        diff = (proposed_u - proposed_k - current_u + current_k) /T
        alpha = np.min([0,diff])
        u = np.log(np.random.random(1)[0])
        self.log_alpha = alpha
        
        if u < diff:
            msg = 'Accept!'
            self.accept += 1.0
        else:
            msg = 'Reject!'
            self.restore_params()       
        #if persist > 0:
            #self.negateMomenta()
        if verbose:
                metric = 'accuracy'
                if self.net.layers[-1].type == 'Gaussian':
                    metric = 'RMSE'
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
                print 'Current '+metric+' on training set: ' + str(self.net.getTrainAccuracy())
                print 'Acceptance rate: ' + str(self.accept/np.float(self.sim))
                print '----------------------------'
            