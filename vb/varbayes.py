'''
This code aims to perform stochastic variational inference 
for the modified HDP model we use to model RNA-seq data from
 a sample of individuals sharing a set of isoforms.
'''

import argparse
from collections import OrderedDict
import copy
import gc
import math
import pickle, gzip
from random import gauss
import random
import shutil
import sys, os
import tempfile
from time import time

from joblib import Parallel, delayed
import numpy as np
from scipy.special import gammaln

from compute_expectations import expectation_log_beta, expectation_log_dirichlet, expectation_log_sigma_beta, expectation_b
from compute_expectations import fast_expectation_deterministic_b, construct_b_from_bprime, EXISTANCE_THRESHOLD, LAMBDAP_UNCERTAINTY
from compute_expectations import jnxTupleMuInconsistent, expectation_sigma_beta, MU_UNCERTAINTY, BPRIME_UNCERTAINTY
from init_vb import initialize_iso_splicegraph, initialize_iso_simple, compatibility_v
import transformations as tf
from update_vb import f_update_zeta, f_update_phi

import pyximport; pyximport.install()
try:
    import fast_update as fu
except ImportError:
    print >> sys.stderr, 'could not find fast_update, cannot use fast cython implementation'

MIN_NUMBER_ITS=0
ROUNDS_IN_BETWEEN_PROPOSAL=0
MIN_NUMBER_PROPOSALS=0
MAX_NUMBER_PROPOSALS=0
NUM_NEW_ISO = 0
REDUNDANT_ISOS = 0
USE_CYTHON = 0
MERGE_BURN_IN = 100
INDIVIDUAL_PLATE_CONVERGE = 1e-3
INDIVIDUAL_PLATE_MAXIT = 100
np.set_printoptions(precision=6)        
np.set_printoptions(suppress=True)

# time_logging = np.zeros((10),dtype=np.float)
mus_added = []

class VarDist:      
    
    def __init__(self,V,alpha,eta,omega,r,s,X,N,M,K,term2exon,term2junction,junction_set,initialize_from,existence_threshold,threads,maxiso,splicegraph,exon_pos_and_lengths,term2dis,max_v,C):
        self.param,self.folder = self.get_initial_param(V,alpha,eta,omega,r,s,X,N,M,K,term2exon,term2junction,junction_set,
                                                        initialize_from,existence_threshold,threads,maxiso,splicegraph,exon_pos_and_lengths,term2dis,max_v,C)
    
    def  get_initial_param(self, V,alpha,eta,omega,r,s,X,N,M,K,term2exon,term2junction,junction_set,
                           initialize_from,existence_threshold,threads,maxiso,splicegraph,exon_pos_and_lengths,term2dis,max_v,C):
        """
        generates the initial values of the GLOBAL variational parameters: a1,
        a2, lambda, rho, mu
        """
        
        if not initialize_from==None:
            # first option, initialize from a npy file, with predefined values, the variables must have the same names as here
            param_init=np.load(initialize_from)
            a1=param_init['a1']
            a2=param_init['a2']
            phi=param_init['phi']
            zeta=param_init['zeta']
            rho1=param_init['rho1']
            rho2=param_init['rho2']
            mu=param_init['mu']
            lambdap=param_init['lambdap'] 
            gamma1=param_init['gamma1']
            gamma2=param_init['gamma2']
            
            param = {'mu': mu, 'lambdap':lambdap, 'a1':a1, 'a2':a2, 'rho1':rho1, 'rho2':rho2, 'zeta':zeta, 'phi':phi ,'gamma1':gamma1, 'gamma2':gamma2}

        else:
            '''initialize mu: var param for bprime'''
            # number of exons
#             E=max(max(term2exon),max([max(item) for sublist in term2junction.values() for item in sublist]))+1
            if len(term2junction)>0:
                E=max(max(term2exon),max([item for sublist in term2junction.values() for item in sublist]))+1
            else :
                E=max(term2exon)+1
                                
            if math.log(K+1) > E*math.log(2):
                # this is the maximum number of isoforms, given the number of exons
                K=int(math.pow(2,E))-1
            
            if(K>maxiso):
                K=maxiso
            # same max number of global isoforms as isoforms within a single individual
            # maybe this could be changed later
            
            # initialize K isoforms from the E exons
            if splicegraph:
                iso_init=initialize_iso_splicegraph(K,E,term2exon,term2junction,maxiso)
                if iso_init is None:
                    print "too many isoforms in the splice graph initialization, falling back on random"
#                     iso_init=initialize_iso(K,E,term2exon,term2junction,r,s)
                    iso_init=initialize_iso_simple(K,E,term2exon,term2junction,r,s)
            else:
#                 if K < maxiso:
#                     iso_init=initialize_iso(K,E,term2exon,term2junction,r,s)
#                 else: 
                iso_init=initialize_iso_simple(K,E,term2exon,term2junction,r,s)
            K=iso_init.shape[0]
            assert iso_init.shape[0]==K
            self.K = K
            # check that the set of skipped exons is coherent with the junction reads
            # ie that each junction of the junction set is present in at least one isoform
            #mu=check_iso(iso_init,junction_set)
            folder = tempfile.mkdtemp()
            mu=iso_init
            
            # leave some room for errors
            mu[mu==1]=0.95
            mu[mu==0]=0.05
            assert mu.shape[0]==K
            
            ''' construct term selector vector from average exon selector vector mu '''
            b=construct_b_from_bprime(V,mu,term2exon,term2junction,0)
#             b=construct_b_from_bprime_pairedend(V,mu,term2exon,term2junction,0,exon_pos_and_lengths)
            for bv in range(len(b)):
                mus_added.append(b[bv])
            if any(np.sum(b,0)==0):
                print 'every term should be in at least one isoform (topic)'
            
            '''initialize lambda: var param for beta'''
            (term2exonsCovered,term2exonsCoveredLengths) = tf.transform_term_2_exon_nonpaired(V,term2junction,term2exon,term2dis,max_v)
            self.term2exonsCovered = term2exonsCovered
            self.term2exonsCoveredLengths = term2exonsCoveredLengths
            jnxKeysArray=np.asarray(term2junction.keys(), dtype=np.int64)
            nonjnxKeysArray=np.asarray(np.setdiff1d(range(V),jnxKeysArray), dtype=np.int64)
            if USE_CYTHON:
                lambdap=fu.initLambda(K,V,nonjnxKeysArray,jnxKeysArray,mu,eta,existence_threshold,BPRIME_UNCERTAINTY,LAMBDAP_UNCERTAINTY,threads,term2exonsCovered,term2exonsCoveredLengths)
            else:
                lambdap=initLambda(K,V,term2exon,term2junction,mu,eta,existence_threshold, BPRIME_UNCERTAINTY)
            '''initialize a: var param for t''' 
            a1 = np.ones(K,dtype=np.float32)
            a2 = omega*np.ones(K,dtype=np.float32)
            
            '''initialize rho: var param for pi'''
            rho1= np.float32(r+np.sum(mu,1))
            rho2= np.float32(s+E-np.sum(mu,1))
            
            
            T = np.zeros(M,dtype=np.uint32)
            T = T + K
            # local parameters          
            gamma1 = np.zeros((M,max(T)),dtype=np.float32)
            gamma2 = np.zeros((M,max(T)),dtype=np.float32)
            zeta = np.zeros((M,max(T),K),dtype=np.float32)
            phi = np.zeros((M,V,max(T)),dtype=np.float32)
#             phi = np.zeros((M,max(N),max(T)),dtype=np.float32)
            E_psi = np.zeros((M,max(T)),dtype=np.float32)
            
            # initialize local param (+related expectations) for all individuals
    
            '''initialize zeta: var param for c'''
            # we initialize with zeros and ones
            # in individual j, zeta is one for l and k such that isoform k has the l-th larger 
            # average proportion, as shown from the compatibility matrix. It is 0 otherwise
            for j in range(M):
                # build compatibility matrix of dimension (M,max(N),K)
                comp=compatibility_v(b,K,j,C,V)
                # average on the reads within each document, which yields a proportion for each couple (individual,isoform)       
                Pjk = np.sum(comp,0)
                Pjk = (Pjk.T/np.count_nonzero(C[j])).T
                
                # rank isoforms by decreasing proportions in each individual
                cj = np.argsort(-Pjk,0)
                
                if FAST_SOLVER:
                    flag_matrix = np.zeros((V+1,))
                    sol_matrix = np.zeros((V+1, comp.shape[1]))

                for l in range(K):
                    zeta[j][l,cj[l]]=1
                for v in range(V):
                    # i,j fixed, for all k, P(z_{ji}=k) = sum_l phi[j,i,l]*zeta[j,l,k]
                    # solve linear system of equations to get phi
                    if C[j,v]>0:
                        if FAST_SOLVER:
                            if(flag_matrix[v] == 1):
                                phi[j][v,:] = sol_matrix[v, :]
                            else:
                                phi[j][v,:]=np.linalg.solve(zeta[j][:,:].T,comp[v,:]) # prob a read comes from a particular isoform
                                flag_matrix[v] = 1
                                sol_matrix[v, :] = phi[j][v,:]
                        else:
                            phi[j][v,:]=np.linalg.solve(zeta[j][:,:].T,comp[v,:]) # prob a read comes from a particular isoform

                assert np.allclose(np.sum(phi[j][0:V][np.any(phi[j],axis=1)],1),1)

        
            '''initialize gamma1,gamma2: var param for psi'''
            for j in range(M):
                for l in range(K):
                    gamma1[j][l]=1 + np.dot(phi[j][:,l],C[j])
                    gamma2[j][l]=alpha + np.sum(np.dot(phi[j][:,(l+1):].T,C[j]))
                
                # update expectation depending on gamma: log beta   
                E_psi[j][:] = expectation_log_beta(gamma1[j][:],gamma2[j][:])
                    
            param = {'mu': mu, 'lambdap':lambdap, 'a1':a1, 'a2':a2, 'rho1':rho1, 'rho2':rho2, 'zeta':zeta, 'phi':phi ,'gamma1':gamma1, 'gamma2':gamma2}
        return param,folder
        
class HDP:
    def __init__(self,C,term2exon_file=None,alpha=1.0,omega=1.0,eta=1.0,r=1.0,s=5.0,K=5,initialize_from=None, 
                 exon_lengths=None, read_length=0, compress_input=0, monte_carlo_mu=0, log=0, existence_threshold=0.5, threads=1, maxiso=600, splicegraph=1, kappa=1,exon_info_file=None):
        
        # 1. data
        ofh=open(term2exon_file,'r')
        term2exon_read=ofh.readlines()
        ofh.close()
        
        if exon_info_file is not None:
            exon_pos_and_lengths = np.genfromtxt(exon_info_file, delimiter=':')
            self.exon_pos_and_lengths = exon_pos_and_lengths
        else: 
            exon_pos_and_lengths = None
            self.exon_pos_and_lengths = None
        ''' which exon does the read term map to, does it map to an exon junction, to which one'''

        term2exon,isjunction,term2junction,term2dis,term2ends,max_v=tf.term_exon_junction_init(term2exon_read,1)

        # compress input
        if compress_input:
            exon_to_readterms = OrderedDict()
                    
            uniqctr = 0
            newisjunction = []
            newterm2junction = dict()
            for idx in range(len(term2exon)):
                exon = term2exon[idx]
                if isjunction[idx]==1:
                    if term2junction[idx] not in exon_to_readterms:
                        newisjunction.append(isjunction[idx])
                        newterm2junction[uniqctr]=term2junction[idx]
                        uniqctr += 1
                        exon_to_readterms[term2junction[idx]]=[]
                    exon_to_readterms[term2junction[idx]].append(idx)
                else: 
                    if exon not in exon_to_readterms:
                        newisjunction.append(isjunction[idx])
                        exon_to_readterms[exon]=[]
                        uniqctr += 1
                    exon_to_readterms[exon].append(idx)
            
            newisjunction = np.asarray(newisjunction,dtype=np.uint32)
            newterm2exon = np.zeros(len(exon_to_readterms.keys()),dtype=np.uint32)
            newC = np.zeros((len(C),len(exon_to_readterms.keys())),dtype=np.float32)
            
            ctr = 0
            for person in C:
                idx = 0
                for exon in exon_to_readterms.keys():
                    if type(exon) is tuple:
                        newterm2exon[idx]=exon[0]
                    else:
                        newterm2exon[idx]=exon
                    for readterm in exon_to_readterms[exon]:
                        newC[ctr,idx]+=person[readterm]
                    idx += 1
                ctr += 1
                
            C = newC
            isjunction = newisjunction
            term2exon = newterm2exon
            term2junction = newterm2junction
        
        # fix counts per exon lengths
        if exon_lengths is not None:
            # get the max length from an exon or junction for normalizing purposes
            max_exon = np.float32(max(exon_lengths))
            for term in term2junction.keys():
                if(len(term2junction[term])==2):
                    max_exon=max(max_exon,np.float32(read_length))
                else: 
                    # get distance between first and last 
                    max_exon=max(max_exon,np.float32(read_length)-np.float32(np.sum(exon_lengths[term2junction[term][1]:term2junction[term][-1]])))
            for person in C:
                exonidx = 0
                for read_term_idx in range(len(term2exon)):
                    if isjunction[read_term_idx]:
                        # could pass k exons
                        jnx_exons = term2junction[read_term_idx]
                        if(len(jnx_exons)==2):
                            person[read_term_idx]=person[read_term_idx]*(max_exon/read_length)
                        else: 
                            # get distance between first and last 
                            person[read_term_idx]=person[read_term_idx]*(max_exon/(read_length-np.sum(exon_lengths[jnx_exons[1]:jnx_exons[-1]])))
                    else:
                        # fits in a single exon
                        person[read_term_idx]=person[read_term_idx]*(max_exon/exon_lengths[exonidx])
                        exonidx += 1                            
        
        junction_set=set(term2junction.values())    
        
        self.C = C
        self.junction_set=junction_set
        self.term2junction=term2junction
        self.term2dis = term2dis
        self.term2exon = term2exon
        self.max_v = max_v
        if len(term2junction)>0:
            self.E=max(max(term2exon),max([item for sublist in term2junction.values() for item in sublist]))+1
        else :
            self.E=max(term2exon)+1
        M = C.shape[0]; self.M = M # number of individuals
        N = np.zeros(M,dtype=np.int32)
        pidx = 0
        for person in C:
            N[pidx] = np.sum(np.ceil(person))
            pidx += 1
            
        self.N = N # this is a vector, with the number of reads for each individual
        V = C.shape[1]; self.V = V # size of read dictionary, or number of terms
        
        self.monte_carlo_mu = monte_carlo_mu
        self.log = log
        
        # convert from aggregated counts to individual observations
        N_max = N.max()
        X = np.zeros((M,N_max),dtype=np.int32)
        X[:,:] = -1
        for j in range(M):
            X[j,:N[j]] = tf.get_vector_from_bincounts(np.ceil(C[j,:]))
        self.X = X
        
        # 2. fixed model parameters
        self.alpha = alpha
        self.eta = eta
        self.omega = omega
        self.r = r
        self.s = s
        self.kappa = kappa

        # 3. variational distribution
        self.var_dist = VarDist(V,alpha,eta,omega,r,s,X,N,M,K,term2exon,term2junction,junction_set,initialize_from,existence_threshold,
                                threads,maxiso,splicegraph,exon_pos_and_lengths,term2dis,max_v,C)
    
    
    def update(self,output_dir, N_iter, forget_rate, delay, update,log,num_iterations_to_save,threads,converge,stochastic,batchsize,tmpfolder,
               existence_threshold,term2exonsCovered,term2exonsCoveredLengths,C):
        folder=self.var_dist.folder
        with Parallel(n_jobs=threads,temp_folder=tmpfolder,max_nbytes=None, backend="threading") as parallel:    # other backend "multiprocessing"
            global ROUNDS_IN_BETWEEN_PROPOSAL
            
            # num iterations to save
            itsToSave=np.linspace(1,N_iter,num=num_iterations_to_save)
            monte_carlo_mu=self.monte_carlo_mu
                        
            # dimensions, same notations as in init
            M=self.M
            
            K=self.var_dist.K
            T = np.zeros(M,dtype=np.uint32)
            T = T + K
            N=self.N
            V=self.V
            E=self.E
            # data
            X=self.X

            # variational parameters (initial global)
            param=self.var_dist.param
            a1=param['a1']
            a2=param['a2']
            rho1=param['rho1']
            rho2=param['rho2']
            lambdap=param['lambdap']
            mu=param['mu']
            # variational parameters (initial local)
            zeta=param['zeta']
            phi=param['phi']
            gamma1=param['gamma1']
            gamma2=param['gamma2']

            np.seterr(divide='ignore', invalid='ignore')
            Cbool = np.nan_to_num(C/C)
            np.seterr(divide='warn', invalid='warn')
            
            # hyperparameters
            alpha=self.alpha
            eta=self.eta
            kappa=self.kappa
            omega=self.omega
            r=self.r
            s=self.s
            max_v=self.max_v
            term2junction=self.term2junction
            term2exon=self.term2exon
            exon_pos_and_lengths = self.exon_pos_and_lengths
            term2dis=self.term2dis
            
            # speed up mu computation
            allterms=np.arange(len(term2exon))
            jnxKeysArray=np.asarray(term2junction.keys(), dtype=np.int64)
            nonjnxKeysArray=np.asarray(np.setdiff1d(range(V),jnxKeysArray), dtype=np.int64)
        
            exonToTermsNonJnc=dict()
            for term in nonjnxKeysArray:
                if term2exon[term] not in exonToTermsNonJnc:
                    exonToTermsNonJnc[term2exon[term]]=np.zeros(0,dtype=np.int32)
                appended = np.append(exonToTermsNonJnc[term2exon[term]],[term],axis=0)
                exonToTermsNonJnc[term2exon[term]]=appended
            
            exonToTermsJncs=dict()
            for term in jnxKeysArray:
                for exon in term2junction[term]:
                    if exon not in exonToTermsJncs:
                        exonToTermsJncs[exon]=np.zeros(0,dtype=np.int32)
                    appended=np.append(exonToTermsJncs[exon],[term],axis=0)
                    exonToTermsJncs[exon]=appended

            E_psi=(np.zeros((M,max(T)),dtype=np.float32))
            
            if update is None:
                # update all variables, default behaviour
                update_zeta=True
                update_phi=True
                update_gamma=True
                update_lambda=True
                update_a=True
                update_rho=True
                update_mu=True
            else:
                update_zeta=int(update[0])
                update_phi=int(update[1])
                update_gamma=int(update[2])
                update_lambda=int(update[3])
                update_a=int(update[4])
                update_rho=int(update[5])
                update_mu=int(update[6])
            
            ''' compute expectation relevant to local parameters, calls functions from helper file compute_expectations.py '''
            # log pi, expectation of a log[beta distribution] 
            E_pi = expectation_log_beta(rho1,rho2) 
            # log 1-pi
            E_1minuspi = expectation_log_beta(rho2,rho1)         
            # log beta
            E_beta = expectation_log_dirichlet(lambdap)   # care with using E_beta, read terms that don't appear == read terms for isoforms which only a single read term appears
            # log t
            E_t = expectation_log_sigma_beta(a1,a2)

            for j in range(M):
                E_psi[j][:] = expectation_log_sigma_beta(gamma1[j][:],gamma2[j][:])

            # stochastic variational inference

#            last_gamma1=copy.deepcopy(gamma1)
#            last_gamma2=copy.deepcopy(gamma2)
#            last_E_psi=copy.deepcopy(E_psi)
#            last_zeta=copy.deepcopy(zeta)
#            last_phi=copy.deepcopy(phi)
#            last_E_t=copy.deepcopy(E_t)
#            last_E_beta=copy.deepcopy(E_beta)
            last_a1=copy.deepcopy(a1)
            last_a2=copy.deepcopy(a2)
            last_lambdap=copy.deepcopy(lambdap)
            last_mu=copy.deepcopy(mu)
            
            if update_zeta:
                zeta_final=[]
                zeta_final.append(zeta)
            else:
                zeta_final=zeta
                 
            if update_phi:
                phi_final=[]
                phi_final.append(phi)
            else:
                phi_final=phi
             
            if update_gamma:
                gamma1_final=[] 
                gamma1_final.append(gamma1)
                gamma2_final=[] 
                gamma2_final.append(gamma2)
            else:
                gamma1_final=gamma1
                gamma2_final=gamma2
                                                  
            # global parameters, final updates
            if update_mu:
                mu_final=[]
                mu_final.append(mu)
            else:
                mu_final= mu
            if update_rho:
                rho1_final=[]
                rho2_final=[]
                rho1_final.append(rho1)
                rho2_final.append(rho2)
            else:
                rho1_final= rho1
                rho2_final= rho2
            if update_a:
                a1_final=[]
                a2_final=[]
                a1_final.append(a1)
                a2_final.append(a2)
            else:
                a1_final= a1
                a2_final= a2
            if update_lambda:
                lambda_final=[]
                lambda_final.append(lambdap)
            else:
                lambda_final= lambdap
                
            # caching for fast lambda update
            starts=np.zeros((M,V),dtype=np.int32)
            ends=np.zeros((M,V),dtype=np.int32)
            log_likelihood=np.zeros(N_iter+100,dtype=np.float32)
            startsj=np.zeros(M,dtype=np.int32)
            endsj=np.zeros(M,dtype=np.int32)
            for v in range(V):
                (startsj,endsj)=incrementXCntrs(startsj,endsj,v,X)
                starts[:,v]=startsj[:]
                ends[:,v]=endsj[:]
            
            '''Start the main loop'''
            savedItCtr=0
            it=0
            propit=0
            D=1
            bad_k=[]
            bad_k=np.asarray(bad_k,dtype=np.int)

            while True:
                update_start_time = time()
                if it%25==0:
                    print >> sys.stderr, "iteration " + str(it)
                if log:
                    print >> sys.stderr, "iteration " + str(it)
                    print >> sys.stderr, "\n\n"
                    print >> sys.stderr, "zeta"
                    print >> sys.stderr, zeta
                    print >> sys.stderr, "\ngamma1"
                    print >> sys.stderr, gamma1
                    print >> sys.stderr, "\ngamma2"
                    print >> sys.stderr, gamma2
                    print >> sys.stderr, "\nmu"
                    print >> sys.stderr, mu
                    print >> sys.stderr, "\na1"
                    print >> sys.stderr, a1
                    print >> sys.stderr, "\na2"
                    print >> sys.stderr, a2
                    print >> sys.stderr, "\nlambdap"
                    print >> sys.stderr, lambdap
                
                E_freqIso = np.zeros((zeta.shape[0],gamma1.shape[1]),np.float32)
                for j in range(zeta.shape[0]):
                    E_freqIso[j][:] = expectation_sigma_beta(gamma1[j][:],gamma2[j][:])
                    E_freqIso[j] = E_freqIso[j]/np.sum(E_freqIso[j])    
                global_iso_local_freq=np.einsum('jl,jlk->jk',E_freqIso,zeta)
                
                time_t=time()
                (conv,conv_like) = converged_likelihood(log,log_likelihood,zeta,phi,lambdap,term2exon,term2junction,converge,it,N_iter,stochastic,
                                                        N,X,mu,V,propit,term2exonsCovered,term2exonsCoveredLengths,bad_k,global_iso_local_freq,threads,C)
                if log:
                    print >> sys.stderr, 'converged likelihood took ' + str(time()-time_t)

                if conv:
                    print 'likelihood: ' + str(conv_like)
                    break;
                tstart = time()
                
                if stochastic:
                    inds = np.random.choice(range(M),threads,False)
                    grps = []
                    for i in range(threads):
                        grps.append(np.array([inds[i]]));
                else:
                    grps=np.array_split(np.arange(M,dtype=np.int),threads)
                
                # individual plate is the same, just on different individuals
                if parallel.n_jobs==1:
                    if(USE_CYTHON_UPDATE):
                        chunks = [fu.updateIndividualPlate(grps[i],X,N,update_gamma,update_zeta,update_phi,update_a,update_lambda,gamma1,gamma2,phi,
                                                           zeta,a1,a2,omega,E_psi,E_beta,alpha,E_t,M,K,lambdap,V,mu,term2exon,term2junction,eta,
                                                           stochastic,converge,INDIVIDUAL_PLATE_CONVERGE,INDIVIDUAL_PLATE_MAXIT) for i in range(len(grps))]
                    else:
                        chunks = [updateIndividualPlate(grps[i],X,N,update_gamma,update_zeta,update_phi,update_a,update_lambda,gamma1,gamma2,phi,
                                                        zeta,a1,a2,omega,E_psi,E_beta,alpha,E_t,M,K,lambdap,V,mu,term2exon,term2junction,eta,
                                                        stochastic,converge,C,Cbool) for i in range(len(grps))]
                else: 
                    chunks = parallel(delayed(updateIndividualPlate)(np.arange(len(grps[i])),X[grps[i]],N[grps[i]],update_gamma,update_zeta,update_phi,update_a,update_lambda,
                                                                     gamma1[grps[i]],gamma2[grps[i]],phi[grps[i]],zeta[grps[i]],a1,a2,omega,E_psi[grps[i]],E_beta, 
                                                                     alpha, E_t, M, K, lambdap, V,mu,term2exon,term2junction, eta, stochastic,converge,C,Cbool) for i in range(len(grps)))

                       

                for cnt in range(len(grps)):
                    chnk_ctr = 0
                    for j in grps[cnt]:
                        gamma1[j]=chunks[cnt][0][chnk_ctr]
                        gamma2[j]=chunks[cnt][1][chnk_ctr]
                        zeta[j]=chunks[cnt][2][chnk_ctr]
                        phi[j]=chunks[cnt][3][chnk_ctr]
                        chnk_ctr+=1
                for j in range(M):
                    E_psi[j][gamma1[j]!=0] = expectation_log_sigma_beta(gamma1[j][gamma1[j]!=0],gamma2[j][gamma2[j]!=0])
                if log:
                    print >> sys.stderr, 'updating individual plate took ' + str(time()-tstart)
                
                tstart = time()
                if stochastic:
                    (a1,a2,E_t,lambdap,E_beta,mu) = update_stochastic(log,it,delay,forget_rate,M,update_a,a1,a2,K,zeta,omega,last_a1,last_a2,update_lambda,V,mu,term2exon,term2junction,threads,parallel,phi,
                                                                      lambdap,eta,X,starts,ends,last_lambdap,update_mu,monte_carlo_mu,r,s,nonjnxKeysArray,exonToTermsJncs,jnxKeysArray,exonToTermsNonJnc,
                                                                      E,last_mu,D,kappa,bad_k,E_t,term2exonsCovered,term2exonsCoveredLengths,C)
                else:
                    (a1,a2,E_t,lambdap,E_beta,mu) = update_deterministic(M,update_a,a1,a2,K,zeta,omega,update_lambda,V,mu,term2exon,term2junction,threads,parallel,phi,
                         lambdap,eta,X,starts,ends,update_mu,monte_carlo_mu,r,s,nonjnxKeysArray,exonToTermsJncs,jnxKeysArray,exonToTermsNonJnc,
                         E,kappa,bad_k,E_t,term2exonsCovered,term2exonsCoveredLengths,C)

                if log:
                    print >> sys.stderr, 'updating lambda and mu took ' + str(time()-tstart)
                
                if it in itsToSave:       
                    if update_mu:
                        mu_final.append(mu)      
                    if update_zeta:
                        zeta_final.append(zeta)
                    if update_phi:
                        phi_final.append(phi)
                    if update_gamma:
                        gamma1_final.append(gamma1)
                        gamma2_final.append(gamma2)
                    if update_a:
                        a1_final.append(a1) 
                        a2_final.append(a2)
                    if update_lambda:
                        lambda_final.append(lambdap)
                    savedItCtr+=1
                    
                tstart = time()
                phiSum = [np.sum(np.tensordot(phi[j],zeta[j],axes=([1],[0])),axis=0) for j in range(M)]
                phiNorm = [phiSum[j]/np.sum(phiSum[j]) for j in range(M)]
                bad_k=[]
                for k in range(K):
                    num_good = 0
                    for j in range(M):
                        if phiNorm[j][k]>0.01:
                            num_good+=1
                    if num_good==0:
                        bad_k.append(k)
                bad_k = np.asarray(bad_k,dtype=np.int)
                if log:
                    print >> sys.stderr, 'before proposals ' + str(time()-tstart)
                tstart = time()
                (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K) = removeBadIsoforms(bad_k,gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K)
#                 time_logging[4]+=time()-time_t
                if it % ROUNDS_IN_BETWEEN_PROPOSAL == 0 and propit < MAX_NUMBER_PROPOSALS:
                    E_freqIso[:,:]=0
                    for j in range(zeta.shape[0]):
                        E_freqIso[j][:] = expectation_sigma_beta(gamma1[j][:],gamma2[j][:])
                        E_freqIso[j] = E_freqIso[j]/np.sum(E_freqIso[j])    
                    global_iso_local_freq=np.einsum('jl,jlk->jk',E_freqIso,zeta)
#                     if USE_CYTHON:
#                         (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,bad_k) = fu.proposeNewIsoformsAndMerge(bad_k,gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,V,term2exon,term2junction,N,X,r,s,existence_threshold,
#                                                                                                                                      omega,M,alpha,eta,term2exonsCovered,term2exonsCoveredLengths,it,global_iso_local_freq,threads,jnxKeysArray,nonjnxKeysArray,
#                                                                                                                                      NUM_NEW_ISO,REDUNDANT_ISOS,MERGE_BURN_IN,BPRIME_UNCERTAINTY, LAMBDAP_UNCERTAINTY,EXISTANCE_THRESHOLD,MU_UNCERTAINTY,mus_added,FAST_SOLVER, random,gauss)
#                     else:
                    (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,bad_k) = proposeNewIsoformsAndMerge(log,bad_k,gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,
                                                                                                                              lambdap,mu,E_beta,K,V,term2exon,term2junction,N,X,r,s,
                                                                                                                              existence_threshold,omega,M,alpha,eta,term2exonsCovered,
                                                                                                                              term2exonsCoveredLengths,it,global_iso_local_freq,threads,
                                                                                                                              jnxKeysArray,nonjnxKeysArray,C)
#                     time_logging[5]+=time()-time_t
                    propit+=1
                    (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K) = removeBadIsoforms(bad_k,gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K)

                if log:
                    print >> sys.stderr,  'removing and proposing took ' + str(time()-tstart)
#                 last_gamma1=copy.deepcopy(gamma1)
#                 last_gamma2=copy.deepcopy(gamma2)
#                 last_E_psi=copy.deepcopy(E_psi)
#                 last_zeta=copy.deepcopy(zeta)
#                 last_phi=copy.deepcopy(phi)
#                 last_E_beta=copy.deepcopy(E_beta)
#                 last_E_t=copy.deepcopy(E_t)
#                 last_rho1=copy.deepcopy(rho1)
#                 last_rho2=copy.deepcopy(rho2)
                last_a1=copy.deepcopy(a1)
                last_a2=copy.deepcopy(a2)
                last_lambdap=copy.deepcopy(lambdap)
                last_mu=copy.deepcopy(mu)
                it+=1
                if log:
                    print >> sys.stderr,  'full iteration took ' + str(time()-update_start_time)
                
                # clean isoforms
        for i in range(savedItCtr,len(itsToSave)):
            if update_mu:
                mu_final.append(mu)      
            if update_zeta:
                zeta_final.append(zeta)
            if update_phi:
                phi_final.append(phi)
            if update_gamma:
                gamma1_final.append(gamma1)
                gamma2_final.append(gamma2)
            if update_a:
                a1_final.append(a1) 
                a2_final.append(a2)
            if update_lambda:
                lambda_final.append(lambdap)
        try:
            shutil.rmtree(folder)
        except OSError:
            pass  # this can sometimes fail under Windows
        result = {'mu':mu_final,'lambda':lambda_final,'a1':a1_final,'a2':a2_final,'gamma1':gamma1_final,'gamma2':gamma2_final} 
        gc.collect() # garbage collection
        try:
            with gzip.open(output_dir+'outfile.pkz','wb') as f:
                pickle.dump((mu_final,lambda_final,a1_final,a2_final,phi_final,zeta_final,gamma1_final, gamma2_final),f)
        except:
            try:
                print 'caught error while dumping gzipped pickle, trying uncompressed'
                with file.open(output_dir+'outfile.pkl','wb') as f:
                    pickle.dump((mu_final,lambda_final,a1_final,a2_final,phi_final,zeta_final,gamma1_final, gamma2_final),f)
            except:
                print 'caught error while dumping pickle, trying numpy and only saving LAST iteration'
                np.savez_compressed(output_dir+'outfile.npz', mu=mu,lambdap=lambdap,a1=a1,a2=a2,gamma1=gamma1,gamma2=gamma2,zeta=zeta,phi=phi)
            

        return result

def proposeNewIsoformsAndMerge(log,bad_k,gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,V,
                               term2exon,term2junction,N,X,r,s,existence_threshold,omega,M,alpha, eta,
                               term2exonsCovered,term2exonsCoveredLengths,it,E_freqIso,numthreads,jnxKeysArray,nonjnxKeysArray,C):
    
#     timing = time()
    global NUM_NEW_ISO
    global REDUNDANT_ISOS
    global MERGE_BURN_IN
    NUM_NEW_ISO_LOCAL = NUM_NEW_ISO
    tstart = time()
    if it > MERGE_BURN_IN:
        global BPRIME_UNCERTAINTY
        global LAMBDAP_UNCERTAINTY
        (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,bad_k) = mergeIsoforms(gamma1,gamma2,E_psi,zeta,phi,a1,a2,
                                                                                                     E_t,rho1,rho2,lambdap,mu,E_beta,K,bad_k,
                                                                                                     X,N,V,term2exonsCovered,term2exonsCoveredLengths,
                                                                                                     term2exon,term2junction,eta,existence_threshold,
                                                                                                     E_freqIso,numthreads,jnxKeysArray,nonjnxKeysArray,C)

    if log:
        print >> sys.stderr,  'merging took ' + str(time()-tstart)
    tstart = time()
    if USE_CYTHON:
        b = fu.construct_b_from_bprime(V,mu,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
    else:
        b = construct_b_from_bprime(V,mu,term2exon,term2junction,1)
#     time_logging[3]+=time()-time_t
    num_iso = mu.shape[0]
    E = mu.shape[1]
    new_rows_in_mu = NUM_NEW_ISO_LOCAL-len(bad_k)
    if new_rows_in_mu>0:
        mu = np.append(mu, np.zeros((new_rows_in_mu,E),mu.dtype), axis=0)
    mu[mu==1]=0.95
    mu[mu==0]=0.05
    
    ks_to_insert=[]
    for k in bad_k:
        if len(ks_to_insert)<NUM_NEW_ISO_LOCAL:
            ks_to_insert.append(k)

    cnt=0    
    while len(ks_to_insert)<NUM_NEW_ISO_LOCAL:
        ks_to_insert.append(num_iso+cnt)
        cnt+=1
        
    badterm_to_cnt = dict()
    total = 0
            
    read_assignments = np.einsum('jvl,jlk->jvk',phi,zeta)
    for j in range(len(N)): # over individuals
        for v in range(V): # over reads
            if C[j,v]>0:
                it_likelihood = np.sum(read_assignments[j,v,:]*(b[:,v]))
                if it_likelihood < 0.5:
                    if v not in badterm_to_cnt:
                        badterm_to_cnt[v]=0
                    badterm_to_cnt[v]+=C[j,v]
                    total += C[j,v]
    

    # the first third, is just random exons        
    k=num_iso-len(bad_k)
    total_added = 0
    ks_inserted = []
    kindx=0
    
    new_bad_k=[]
    for k in range(len(bad_k)):
        if bad_k[k]<len(mu):
            new_bad_k.append(bad_k[k])
    
    # IMPLEMENT THIS PART IN CYTHON
    for k in ks_to_insert:            
        thismu=np.ones((E),dtype=np.float32)
        thismu[:]=1-MU_UNCERTAINTY
        j = random.randint(0,M-1)
        for ctr in range(np.int(np.ceil(np.abs(gauss(E,np.sqrt(E)))))):
            if total==0 or (kindx>=NUM_NEW_ISO_LOCAL-REDUNDANT_ISOS):  
                randomJncIdx = X[j,random.randint(0, len(X[j])-1)]
            else:
                randomJnc = random.randint(0, total-1)
                randomJncIdx = getJnc(badterm_to_cnt,randomJnc)
            if randomJncIdx in term2junction:
                randomJnc = term2junction[randomJncIdx]
                for jncctr in range(len(randomJnc)-1):
                    thismu[randomJnc[jncctr]]=1-MU_UNCERTAINTY
                    thismu[randomJnc[jncctr+1]]=1-MU_UNCERTAINTY
                    thismu[randomJnc[jncctr]+1:randomJnc[jncctr+1]]=MU_UNCERTAINTY
            else:
                randomTerm = term2exon[randomJncIdx]
                thismu[randomTerm]=1-MU_UNCERTAINTY
        inarray = 0;
        wrapper = np.zeros((1,E),dtype=np.float32)
        wrapper[0]=thismu
        kindx+=1
#         

        if USE_CYTHON:
            b = fu.construct_b_from_bprime(V,wrapper,term2exonsCovered,term2exonsCoveredLengths,0,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
        else:
            b = construct_b_from_bprime(V,wrapper,term2exon,term2junction,0)

        for mui in mus_added:
            if np.array_equal(mui, b[0]):
                inarray+=1
        if inarray == 0:
            mus_added.append(np.array(b[0], copy=True) )
            thismu[thismu<MU_UNCERTAINTY]=MU_UNCERTAINTY
            mu[k,:]=thismu  
            total_added+=1
            ks_inserted.append(k)
        elif k not in bad_k:
            new_bad_k.append(k)

    NUM_NEW_ISO_LOCAL=total_added
    if log:
        print >> sys.stderr,  'figuring out mu took ' + str(time()-tstart)
    tstart=time()
    if USE_CYTHON:
        b = fu.construct_b_from_bprime(V,mu,term2exonsCovered,term2exonsCoveredLengths,0,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
    else:
        b = construct_b_from_bprime(V,mu,term2exon,term2junction,0)

    bad_k=np.asarray(new_bad_k, dtype=np.int)
    bad_k=np.setdiff1d(bad_k,ks_inserted)
    if (len(mu)-len(lambdap))>0:
        lambdap=np.append(lambdap, np.zeros(((len(mu)-len(lambdap)),V),dtype=np.float32), axis=0)
    
    
    for k in ks_inserted:  
        for v in range(V):
            num_inconsistent=0
            if v in term2junction:
                jnx = term2junction[v]

                if USE_CYTHON:
                    num_inconsistent=fu.jnxTupleMuInconsistent(term2exonsCovered[v],term2exonsCoveredLengths[v],mu[k],EXISTANCE_THRESHOLD)
                else:
                    num_inconsistent=jnxTupleMuInconsistent(jnx,mu[k])
            else:
                jnx = term2exon[v]
                if mu[k][jnx]<existence_threshold:
                    num_inconsistent=1
            
            if num_inconsistent==0:
                lambdap[k,v] = eta
            else:
                lambdap[k,v] = LAMBDAP_UNCERTAINTY
    lambdap[lambdap<0.05]=0.05
    lambdap[bad_k]=0.05

    timing = time()
    E_beta = np.append(E_beta, np.zeros((NUM_NEW_ISO_LOCAL,V),E_beta.dtype), axis=0)
    E_beta = expectation_log_dirichlet(lambdap)
    
    K=mu.shape[0]
    rho1= np.float32(r+np.sum(mu,1))
    rho2= np.float32(s+E-np.sum(mu,1))
    a1 = np.ones(K,dtype=np.float32)
    a2 = omega*np.ones(K,dtype=np.float32)
    E_t = expectation_log_sigma_beta(a1,a2) 
    
    gamma1=np.zeros((M,K),dtype=np.float32)
    gamma2=np.zeros((M,K),dtype=np.float32)
    zeta=np.zeros((M,K,K),dtype=np.float32)
    phi=np.zeros((M,V,K),dtype=np.float32)
    E_psi=np.zeros((M,K),dtype=np.float32)
    
    for j in range(M): 
        comp=compatibility_v(b,K,j,C,V)
        # average on the reads within each document, which yields a proportion for each couple (individual,isoform)       
        Pjk = np.sum(comp,0)
        Pjk = (Pjk.T/np.count_nonzero(C[j])).T
        
        # rank isoforms by decreasing proportions in each individual
        cj = np.argsort(-Pjk,0)
        if FAST_SOLVER:
            flag_matrix = np.zeros((V+1,))
            sol_matrix = np.zeros((V+1, comp.shape[1]))

        for l in range(K):
            zeta[j][l,cj[l]]=1
        for v in range(V):
            # i,j fixed, for all k, P(z_{ji}=k) = sum_l phi[j,i,l]*zeta[j,l,k]
            # solve linear system of equations to get phi
            if C[j,v]>0:
                if FAST_SOLVER:
                    if(flag_matrix[v] == 1):
                        phi[j][v,:] = sol_matrix[v, :]
                    else:
                        phi[j][v,:]=np.linalg.solve(zeta[j][:,:].T,comp[v,:]) # prob a read comes from a particular isoform
                        flag_matrix[v] = 1
                        sol_matrix[v, :] = phi[j][v,:]
                else:
                    phi[j][v,:]=np.linalg.solve(zeta[j][:,:].T,comp[v,:]) # prob a read comes from a particular isoform

        assert np.allclose(np.sum(phi[j][0:V][np.any(phi[j],axis=1)],1),1)

    timing = time()

    for j in range(M):
        for l in range(K):
            gamma1[j][l]=1 + np.dot(phi[j][:,l],C[j])
            gamma2[j][l]=alpha + np.sum(np.dot(phi[j][:,(l+1):].T,C[j]))
        
        # update expectation depending on gamma: log beta   
        E_psi[j][:] = expectation_log_beta(gamma1[j][:],gamma2[j][:])

    
    if log:
        print >> sys.stderr,  'reinit took ' + str(time()-tstart)
    return (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,bad_k)

def mergeIsoforms(gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,bad_k,X,N,V,term2exonsCovered,
                  term2exonsCoveredLengths,term2exon,term2junction,eta,existence_threshold,E_freqIso,numthreads,jnxKeysArray,nonjnxKeysArray,C):
    global BPRIME_UNCERTAINTY
    global LAMBDAP_UNCERTAINTY
    
    kit=range(len(gamma1[0]))
    kit2=range(len(gamma1[0]))
    random.shuffle(kit)
    random.shuffle(kit2)
    bad_k_list = bad_k.tolist()
    


    if USE_CYTHON:
        b = fu.construct_b_from_bprime(V,mu,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
    else:
        b = construct_b_from_bprime(V,mu,term2exon,term2junction,1)
    
    for k in kit:
        for k2 in kit2:
            if k>k2 and k not in bad_k_list and k2 not in bad_k_list:
#                 time_t=time()
                
                if USE_CYTHON:
                    prior_l = np.asarray(fu.compute_likelihood_per_person(X,phi,zeta[:,:,[k,k2]],b[[k,k2]]*lambdap[[k,k2]],np.arange(2,dtype=int),E_freqIso[:,[k,k2]],N,numthreads,V,C))
                else:
                    prior_l = compute_likelihood_per_person(X,phi,zeta[:,:,[k,k2]],b[[k,k2]]*lambdap[[k,k2]],np.zeros((0),dtype=int),E_freqIso[:,[k,k2]],N,V,C)
                # try merging k and k2
                merged_mu = np.maximum(mu[k],mu[k2]).reshape((1,len(mu[k])))
                
                merged_zeta = np.sum(zeta[:,:,[k,k2]],axis=2,keepdims=True)

                if USE_CYTHON:
                    merged_lambda = fu.initLambda(1,V,nonjnxKeysArray,jnxKeysArray,merged_mu,eta,existence_threshold,BPRIME_UNCERTAINTY,LAMBDAP_UNCERTAINTY,numthreads,term2exonsCovered,term2exonsCoveredLengths)
                else: 
                    merged_lambda = initLambda(1,V,term2exon,term2junction,mu,eta,existence_threshold, BPRIME_UNCERTAINTY)

                if USE_CYTHON:
                    merged_b = fu.construct_b_from_bprime(V,merged_mu,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
                    post_l = np.asarray(fu.compute_likelihood_per_person(X,phi,merged_zeta,merged_b*merged_lambda,np.arange(1,dtype=int),np.sum(E_freqIso[:,[k,k2]],axis=1,keepdims=True),N,numthreads,V,C))
                else:
                    merged_b = construct_b_from_bprime(V,merged_mu,term2exon,term2junction,1)
                    post_l = compute_likelihood_per_person(X,phi,merged_zeta,merged_b*merged_lambda,np.zeros((0),dtype=int),np.sum(E_freqIso[:,[k,k2]],axis=1,keepdims=True),N,V,C)
                if all((post_l>=prior_l) | np.isclose(post_l, prior_l)):
                    bad_k_list.append(k)
                    mu[k2]=merged_mu
                    zeta[:,:,k2]=np.sum(zeta[:,:,[k,k2]],axis=2,keepdims=False)
                    lambdap[k2]=merged_lambda
                    mu[k]=0
                    zeta[:,:,k]=0
                    lambdap[k]=0
    
    return (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K,np.asarray(bad_k, dtype=int))

def initLambda(K,V,term2exon,term2junction,mu,eta,existence_threshold, BPRIME_UNCERTAINTY):
    initlambdap=np.zeros((K,V), dtype=np.float32)
    for k in range(K):
        for v in range(V):
            num_inconsistent=0
            if v in term2junction:
                jnx = term2junction[v]
                num_inconsistent=jnxTupleMuInconsistent(jnx,mu[k])
            else:
                jnx = term2exon[v]
                if mu[k][jnx]<existence_threshold:
                    num_inconsistent=1
             
            if num_inconsistent==0:
                initlambdap[k,v] = eta
            else:
                initlambdap[k,v] = LAMBDAP_UNCERTAINTY
    initlambdap[initlambdap<LAMBDAP_UNCERTAINTY]=LAMBDAP_UNCERTAINTY
    return initlambdap

def getJnc(badterm_to_cnt,randomJnc):
    ptr = 0
    for term in badterm_to_cnt.keys():
        if randomJnc >= ptr and randomJnc < ptr+badterm_to_cnt[term]:
            return term
        else:
            ptr += badterm_to_cnt[term]
    print "couldn't find term"
    return term

def removeBadIsoforms(bad_k,gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K):
    E_beta[bad_k] = 0
    mu[bad_k] = 0
    lambdap[bad_k] = 0
    rho1[bad_k] = 0
    rho2[bad_k] = 0
    E_t[bad_k] = 0
    a1[bad_k] = 0
    a2[bad_k] = 0
    
    badLocalIso = dict()
    for j in range(len(gamma1)):
        badLocalIso[j]=[]
        if len(bad_k)>0:
            for l in range(len(zeta[j])):
                if(np.sum(zeta[j][l,bad_k])>0.99):
                    if l not in badLocalIso[j]:
                        badLocalIso[j].append(l)
        # ALSO DELETE THE GLOBAL ISOFORMS k
        phi[j,:,badLocalIso[j]]=0
        zeta[j,badLocalIso[j],:]=0
        zeta[j,:,bad_k]=0
        gamma1[j,badLocalIso[j]]=0
        gamma2[j,badLocalIso[j]]=0
        E_psi[j,badLocalIso[j]]=0
        
    return (gamma1,gamma2,E_psi,zeta,phi,a1,a2,E_t,rho1,rho2,lambdap,mu,E_beta,K)
    

def update_stochastic(log,it,delay,forget_rate,M,update_a,a1,a2,K,zeta,omega,last_a1,last_a2,update_lambda,V,mu,term2exon,
                      term2junction,threads,parallel,phi,lambdap,eta,X,starts,ends,last_lambdap,update_mu,monte_carlo_mu,
                      r,s,nonjncterms,exonToTermsJncs,jncterms,exonToTermsNonJnc,E,last_mu,D,kappa,bad_k,E_t,
                      term2exonsCovered,term2exonsCoveredLengths,C):
    # set the step size
    step= pow(it+delay,-forget_rate) 
    '''stochastic version: sample an individual randomly'''
    time_t=time()
    if update_a: 
        '''intermediate a: var param for t'''
#         a1[:]=0
#         a2[:]=0
        for k in range(K):
            if k not in bad_k:
                for j in range(M):
                    T=zeta[j].shape[0]
                    for l in range(T):
                        a1[k] += np.sum(zeta[j][l,k])
                        a2[k] += np.sum(zeta[j][l,k+1:])
                ''' compute expectations relevant to local parameters '''  
                a1[k] = step*(1 + D*a1[k]) + (1-step)*last_a1[k]
                a2[k] = step*(omega + D*a2[k]) + (1-step)*last_a2[k]
        # log t
        E_t[a1!=0] = expectation_log_sigma_beta(a1[a1!=0],a2[a2!=0]) 
        if log:
            print >> sys.stderr, 'updating a took ' + str(time()-time_t)        
    
    time_t=time()
    if update_lambda:
        '''intermediate lambda: var param for beta'''
#         
#         time_t=time()
        if USE_CYTHON:
            b = fu.construct_b_from_bprime(V,mu,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
        else:
            b = construct_b_from_bprime(V,mu,term2exon,term2junction,1)
#         time_logging[3]+=time()-time_t
        firstgrps=np.array_split(np.arange(K,dtype=np.int),threads)
        grps=[]
        for grp in firstgrps:
            grp=np.setdiff1d(grp,bad_k)
            if len(grp)>0:
                grps.append(grp)
        
        if parallel.n_jobs==1:
            if USE_CYTHON:
                lambda_chunks = [fu.fastUpdateLambda(grps[i],M,V,phi,zeta,lambdap,b,eta,X,starts,ends,D,bad_k,C) for i in range(len(grps))]
            else:
                lambda_chunks = [fastUpdateLambda(grps[i],M,V,phi,zeta,lambdap,b,eta,X,starts,ends,D,bad_k,C) for i in range(len(grps))]
        else: 
            if USE_CYTHON:
                lambda_chunks = parallel(delayed(fu.fastUpdateLambda)(np.arange(len(grps[i])),M,V,
                                                                   phi,zeta[:,:,grps[i]],lambdap[grps[i]],b[grps[i]],
                                                                   eta,X,starts,ends,D,np.zeros(0,dtype=np.int64),C) for i in range(len(grps)))
            else:
                lambda_chunks = parallel(delayed(fastUpdateLambda)(np.arange(len(grps[i])),M,V,
                                                                   phi,zeta[:,:,grps[i]],lambdap[grps[i]],b[grps[i]],
                                                                   eta,X,starts,ends,D,np.zeros(0,dtype=np.int64),C) for i in range(len(grps)))
        for cnt in range(len(grps)):
            lambdap[grps[cnt]]=step*lambda_chunks[cnt]+(1-step)*last_lambdap[grps[cnt]]
        
        lambdap[lambdap<LAMBDAP_UNCERTAINTY]=LAMBDAP_UNCERTAINTY
        # log beta
        E_beta = expectation_log_dirichlet(lambdap)    
        if log:
            print >> sys.stderr, 'updating lambda took ' + str(time()-time_t)
        
    time_t=time()
    if update_mu:
        '''intermediate mu: var param for bprime'''
        firstgrps=np.array_split(np.arange(K,dtype=np.int),threads)
        grps=[]
        for grp in grps:
            grp=np.setdiff1d(grp,bad_k)
            if len(grp)>0:
                grps.append(grp)
        if USE_CYTHON:
            mu_chunks = parallel(delayed(fu.updateMuForIsoform)(E,mu[grps[i]],np.arange(len(grps[i])),term2exon,term2junction,eta,V,r,s,E_beta[grps[i]],nonjncterms,
                                                             exonToTermsJncs,jncterms,exonToTermsNonJnc,np.zeros(0,dtype=np.int64),term2exonsCovered,term2exonsCoveredLengths,
                                                             MU_UNCERTAINTY,EXISTANCE_THRESHOLD,BPRIME_UNCERTAINTY) for i in range(len(grps)))
        else:
            if parallel.n_jobs==1:
                mu_chunks = [updateMuForIsoform(E,mu,grps[i],monte_carlo_mu,lambdap,term2exon,term2junction,eta,kappa,V,r,s,E_beta,nonjncterms,exonToTermsJncs,
                                                jncterms,exonToTermsNonJnc,bad_k,term2exonsCovered,term2exonsCoveredLengths) for i in range(len(grps))]
            else: 
                mu_chunks = parallel(delayed(updateMuForIsoform)(E,mu[grps[i]],np.arange(len(grps[i])),monte_carlo_mu,lambdap[grps[i]],term2exon,term2junction,eta,
                                                                 kappa,V,r,s,E_beta[grps[i]],nonjncterms,exonToTermsJncs,jncterms,exonToTermsNonJnc,
                                                                 np.zeros(0,dtype=np.int64),term2exonsCovered,term2exonsCoveredLengths) for i in range(len(grps)))

        for cnt in range(len(grps)):
            mu[grps[cnt],:]=step*mu_chunks[cnt]+(1-step)*last_mu[grps[cnt]]
        if log:
            print >> sys.stderr, 'updating mu took ' + str(time()-time_t)
    return (a1,a2,E_t,lambdap,E_beta,mu)

def update_deterministic(M,update_a,a1,a2,K,zeta,omega,update_lambda,V,mu,term2exon,term2junction,threads,parallel,phi,
                         lambdap,eta,X,starts,ends,update_mu,monte_carlo_mu,r,s,nonjncterms,exonToTermsJncs,jncterms,
                         exonToTermsNonJnc,E,kappa,bad_k,E_t,term2exonsCovered,term2exonsCoveredLengths,C):
    '''loop on all individuals, non stochastic version'''
    
          
    
    if update_a: 
        '''intermediate a: var param for t'''
        a1[:]=0
        a2[:]=0
        for k in range(K):
            for j in range(M):
                T=zeta[j].shape[0]
                for l in range(T):
                    a1[k] += np.sum(zeta[j][l,k])
                    a2[k] += np.sum(zeta[j][l,k+1:])
            ''' compute expectations relevant to local parameters '''  
            a1[k] = (1 + a1[k]) 
            a2[k] = (omega + a2[k])
        # log t
        E_t = expectation_log_sigma_beta(a1[a1!=0],a2[a2!=0]) 

    ''' intermediate global parameters '''
    
    if update_lambda:
        '''intermediate lambda: var param for beta'''
#         b = construct_b_from_bprime(V,mu,term2exon,term2junction)
#         time_t=time()
        if USE_CYTHON:
            b = fu.construct_b_from_bprime(V,mu,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
        else:
            b = construct_b_from_bprime(V,mu,term2exon,term2junction,1)
#         time_logging[3]+=time()-time_t
        grps=np.array_split(np.arange(K,dtype=np.int),threads)
        
        if parallel.n_jobs==1:
            if USE_CYTHON:
                lambda_chunks = [fu.fastUpdateLambda(grps[i],M,V,phi,zeta,lambdap,b,eta,X,starts,ends,1,bad_k,C) for i in range(len(grps))]
            else:
                lambda_chunks = [fastUpdateLambda(grps[i],M,V,phi,zeta,lambdap,b,eta,X,starts,ends,1,bad_k,C) for i in range(len(grps))]
        else:
            if USE_CYTHON:
                lambda_chunks = parallel(delayed(fu.fastUpdateLambda)(grps[i],M,V,phi,zeta,lambdap,b,eta,X,starts,ends,1,bad_k,C) for i in range(len(grps)))
            else:
                lambda_chunks = parallel(delayed(fastUpdateLambda)(grps[i],M,V,phi,zeta,lambdap,b,eta,X,starts,ends,1,bad_k,C) for i in range(len(grps)))
        for cnt in range(len(grps)):
            lambdap[grps[cnt]]=lambda_chunks[cnt]
        
        lambdap[lambdap<LAMBDAP_UNCERTAINTY]=LAMBDAP_UNCERTAINTY


        # log beta
        E_beta = expectation_log_dirichlet(lambdap)    
    

    if update_mu:
        '''intermediate mu: var param for bprime'''
        grps=np.array_split(np.arange(K,dtype=np.int),threads)
        '''loop on all individuals, non stochastic version'''
        if USE_CYTHON:
            mu_chunks = parallel(delayed(fu.updateMuForIsoform)(E,mu,grps[i],term2exon,term2junction,eta,V,r,s,E_beta,nonjncterms,
                                                             exonToTermsJncs,jncterms,exonToTermsNonJnc,bad_k,term2exonsCovered,term2exonsCoveredLengths,MU_UNCERTAINTY,EXISTANCE_THRESHOLD,BPRIME_UNCERTAINTY) for i in range(len(grps)))
        else:
            if parallel.n_jobs==1:
                mu_chunks = [updateMuForIsoform(E,mu,grps[i],monte_carlo_mu,lambdap,term2exon,term2junction,eta,kappa,V,r,s,E_beta,nonjncterms,
                                                             exonToTermsJncs,jncterms,exonToTermsNonJnc,bad_k,term2exonsCovered,term2exonsCoveredLengths) for i in range(len(grps))]
            else: 
                mu_chunks = parallel(delayed(updateMuForIsoform)(E,mu,grps[i],monte_carlo_mu,lambdap,term2exon,term2junction,eta,kappa,V,r,s,E_beta,nonjncterms,
                                                             exonToTermsJncs,jncterms,exonToTermsNonJnc,bad_k,term2exonsCovered,term2exonsCoveredLengths) for i in range(len(grps)))
            
        for cnt in range(len(grps)):
            mu[grps[cnt],:]=mu_chunks[cnt]


    return (a1,a2,E_t,lambdap,E_beta,mu)
    
def converged(list_of_variables,converge,it,N_iter,stochastic):

    if it == N_iter:
        return True
    
    for variable_pair in list_of_variables:
        if np.allclose(variable_pair[0],variable_pair[1],converge)==False:
            return False

        
    return True

def converged_likelihood(log,log_likelihood,zeta,phi,lambdap,term2exon,term2junction,converge,it,N_iter,stochastic,N,X,mu,V,
                         propit,term2exonsCovered,term2exonsCoveredLengths,bad_k,global_iso_local_freq,numthreads,C):

# it starts with each read being assigned at least partially to an isoform with only that exon present (very high likelihood)
# solution moves away from this which is why we get lower likelihoods
    global MIN_NUMBER_ITS
    global MIN_NUMBER_PROPOSALS
    

    time_t=time()

    if USE_CYTHON:
        b = fu.construct_b_from_bprime(V,mu,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
    else:
        b = construct_b_from_bprime(V,mu,term2exon,term2junction,1)

    if USE_CYTHON:
        log_likelihood[it]=np.sum(fu.compute_likelihood_per_person(X,phi,zeta,b*lambdap,np.setdiff1d(np.arange(lambdap.shape[0]),bad_k),
                                                                   global_iso_local_freq,N,numthreads,V,C))/np.float(zeta.shape[0])
    else:
        log_likelihood[it]=np.sum(compute_likelihood_per_person(X,phi,zeta,b*lambdap,bad_k,global_iso_local_freq,N,V,C))/np.float(zeta.shape[0])
    if log:
        print >> sys.stderr,  'computing likelihood inside convered_likelihood took ' + str(time()-time_t)

#     time_ts=time()
    if np.isnan(log_likelihood[it]):
        print "log likelihood is NaN"
    
    # if the AVERAGE likelihood hasn't change much in the last 5 iterations AND the current likelihood is less than the last, stop
#     if it > 5 and np.abs(np.max(log_likelihood[it-4:it+1])-np.min(log_likelihood[it-5:it]))<converge:

    if it<5:
        start=0
    else:
        start=it-5
    if ((it > MIN_NUMBER_ITS and propit >= MIN_NUMBER_PROPOSALS and np.abs(np.average(log_likelihood[start:it])-log_likelihood[it])<converge) and (log_likelihood[it]<log_likelihood[it-1])) or (it >= N_iter and log_likelihood[it]>=log_likelihood[it-1]):
        return (True,log_likelihood[it])
    
#     time_logging[7]+=time()-time_ts
    if (it+1)>=len(log_likelihood):
        print "could not find a non-decreasing likelihood... might want to increase the number of iterations."
#         time_logging[8]+=time()-time_tss
        return (True,log_likelihood[it])
    
    return (False,log_likelihood[it])
def compute_likelihood_perread(X,phi,zeta,lambdap,bad_k,E_freqIso,N):
#     isoform_likelihood = np.transpose((b[:,X]),axes=[1,0,2])

    # person x reads x global iso
    read_assignments = np.einsum('jil,jlk->jik',phi,zeta)
    num_reads=0
    totallikely=0
    for j in range(zeta.shape[0]):
        for i in range(N[j]):
            num_reads+=1
            readlikely=0
            for k in range(lambdap.shape[0]):
                readlikely+=E_freqIso[j,k]*read_assignments[j,i,k]*(lambdap[k,X[j,i]]/np.sum(lambdap[k]))
            totallikely+=np.log(readlikely)
            
    return totallikely/num_reads

def compute_likelihoodnew(X,phi,zeta,lambdap,bad_k,E_freqIso,N):
#     isoform_likelihood = np.transpose((b[:,X]),axes=[1,0,2])

    # person x reads x global iso
    read_assignments = np.einsum('jil,jlk->jik',phi,zeta).reshape(zeta.shape[0]*phi.shape[1],zeta.shape[2]).T
    
    n_cnts = np.zeros(lambdap.shape,dtype=np.float32) # global iso x read-terms    
    flat_X=X.flatten()
    for i in range(len(flat_X)):
        n_cnts[:,flat_X[i]]+=read_assignments[:,i] # set all isos, for this read term
#     n_cnts[:,flat_X]+=read_assignments[:,:]
    reads_per_iso = np.sum(n_cnts,axis=1)
    
    read_assignments = np.einsum('jil,jlk->jik',phi,zeta)
    n_cnts = np.zeros((zeta.shape[0],lambdap.shape[0],lambdap.shape[1]),dtype=np.float32) # global iso x read-terms    
    for j in range(zeta.shape[0]):
        for i in range(N[j]):
            n_cnts[j,:,X[j,i]]+=read_assignments[j,i] # set all isos, for this read term
    #     n_cnts[:,flat_X]+=read_assignments[:,:]
    reads_per_iso = np.sum(n_cnts,axis=2)

    totallikely=0
    for j in range(zeta.shape[0]):
        likelyj=0
        for k in range(reads_per_iso.shape[1]):
            if reads_per_iso[j,k] > 0:
                likelyj+=(E_freqIso[j,k]*np.exp(np.float64(dirMult(n_cnts[j,k],reads_per_iso[j,k],lambdap[k]))))
        totallikely+=np.log(likelyj)
        
    return totallikely

def compute_likelihood_per_person(X,phi,zeta,lambdap,bad_k,E_freqIso,N,V,C):
#     isoform_likelihood = np.transpose((b[:,X]),axes=[1,0,2])

    # person x reads x global iso
    read_assignments = np.einsum('jvl,jlk->jvk',phi,zeta)
    
    
    n_cnts = np.zeros((zeta.shape[0],lambdap.shape[0],lambdap.shape[1]),dtype=np.float32) # global iso x read-terms    
    for j in range(zeta.shape[0]):
        for v in range(V):
            if C[j,v]>0:
                n_cnts[j,:,v]+=read_assignments[j,v,:]*C[j,v]
#     n_cnts[:,flat_X]+=read_assignments[:,:]
    reads_per_iso = np.sum(n_cnts,axis=2)

    likely=np.zeros(zeta.shape[0],dtype=np.float)
    for j in range(zeta.shape[0]):
        for k in range(lambdap.shape[0]):
            if k not in bad_k and reads_per_iso[j,k] > 0 and E_freqIso[j,k]>0:
                likely[j]+=np.log(E_freqIso[j,k])+dirMult(n_cnts[j,k],reads_per_iso[j,k],lambdap[k])
#                 likely[j]+=dirMult(n_cnts[j,k],reads_per_iso[j,k],lambdap[k])
    return likely

def compute_likelihood(X,phi,zeta,lambdap,bad_k,N):
#     isoform_likelihood = np.transpose((b[:,X]),axes=[1,0,2])

    # person x reads x global iso
    read_assignments = np.einsum('jil,jlk->jik',phi,zeta)
    
    n_cnts = np.zeros(lambdap.shape,dtype=np.float32) # global iso x read-terms    
    for j in range(zeta.shape[0]):
        for i in range(N[j]):
            n_cnts[:,X[j,i]]+=read_assignments[j,i,:]
#     n_cnts[:,flat_X]+=read_assignments[:,:]
    reads_per_iso = np.sum(n_cnts,axis=1)

    likely=0
    for k in range(len(reads_per_iso)):
        if k not in bad_k and reads_per_iso[k] > 0:
            likely+=dirMult(n_cnts[k],reads_per_iso[k],lambdap[k])
    return likely

def compute_likelihood_old(b,X,phi,zeta,N,lambdap,bad_k):
#     isoform_likelihood = np.transpose((b[:,X]),axes=[1,0,2])
    X_mask = np.zeros((len(b),X.shape[0],X.shape[1]),dtype=int)
    X_mask[:]=X
    b_selector=np.ma.masked_where(X_mask==-1, b[:,X])
    b_selector.filled(0)
    isoform_likelihood = np.transpose(b_selector.filled(0)*(np.ma.masked_where(X_mask==-1, lambdap[:,X]).filled(0)/((np.sum(lambdap[:],axis=1))[:, np.newaxis, np.newaxis])),axes=[1,0,2])
    isoform_likelihood[:,bad_k,:]=0
    
    try:
        logsum = np.log(np.sum(np.einsum('ijk,ikl->ilj',phi,zeta)*isoform_likelihood,axis=1))
    except:
        print 'exception in old likelihood'
    
    logsum[logsum==-np.inf] = 0
    return np.sum(np.sum(logsum))/np.float32(np.sum(N))
    
def dirMult(n_cnts,num_reads,lambdap):
    sum_alpha = np.sum(lambdap)
    coeff=gammaln(num_reads+1)+gammaln(sum_alpha)-gammaln(num_reads+sum_alpha)
    l=coeff+np.sum(gammaln(n_cnts+lambdap)-gammaln(n_cnts+1)-gammaln(lambdap))
    return l
    
def fastUpdateLambda(grp,M,V,phi,zeta,lambdap,b,eta,X,starts,ends,D,bad_k,C):
    lambdapCopy = np.zeros((len(grp),lambdap.shape[1]),dtype=lambdap.dtype)
    for v in range(V): 
        phisums=np.asarray([phi[j][v,:]*C[j,v] for j in range(M)]) # people by isoforms
        idx = 0
        for k in grp:
            if k not in bad_k:
                thislambda=0
                for j in range(M):
                    thislambda += np.sum(zeta[j][:,k]*phisums[j])
    #             lambdapCopy[k,v]=b[k,v]*(eta+thislambda)
                lambdapCopy[idx,v]=b[k,v]*(eta+D*thislambda)
            idx+=1
    return lambdapCopy
    
def updateLambda(grp,M,V,phi,zeta,lambdap,b,eta,X):
    starts=np.zeros(M,dtype=np.int32)
    ends=np.zeros(M,dtype=np.int32)
    lambdapCopy = np.zeros(lambdap.shape,dtype=lambdap.dtype)
    for k in grp:
        starts[:]=0
        ends[:]=0
        for v in range(V): 
            (starts,ends)=incrementXCntrs(starts,ends,v,X)
#             (starts,ends)=incrementXCntrs2(ends,v,X)
            phisums=np.asarray([np.sum(phi[i,starts[i]:ends[i],:],axis=0) for i in range(M)])
            thislambda = np.sum(np.sum(zeta[:,:,k]*phisums))

            lambdapCopy[k,v]=b[k,v]*(eta+thislambda)
            
    lambdap[grp,:]=lambdapCopy[grp,:]

def updateMuForIsoform(E,mu,grp,monte_carlo_mu,lambdap,term2exon,term2junction,eta,kappa,V,r,s,E_beta,nonjncterms,
                       exonToTermsJncs,jncterms,exonToTermsNonJnc,bad_k,term2exonsCovered,term2exonsCoveredLengths):
    global USE_CYTHON
    muCopy = np.zeros(mu.shape,dtype=mu.dtype)
    muCopy[:,:] = mu[:,:]
    cntr=0
    for k in grp:
        if k not in bad_k:
            for e in range(E):
                # if all other values in mu_k else than e are 0, the proba that mu_ke=1 is 1 
                #print "updating mu for splice form " + str(mu[k,:]) + " and exon " + str(e)              
                if np.sum(mu[k,:])==mu[k,e]:
                    muCopy[k,e]=0.95
                else:                                
                    if monte_carlo_mu:
                        term1,term0 = expectation_b(lambdap[k,:],muCopy[k,:],e,term2exon,term2junction,eta,V,r,s,E_beta[k])
                    else:
                        prob = fast_expectation_deterministic_b(lambdap[k,:],muCopy[k,:],e,term2exon,term2junction,eta,V,r,s,E_beta[k],nonjncterms,exonToTermsJncs,jncterms,exonToTermsNonJnc,1,term2exonsCovered,term2exonsCoveredLengths,USE_CYTHON)
                    muCopy[k,e] = prob
                    cntr+=1
                muCopy[k,:][muCopy[k,:]<MU_UNCERTAINTY]=MU_UNCERTAINTY
                muCopy[k,:][muCopy[k,:]>(1-MU_UNCERTAINTY)]=1-MU_UNCERTAINTY
    return muCopy[grp,:]
    
def updateIndividualPlate(grp,X,N,update_gamma,update_zeta,update_phi,update_a,update_lambda,
                          gamma1,gamma2,phi,zeta,a1,a2,omega,E_psi,E_beta, alpha, E_t, M, K, 
                          lambdap, V,mu,term2exon,term2junction, eta, stochastic, converge,C,Cbool):
    gamma1Copy=(np.zeros(gamma1.shape,dtype=gamma1.dtype))
    gamma2Copy=(np.zeros(gamma2.shape,dtype=gamma2.dtype))
    zetaCopy=(np.zeros(zeta.shape,dtype=zeta.dtype))
    lastZeta=(np.zeros(zeta.shape,dtype=zeta.dtype))
    phiCopy=(np.zeros(phi.shape,dtype=phi.dtype))
#     pp = pprint.PrettyPrinter(indent=4)
    ''' loop for local parameters to converge, now only 2 for reasons of speed '''
    it=0
    while(True):
        numTrue = 0
        for j in grp:
            T=zeta[j].shape[0]
#             words=X[j,:N[j]]
            if update_gamma:
                '''update gamma1,gamma2: var param for psi'''
                for l in range(T):
                    if gamma1[j,l]>0:
                        gamma1Copy[j][l] = 1+np.dot(phi[j][:,l],C[j])
                        gamma2Copy[j][l] = alpha + np.sum(np.dot(phi[j][:,(l+1):].T,C[j]))
                    # compute expectation(log phi)
            if update_zeta:
                '''update zeta: var param for c'''
                if it==0:
                    zetaCopy[j][:,:] = f_update_zeta(phi[j][:,:], E_beta, E_t, T, K, N[j],gamma1[j],C[j])
                else:
                    zetaCopy[j][:,:] = f_update_zeta(phiCopy[j][:,:], E_beta, E_t, T, K, N[j],gamma1[j],C[j])
                    
#                 assert np.allclose(np.sum(zetaCopy[j][:,:],axis=1),1)                                            
            if update_phi:
                '''update phi: var param for z'''
                phiCopy[j] =  f_update_phi(zetaCopy[j][:,:], E_psi[j][:], E_beta, T, N[j],np.where(gamma1[j]>0)[0],C[j],Cbool[j])
                assert np.allclose(np.sum(phiCopy[j][np.nonzero(Cbool[j])],1),1)
            if (it>0 and not stochastic) or np.all(np.abs(zetaCopy[j][:,:]-lastZeta[j][:,:])<INDIVIDUAL_PLATE_CONVERGE) or it>INDIVIDUAL_PLATE_MAXIT:
                numTrue+=1
        it+=1
        if numTrue==len(grp):
            break
        lastZeta=copy.deepcopy(zetaCopy)
    return ([gamma1Copy[i] for i in grp],[gamma2Copy[i]  for i in grp],[zetaCopy[i]  for i in grp],[phiCopy[i] for i in grp])

def incrementXCntrs(starts,ends,v,X):
    starts[:]=ends[:]
    
    for idx in range(len(X)):
        while ends[idx]<len(X[idx]) and X[idx,ends[idx]]==v:
            ends[idx]+=1
            
    return (starts,ends)

def incrementXCntrs2(ends,v,X):
    starts=ends[:]
    wheres=[np.where(X[i]==v) for i in range(len(X))]
    for idx in range(len(X)):
        if(wheres[idx][0].size):
            ends[idx]=wheres[idx][0][-1]
        else:
            ends[idx]=starts[idx]
    return (starts,ends)

def main(args):
    if args.seed != -1:
        np.random.seed(args.seed)
        random.seed(args.seed)
    
    t0 = time()
    print os.getcwd()
    result_directory=args.output_dir
    print "creating directory %s" % result_directory
    if not os.path.isdir(result_directory):
        os.makedirs(result_directory)
    # check that delay is between 0.5 and 1
    assert args.delay>0.5 and args.delay <=1
    
    if (args.data_file_text ):
        C = np.loadtxt(args.data_file_text, delimiter="\t", dtype=np.float32)
    if (args.data_file):
        C = np.load(args.data_file)
    if (args.firstM>0):
        C = C[0:args.firstM]
    if (len(C.shape)==1):
        C = np.reshape(C,(1,C.shape[0])) # edge case when there is one individual
    if (C.shape==()):
        temp_val=C
        C = np.zeros((1,1),dtype=np.float32)
        C[0,0]=temp_val

    exon_lengths = args.exon_lengths
    read_length = 0
    if args.exon_lengths is not None:
        read_length = int(args.exon_lengths.split(",")[0])
        exon_lengths = args.exon_lengths.split(",")[1:]
        exon_lengths = [int(i)-read_length+1 for i in exon_lengths]
        
    print "Count data loaded, matrix size %dx%d" %(C.shape[0],C.shape[1])
    H = HDP(C,args.term2exon_file,omega=args.omega,eta=args.eta,alpha=args.alpha,r=args.r,s=args.s,K=args.maxiso,initialize_from=args.initialize_from,
            exon_lengths=exon_lengths,read_length=read_length,compress_input=args.compress,monte_carlo_mu=args.mc_mu,log=args.log,existence_threshold=args.existence_threshold,threads=args.threads,maxiso=args.maxiso,splicegraph=args.splicegraph,kappa=args.kappa,exon_info_file=args.exon_info_file)
    H.update(args.output_dir,args.N_iter,args.forget_rate, args.delay,args.update,log=args.log,num_iterations_to_save=args.num_it_save,threads=args.threads,
             converge=args.converge,stochastic=args.stochastic,batchsize=args.batchsize,tmpfolder=args.tmpfolder,existence_threshold=args.existence_threshold,
             term2exonsCovered=H.var_dist.term2exonsCovered,term2exonsCoveredLengths=H.var_dist.term2exonsCoveredLengths,C=C)
    t1 = time()
    if args.log:
        print >> sys.stderr, "total time taken " + str(t1-t0)
    return 0
    
def read_args_from_cmdline():
    parser = argparse.ArgumentParser(description='VB inference for HDP model')
    parser.add_argument('-e','--term2exon-file',type=str,default='term2exons.txt',required=True,help='A txt file that stores the exon (or exon junction) to which each term of the dictionary maps.')
    parser.add_argument('-f','--data-file',type=str,required=False,help='A numpy file that stores the data matrix (aggregated counts)')
    parser.add_argument('-d','--data-file-text',type=str,required=False,help='Flat text file that stores the data matrix (aggregated counts)')
    parser.add_argument('-x','--exon-lengths',type=str,required=False,default=None,help='String denoting exon lengths denoted by commas preceeded by a read length, e.g. 100,90,180,30')
    parser.add_argument('-i','--initialize-from',type=str,default=None,required=False,help='A directory where npy files can be used to initialize the variational parameters.')
    parser.add_argument('-o','--output-dir',required=True)
    parser.add_argument('-n','--exon-info-file',type=str,default=None,required=False,help='Contains exon start positions and lengths, 1 pair per line seperated by a colon (e.g. 1:10).')
    
    # hyperparameters
    parser.add_argument('--omega',type=float,default=1.0, help='concentration parameter of global DP prior')
    parser.add_argument('--alpha',type=float,default=1.0, help='concentration parameter of DP prior on document-specific topic distributions')
    parser.add_argument('--r',type=float,default=1, help='first parameter of beta prior on exon selector vector')
    parser.add_argument('--s',type=float,default=1, help='second parameter of beta prior on exon selector vector')
    parser.add_argument('--eta',type=float,default=0.9, help='scale parameter of smoothing Dirichlet prior on beta (or shape parameter of the gamma)')
    parser.add_argument('--kappa',type=float,default=1.0, help='scale parameter for Gamma distribution for tilde{\beta}')
    
    # technical parameters
#    parser.add_argument('--K',type=int,default=7,help='Maximum number of isoforms globally allowed (necessary b/c array memory allocation is static)')
    #parser.add_argument('--T',type=int,default=3,help='Maximum number of isoforms locally allowed (per individual)')
    
    # step size scheme parameters
    parser.add_argument('--N-iter',type=int,default=500,help='Maximum number of iterations of algorithm')
    parser.add_argument('--forget-rate',type=float,default=0.7, help='forgetting rate') #has to be between 0.5 and 1
    parser.add_argument('--delay',type=float,default=1, help='delay')

    # program behavior
    parser.add_argument('--compress',type=int,default=0,help='0 [default] if we do not compress input, 1 to compress input')
    parser.add_argument('--mc-mu',type=int,default=0,help='0 [default] if we do not want a monte carlo estimate for mu, 1 to make a monte carlo estimate')
    parser.add_argument('--firstM',type=int,default=-1,help='Subset of the number of individuals')
    parser.add_argument('--log',type=int,default=0,help='0 [default] logging disabled or 1 logging enabled')
    parser.add_argument('--existence-threshold',type=float,default=0.5,help='minimum probability to call an exon as "existing" in an isoform [default 0.5]')
    parser.add_argument('--num-it-save',type=int,default=2,help='number of equally spaced iterations to save [default 2] (which saves the first and last iteration only)')
    parser.add_argument('--threads',type=int,default=1,help='number of parallel threads to use for execution')
    parser.add_argument('--maxiso',type=int,default=600,help='maximum number of isoforms to initialize \beta')
    parser.add_argument('--stochastic',type=int,default=1,help='0 if variational inference, 1 [default] if stochastic variational inference ')
    parser.add_argument('--splicegraph',type=int,default=0,help='[default] 0 if init all splice forms (fallback: totally random); 1 if using splice graph for initial isoforms (fallback: splicegraph=0)')
    parser.add_argument('--batchsize',type=int,default=10,help='10 [default] number of individuals per batch')
    parser.add_argument('--seed',type=long,default=-1,help='[default - random seed] random number generator seed')
    parser.add_argument('--converge',type=float,default=1e-3, help='convergence threshold for E[log(t)] and E[log(beta)] ')
    parser.add_argument('--tmpfolder',type=str,default=None,required=False,help='Temporary folder used to store arrays for parallel processing. By default, it uses /dev/shm or the OS temporary folder.')

    parser.add_argument('--iter-prop',type=int,default=20,help='number of iterations between proposals')
    parser.add_argument('--min-n-prop',type=int,default=2,help='minimum number of proposals for new isoforms')
    parser.add_argument('--max-n-prop',type=int,default=2,help='maximum number of proposals for new isoforms')
    parser.add_argument('--min-n-iter',type=int,default=100,help='minimum number of iterations')
    parser.add_argument('--new-iso-prop',type=int,default=2,help='number of new isoforms to propose during each proposal iteration')
    parser.add_argument('--red-iso-prop',type=int,default=1,help='number of isoforms coming from random read-terms')
    parser.add_argument('--use-cython',type=int,default=0,help='use the fast CYTHON implementation')
    parser.add_argument('--burn-in',type=int,default=100,help='number of iterations before starting to merge isoforms when proposing')
    parser.add_argument('--fast-solver',type=int,default=0,help='fast proposal by reducing the frequency of calling solver')
    parser.add_argument('--use-cython-update',type=int,default=0,help='use the fast CYTHON implementation for updateIndividualPlate')

    # lazy updates
    parser.add_argument('-w','--update',type=str,default=None,required=False,help='A binary denoting which variables to update (zeta, phi, gamma, lambda, a, rho, mu). Default:all')

    args = parser.parse_args();
    # make sure we have an input file 
    if not (args.data_file_text or args.data_file):
        argparse.ArgumentParser(description='VB inference for HDP model').error('Need an input read mapping file, add -process or -upload')
    if (args.data_file_text and args.data_file):
        argparse.ArgumentParser(description='VB inference for HDP model').error('Cannot set both read dictionary text and binary versions')

    # set program constants
    global ROUNDS_IN_BETWEEN_PROPOSAL
    global MIN_NUMBER_ITS
    global MIN_NUMBER_PROPOSALS
    global MAX_NUMBER_PROPOSALS
    global NUM_NEW_ISO
    global REDUNDANT_ISOS
    global USE_CYTHON
    global MERGE_BURN_IN
    global FAST_SOLVER
    global USE_CYTHON_UPDATE
    ROUNDS_IN_BETWEEN_PROPOSAL = args.iter_prop
    MIN_NUMBER_PROPOSALS = args.min_n_prop
    MAX_NUMBER_PROPOSALS = args.max_n_prop
    MIN_NUMBER_ITS = np.max((args.min_n_iter,MIN_NUMBER_PROPOSALS*ROUNDS_IN_BETWEEN_PROPOSAL))
    NUM_NEW_ISO = args.new_iso_prop
    REDUNDANT_ISOS = args.red_iso_prop
    USE_CYTHON = args.use_cython
    MERGE_BURN_IN = args.burn_in
    FAST_SOLVER = args.fast_solver
    USE_CYTHON_UPDATE = args.use_cython_update
    
    if args.N_iter < MIN_NUMBER_ITS:
        print "setting the number of iterations to the minimum number of iterations"
        args.N_iter = MIN_NUMBER_ITS
        
    return args

if __name__ == '__main__':
    return_code = main(read_args_from_cmdline())
    sys.exit(return_code)
