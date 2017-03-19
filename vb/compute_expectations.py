'''
Computes relevant expectations for the variational bayes algorithm
'''
import numpy as np
import sys
from scipy.special import psi
from scipy.special import gammaln
import math
from numpy import dtype
import pyximport; pyximport.install()

try:
    import fast_update as fu
except ImportError:
    print >> sys.stderr, 'could not find fast_update, cannot use fast cython implementation'

MAX_EXPECTATION_TERMS=100000
EXPECTATION_THRESHOLD=0.95
EXPECTATION_PROB_THRESHOLD=0.001
ROUND_SIG_FIGS=1
EXISTANCE_THRESHOLD=0.5
#BPRIME_UNCERTAINTY=0.001
BPRIME_UNCERTAINTY=0.01
LAMBDAP_UNCERTAINTY=1e-3
MU_UNCERTAINTY = 0.01
#LAMBDAP_UNCERTAINTY=0.001

def construct_b_from_bprime(V,iso,term2exon,term2junction,discrete_mapping=0):
    ''' this function constructs the extended term selector vectors from the exon
     selector vectors in each column of matrix iso'''
    nbIso=iso.shape[0]
        
    b=np.zeros((nbIso,V),dtype=np.float32)
    b+=BPRIME_UNCERTAINTY
    
    jncs=term2junction.keys()
    nonjncs=[]
    for i in range(len(term2exon)):
        if i not in jncs:
            nonjncs.append(i)
    
    # for all terms that maps within a single exon, assign the value form the exon selector vector corresponding to that exon
    for i in range(nbIso): # for all isoforms
        for nonjnc in nonjncs: # for all exons in isoform i
            if iso[i,term2exon[nonjnc]]>EXISTANCE_THRESHOLD: # if the exon ex exists in isoform i 
                if(discrete_mapping):
                    b[i,nonjnc]=1-BPRIME_UNCERTAINTY
                else:
                    b[i,nonjnc]=iso[i,term2exon[nonjnc]]
                
    # now take care of junctions
    # for now assume not read paired... when read pair we need to be smart how we assign compatible reads
    for i in range(nbIso): # for all isoforms
        for jnc in jncs: # for all exons in isoform i
            #if np.all(iso[i,list(combinations(term2junction[jnc],2))]>EXISTANCE_THRESHOLD).all(): # if the exon ex exists in isoform i 
            num_inconsistent=0
            for jncctr in range(len(term2junction[jnc])-1):
                # are they both in isoform and adjacent in isoform?
                if iso[i,term2junction[jnc][jncctr]]<EXISTANCE_THRESHOLD or iso[i,term2junction[jnc][jncctr+1]]<EXISTANCE_THRESHOLD or np.any(iso[i,term2junction[jnc][jncctr]+1:term2junction[jnc][jncctr+1]]>EXISTANCE_THRESHOLD):
                    num_inconsistent+=1
            if(discrete_mapping):
                if num_inconsistent>0:
                    b[i,jnc]=BPRIME_UNCERTAINTY
                else:
                    b[i,jnc]=1-BPRIME_UNCERTAINTY
            else:
                if iso[i,term2junction[jnc][len(term2junction[jnc])-1]]<EXISTANCE_THRESHOLD:
                    num_inconsistent+=1
                b[i,jnc]=(1-BPRIME_UNCERTAINTY)-(1-BPRIME_UNCERTAINTY)*np.float(num_inconsistent)/np.float(len(term2junction[jnc]))

    return b

def construct_b_from_bprime_pairedend(V,iso,term2exon,term2junction,discrete_mapping=0,exon_pos_and_lengths=None):
    ''' this function constructs the extended term selector vectors from the exon
     selector vectors in each column of matrix iso'''
    nbIso=iso.shape[0]
        
    b=np.zeros((nbIso,V),dtype=np.float32)
    b+=BPRIME_UNCERTAINTY
    
    jncs=term2junction.keys()
    nonjncs=[]
    for i in range(len(term2exon)):
        if i not in jncs:
            nonjncs.append(i)
    
    # for all terms that maps within a single exon, assign the value form the exon selector vector corresponding to that exon
    for i in range(nbIso): # for all isoforms
        for nonjnc in nonjncs: # for all exons in isoform i
            if iso[i,term2exon[nonjnc]]>EXISTANCE_THRESHOLD: # if the exon ex exists in isoform i 
                if(discrete_mapping):
                    b[i,nonjnc]=1-BPRIME_UNCERTAINTY
                else:
                    b[i,nonjnc]=iso[i,term2exon[nonjnc]]
                
    # now take care of junctions
    # for now assume not read paired... when read pair we need to be smart how we assign compatible reads
    for i in range(nbIso): # for all isoforms
        for jnc in jncs: # for all exons in isoform i
            #if np.all(iso[i,list(combinations(term2junction[jnc],2))]>EXISTANCE_THRESHOLD).all(): # if the exon ex exists in isoform i 
            num_inconsistent=0
            for jncctr in range(len(term2junction[jnc])-1):
                # are they both in isoform and adjacent in isoform?
                if iso[i,term2junction[jnc][jncctr]]<EXISTANCE_THRESHOLD or iso[i,term2junction[jnc][jncctr+1]]<EXISTANCE_THRESHOLD or np.any(iso[i,term2junction[jnc][jncctr]+1:term2junction[jnc][jncctr+1]]>EXISTANCE_THRESHOLD):
                    num_inconsistent+=1
            if(discrete_mapping):
                if num_inconsistent>0:
                    b[i,jnc]=BPRIME_UNCERTAINTY
                else:
                    b[i,jnc]=1-BPRIME_UNCERTAINTY
            else:
                if iso[i,term2junction[jnc][len(term2junction[jnc])-1]]<EXISTANCE_THRESHOLD:
                    num_inconsistent+=1
                b[i,jnc]=(1-BPRIME_UNCERTAINTY)-(1-BPRIME_UNCERTAINTY)*np.float(num_inconsistent)/np.float(len(term2junction[jnc]))

    return b

def construct_b_from_bprime_array(V,iso,term2exonsCovered,term2exonsCoveredLengths,discrete_mapping):
    ''' this function constructs the extended term selector vectors from the exon
     selector vectors in each column of matrix iso'''
    nbIso=iso.shape[0]
        
    b=np.zeros((nbIso,V),dtype=np.float32)
    b+=BPRIME_UNCERTAINTY
    
    # for all terms that maps within a single exon, assign the value form the exon selector vector corresponding to that exon
    for i in range(nbIso): # for all isoforms
        for v in range(V):
            if term2exonsCoveredLengths[v]==1:
                if iso[i,term2exonsCovered[v,0]]>EXISTANCE_THRESHOLD: # if the exon ex exists in isoform i 
                    if(discrete_mapping):
                        b[i,v]=1-BPRIME_UNCERTAINTY
                    else:
                        b[i,v]=iso[i,term2exonsCovered[v,0]]
            else:
                # for now assume not read paired... when read pair we need to be smart how we assign compatible reads
                #if np.all(iso[i,list(combinations(term2junction[jnc],2))]>EXISTANCE_THRESHOLD).all(): # if the exon ex exists in isoform i 
                num_inconsistent=0
                
                for jncctr in range(term2exonsCoveredLengths[v]-1):
                    # are they both in isoform and adjacent in isoform?
                    if iso[i,term2exonsCovered[v,jncctr]]<EXISTANCE_THRESHOLD or iso[i,term2exonsCovered[v,jncctr+1]]<EXISTANCE_THRESHOLD or np.any(iso[i,term2exonsCovered[v,jncctr]+1:term2exonsCovered[v,jncctr+1]]>EXISTANCE_THRESHOLD):
                        num_inconsistent+=1
                if(discrete_mapping):
                    if num_inconsistent>0:
                        b[i,v]=BPRIME_UNCERTAINTY
                    else:
                        b[i,v]=1-BPRIME_UNCERTAINTY
                else:
                    if iso[i,term2exonsCovered[v][term2exonsCoveredLengths[v]-1]]<EXISTANCE_THRESHOLD:
                        num_inconsistent+=1
                    b[i,v]=(1-BPRIME_UNCERTAINTY)-(1-BPRIME_UNCERTAINTY)*np.float(num_inconsistent)/np.float(term2exonsCoveredLengths[v])
    return b


def softmax(w):
    w = np.array(w)
    maxes = np.amax(w)
    #maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e)
    if not( np.sum(dist)==1 ):
        # this is for machine precision issues
        dist[-1]=1-np.sum(dist[:-1])
    assert np.sum(dist)==1
    return dist

def expectation_sigma_beta(a1,a2):
    assert len(a1)==len(a2)
    lenLeft = 1.0

    K=a1.shape[0]
    E_t = np.zeros(a1.shape[0],dtype=np.float64)
    for k in range(K):
        if a1[k]>0:
            E_t[k]=(a1[k]/(a1[k]+a2[k]))*lenLeft
            lenLeft=lenLeft-E_t[k]
    return E_t


def expectation_log_sigma_beta(a1,a2):
    assert len(a1)==len(a2)
    
    K=len(a1)
    E_t = np.zeros(K,dtype=np.float64)
    psi_a1 = np.zeros(K,dtype=np.float64)
    psi_a2 = np.zeros(K,dtype=np.float64)
    psi_a1a2 = np.zeros(K,dtype=np.float64)
    # first term
    psi_a1[0]= psi(a1[0])
    psi_a2[0]= psi(a2[0])
    psi_a1a2[0]= psi(a1[0]+a2[0])
    E_t[0] = psi_a1[0] - psi_a1a2[0]
    for k in range(1,K):
        psi_a1[k]= psi(a1[k])
        psi_a2[k]= psi(a2[k])
        psi_a1a2[k]= psi(a1[k]+a2[k])
        E_t[k] = psi_a1[k] - psi_a1a2[k] + np.sum(psi_a2[:k]) - np.sum(psi_a1a2[:k])
    
    return E_t

def expectation_log_dirichlet(lambdap):
    K,V = lambdap.shape
    E_beta=np.zeros((K,V),dtype=np.float32)
    for k in range(K):
        for v in range(V):
            if not(lambdap[k,v]==0):
                E_beta[k,v]=psi(lambdap[k,v])-psi(np.sum(lambdap[k,:]))
    return E_beta

def expectation_log_beta(rho1,rho2):
    assert len(rho1)==len(rho2)
    K=len(rho1)
    E_pi = np.zeros(K,dtype=np.float64)
    # first term
    for k in range(K):
        E_pi[k] = psi(rho1[k]) - psi(rho1[k]+rho2[k]) 
    
    return E_pi

def expectation_b(lambdap,mu,e,term2exon,term2junction,eta,V,r,s,E_beta):
    # compute the dirichlet part of the \mu update
    # e is the exon index for which we are doing the update of mu
    E=max(max(term2exon),max([item for sublist in term2junction.values() for item in sublist]))+1
    cnt=0
    #cnt0=0
    term_1_all=0
    term_0_all=0
    term_1_0_all=0
    term_1_1_all=0
    term_1_2_all=0
    term_0_0_all=0
    term_0_1_all=0
    term_0_2_all=0
    while cnt<1000: # you can set this number as you wish, the bigger, the more accurate the MC estimate
        
        muOn=np.zeros(mu.shape,mu.dtype)
        muOn[:]=mu[:]
        muOn[e]=(1-MU_UNCERTAINTY)
        muOff=np.zeros(mu.shape,mu.dtype)
        muOff[:]=mu[:]
        muOff[e]=MU_UNCERTAINTY
        '''simulate a b' with b'_e fixed'''
        # compute for b'_e=1 -- exon e is included in mu
        random_vec=np.random.rand(E)
        bprimetemp2_1=np.asarray(random_vec<muOn, dtype=np.float64)

        # build the corresponding term selector vector
        btemp1=construct_b_from_bprime(V,bprimetemp2_1.reshape((1,E)),term2exon,term2junction,1)
        btemp1[btemp1<BPRIME_UNCERTAINTY]=BPRIME_UNCERTAINTY # all for some error 
        
        # same for b'_e=0 -- exon e is not included in mu
        bprimetemp2_0=np.asarray(random_vec<muOff, dtype=np.float64)
        btemp0=construct_b_from_bprime(V,bprimetemp2_0.reshape((1,E)),term2exon,term2junction,1)
        btemp0[btemp0<BPRIME_UNCERTAINTY]=BPRIME_UNCERTAINTY # all for some error 
        
        assert btemp1.shape[0]==1
        
        ''' simulate beta from dirichlet(lambda) to produce a multinomial probability vector'''
        #x=1
        term_1_0=gammaln(np.sum(eta*btemp1))
        term_1_1=np.sum(gammaln(eta*btemp1))
        term_1_2=np.dot((eta*btemp1-1),E_beta)
        term_1_6=np.sum(np.log(muOn[e])*(r-1+1)+np.log(1-muOn[e])*(s-1))
        term_1_all+=term_1_0-term_1_1+term_1_2+term_1_6
        
        
        #x=0
        term_0_0=gammaln(np.sum(eta*btemp0))
        term_0_1=np.sum(gammaln(eta*btemp0))
        term_0_2=np.dot((eta*btemp0-1),E_beta)
        term_0_6=np.sum(np.log(muOff[e])*(r-1)+np.log(1-muOff[e])*(s-1+1))
        term_0_all+=term_0_0-term_0_1+term_0_2+term_0_6
        
        term_1_0_all+=term_1_0
        term_1_1_all+=term_1_1
        term_1_2_all+=term_1_2
        term_0_0_all+=term_0_0
        term_0_1_all+=term_0_1
        term_0_2_all+=term_0_2


        cnt+= 1
        
    E_gammaterm1 = term_1_all/cnt
    E_gammaterm0 = term_0_all/cnt
    return E_gammaterm1,E_gammaterm0

def expectation_gamma_b(lambdap,mu,e,term2exon,term2junction,eta,V,r,s,E_beta,kappa):
    # compute the dirichlet part of the \mu update
    # e is the exon index for which we are doing the update of mu
    E=max(max(term2exon),max([item for sublist in term2junction.values() for item in sublist]))+1
    cnt=0
    
    wrapper = np.zeros((1,len(mu)),dtype=np.float32)
    wrapper[0]=mu
    b = construct_b_from_bprime(V,wrapper,term2exon,term2junction,1)

    term_1_all=0
    term_0_all=0
    gammaTerm0_on = 0
    gammaTerm0_off = 0
    gammaTerm1_on = 0
    gammaTerm1_off = 0
    gammaTerm2_on = 0
    gammaTerm2_off = 0
    gammaTerm3_on = 0
    gammaTerm3_off = 0
    while cnt<1000: # you can set this number as you wish, the bigger, the more accurate the MC estimate
        
        muOn=np.zeros(mu.shape,mu.dtype)
        muOn[:]=mu[:]
        muOn[e]=(1-MU_UNCERTAINTY)
        muOff=np.zeros(mu.shape,mu.dtype)
        muOff[:]=mu[:]
        muOff[e]=MU_UNCERTAINTY
        '''simulate a b' with b'_e fixed'''
        # compute for b'_e=1 -- exon e is included in mu
        random_vec=np.random.rand(E)
        bprimetemp2_1=np.asarray(random_vec<muOn, dtype=np.float64)

        # build the corresponding term selector vector
        btemp1=construct_b_from_bprime(V,bprimetemp2_1.reshape((1,E)),term2exon,term2junction,1)
        btemp1[btemp1<BPRIME_UNCERTAINTY]=BPRIME_UNCERTAINTY # all for some error 
        
        # same for b'_e=0 -- exon e is not included in mu
        bprimetemp2_0=np.asarray(random_vec<muOff, dtype=np.float64)
        btemp0=construct_b_from_bprime(V,bprimetemp2_0.reshape((1,E)),term2exon,term2junction,1)
        btemp0[btemp0<BPRIME_UNCERTAINTY]=BPRIME_UNCERTAINTY # all for some error 
        
        assert btemp1.shape[0]==1
        
        lambdap_on = np.copy(lambdap)
        lambdap_off = np.copy(lambdap)
        
        for v in range(V):
            if b[0][v]>EXISTANCE_THRESHOLD and btemp1[0][v]<EXISTANCE_THRESHOLD:
                lambdap_on[v] = LAMBDAP_UNCERTAINTY
                
            if b[0][v]>EXISTANCE_THRESHOLD and btemp0[0][v]<EXISTANCE_THRESHOLD:
                lambdap_off[v] = LAMBDAP_UNCERTAINTY
        
        ''' simulate beta from dirichlet(lambda) to produce a multinomial probability vector'''
        #x=1
        term_1_0=np.sum(gammaln(btemp1*eta))
        term_1_1=np.sum(btemp1*eta*np.log(kappa))
        term_1_2=np.sum((btemp1*eta-1)*psi(lambdap_on)+np.log(kappa))
        term_1_3=np.sum(lambdap_on)
        term_1_6=np.sum(np.log(muOn[e])*(r-1+1)+np.log(1-muOn[e])*(s-1))
        term_1_all+=term_1_0-term_1_1+term_1_2+term_1_3+term_1_6
        
        
        #x=0
        term_0_0=np.sum(gammaln(btemp0*eta))
        term_0_1=np.sum(btemp0*eta*np.log(kappa))
        term_0_2=np.sum((btemp0*eta-1)*psi(lambdap_off)+np.log(kappa))
        term_0_3=np.sum(lambdap_off)
        term_0_6=np.sum(np.log(muOff[e])*(r-1)+np.log(1-muOff[e])*(s-1+1))
        term_0_all+=term_0_0-term_0_1+term_0_2+term_0_3+term_0_6


        cnt+= 1

    E_gammaterm1 = term_1_all/cnt
    E_gammaterm0 = term_0_all/cnt
    return E_gammaterm1,E_gammaterm0

def fast_expectation_deterministic_b(lambdap,mu,e,term2exon,term2junction,eta,V,r,s,E_beta,nonjncterms,exonToTermsJncs,jncterms,exonToTermsNonJnc, discrete_mapping,term2exonsCovered,term2exonsCoveredLengths,USE_CYTHON):
    # compute the dirichlet part of the \mu update
    #lambdap[lambdap<ERROR_P]=ERROR_P # all for some error 

    nonjnctermsMapToE=nonjncterms[np.where(term2exon[nonjncterms]==e)]
    nonjnctermsNoMapToE=nonjncterms[np.where(term2exon[nonjncterms]!=e)]
    if e in exonToTermsJncs:
        jnctermsForE=exonToTermsJncs[e].tolist()
    else:
        jnctermsForE=[]
    jnctermsNoForE=jncterms[np.in1d(jncterms,jnctermsForE)==False]

    muOn=np.zeros((1,mu.shape[0]),mu.dtype)
    muOn[0,:]=mu[:]
    muOn[0,e]=(1-MU_UNCERTAINTY)
    if USE_CYTHON:
        b_on = fu.construct_b_from_bprime(V,muOn,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
    else:
        b_on = construct_b_from_bprime(V,muOn,term2exon,term2junction,1)

    muOff=np.zeros(mu.shape,mu.dtype)
    muOff[:]=mu[:]
    muOff[e]=MU_UNCERTAINTY
    wrapper = np.zeros((1,len(muOff)),dtype=np.float32)
    wrapper[0]=muOff
#     
    if USE_CYTHON:
        b_off = fu.construct_b_from_bprime(V,wrapper,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD)
    else:
        b_off = construct_b_from_bprime(V,wrapper,term2exon,term2junction,1)

    dirichletTerm0_on = 0
    dirichletTerm1_on = 0
    dirichletTerm2_on = 0
    dirichletTerm0_off = 0
    dirichletTerm1_off = 0
    dirichletTerm2_off = 0
    dirichletTerm0_shared = 0
    
    for term in nonjnctermsMapToE:
        if b_on[0][term]>BPRIME_UNCERTAINTY:
            dirichletTerm0_on += len(nonjnctermsMapToE)*(eta)
            dirichletTerm1_on += len(nonjnctermsMapToE)*gammaln(eta)
            dirichletTerm2_on += np.sum((eta-1)*E_beta[nonjnctermsMapToE])

        if b_off[0][term]>BPRIME_UNCERTAINTY:
            dirichletTerm0_off += len(nonjnctermsMapToE)*(eta)
            dirichletTerm1_off += len(nonjnctermsMapToE)*gammaln(eta)
            dirichletTerm2_off += np.sum((eta-1)*E_beta[nonjnctermsMapToE])
        
    for term in nonjnctermsNoMapToE:
        dirichletTerm0_shared += (eta)
        
    jnxNoTuples = [term2junction[jnctermsNoForE[j]] for j in range(len(jnctermsNoForE))]
    jnxTuples = [term2junction[jnctermsForE[j]] for j in range(len(jnctermsForE))]
    cnt=0
    for EJncTuple in jnxNoTuples:
        term = jnctermsNoForE[cnt]
#         
        if USE_CYTHON:
            onInconsistencies=fu.jnxTupleMuInconsistent(term2exonsCovered[term],term2exonsCoveredLengths[term],muOn[0],EXISTANCE_THRESHOLD)
        else:
            onInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOn[0])
#         
        if USE_CYTHON:
            offInconsistencies=fu.jnxTupleMuInconsistent(term2exonsCovered[term],term2exonsCoveredLengths[term],muOff,EXISTANCE_THRESHOLD)
        else:
            offInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOff)
        if offInconsistencies==0 and onInconsistencies==0:
            dirichletTerm0_shared+=eta
        elif onInconsistencies==offInconsistencies:
            dirichletTerm0_shared+=eta

        if (offInconsistencies==0 and onInconsistencies>0) or (offInconsistencies>0 and onInconsistencies==0):
            jnxTuples.append(EJncTuple)
            jnctermsForE.append(term)
            
        cnt+=1
    
    
    cnt=0
    for EJncTuple in jnxTuples:
#         
        if USE_CYTHON:
            onInconsistencies=fu.jnxTupleMuInconsistent(term2exonsCovered[jnctermsForE[cnt]],term2exonsCoveredLengths[jnctermsForE[cnt]],muOn[0],EXISTANCE_THRESHOLD)
        else:
            onInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOn[0])

        if USE_CYTHON:
            offInconsistencies=fu.jnxTupleMuInconsistent(term2exonsCovered[jnctermsForE[cnt]],term2exonsCoveredLengths[jnctermsForE[cnt]],muOff,EXISTANCE_THRESHOLD)
        else:
            offInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOff)

        if b_on[0][jnctermsForE[cnt]]>BPRIME_UNCERTAINTY:
            dirichletTerm0_on += eta
            dirichletTerm1_on += gammaln(eta)
            dirichletTerm2_on += (eta-1)*E_beta[jnctermsForE[cnt]]

        if b_off[0][jnctermsForE[cnt]]>BPRIME_UNCERTAINTY:
            dirichletTerm0_off += eta
            dirichletTerm1_off += gammaln(eta)
            dirichletTerm2_off += (eta-1)*E_beta[jnctermsForE[cnt]]
        cnt+=1

    dirichletTerm0_on = gammaln(dirichletTerm0_on+dirichletTerm0_shared)
    dirichletTerm0_off = gammaln(dirichletTerm0_off+dirichletTerm0_shared)
    term_1_all=dirichletTerm0_on-dirichletTerm1_on+dirichletTerm2_on+(r-1+1)*np.log(muOn[0,e])+(s-1+0)*np.log(1-muOn[0,e])
    term_0_all=dirichletTerm0_off-dirichletTerm1_off+dirichletTerm2_off+(r-1+0)*np.log(muOff[e])+(s-1+1)*np.log(1-muOff[e])

    x=np.exp(term_1_all-np.max([term_1_all,term_0_all])) # to avoid underflow
    y=np.exp(term_0_all-np.max([term_1_all,term_0_all]))
    return x/(x+y)

def fast_expectation_deterministic_gamma_b(lambdap,mu,e,term2exon,term2junction,eta,V,r,s,E_beta,nonjncterms,exonToTermsJncs,jncterms,exonToTermsNonJnc, kappa, discrete_mapping=1):
    # compute the dirichlet part of the \mu update
    #lambdap[lambdap<ERROR_P]=ERROR_P # all for some error 
    
    nonjnctermsMapToE=nonjncterms[np.where(term2exon[nonjncterms]==e)]
    nonjnctermsNoMapToE=nonjncterms[np.where(term2exon[nonjncterms]!=e)]
    jnctermsForE=exonToTermsJncs[e].tolist()
    jnctermsNoForE=jncterms[np.in1d(jncterms,jnctermsForE)==False]

    wrapper = np.zeros((1,len(mu)),dtype=np.float32)
    wrapper[0]=mu
    b = construct_b_from_bprime(V,wrapper,term2exon,term2junction,1)
    
    muOn=np.zeros(mu.shape,mu.dtype)
    muOn[:]=mu[:]
    muOn[e]=(1-MU_UNCERTAINTY)
    wrapper = np.zeros((1,len(muOn)),dtype=np.float32)
    wrapper[0]=muOn
    b_on = construct_b_from_bprime(V,wrapper,term2exon,term2junction,1)
    lambdap_on = np.copy(lambdap)
    
    
    muOff=np.zeros(mu.shape,mu.dtype)
    muOff[:]=mu[:]
    muOff[e]=MU_UNCERTAINTY
    wrapper = np.zeros((1,len(muOff)),dtype=np.float32)
    wrapper[0]=muOff
    b_off = construct_b_from_bprime(V,wrapper,term2exon,term2junction,1)
    lambdap_off = np.copy(lambdap)
    
    for v in range(V):
        if b[0][v]>EXISTANCE_THRESHOLD and b_on[0][v]<EXISTANCE_THRESHOLD:
            lambdap_on[v] = LAMBDAP_UNCERTAINTY
            
        if b[0][v]>EXISTANCE_THRESHOLD and b_off[0][v]<EXISTANCE_THRESHOLD:
            lambdap_off[v] = LAMBDAP_UNCERTAINTY
            
    wrapper = np.zeros((1,len(lambdap_on)),dtype=np.float32)
    wrapper[0]=lambdap_on
    E_beta_on = expectation_log_dirichlet(wrapper)
    E_beta_on = E_beta_on[0]
    E_beta_nonlog_on = lambdap_on/np.sum(lambdap_on)
    wrapper = np.zeros((1,len(lambdap_off)),dtype=np.float32)
    wrapper[0]=lambdap_off
    E_beta_off = expectation_log_dirichlet(wrapper)
    E_beta_off = E_beta_off[0]
    E_beta_nonlog_off = lambdap_off/np.sum(lambdap_off)
        
    # only count those that map b_on[0,nonjnctermsMapToE] b_off[0,nonjnctermsMapToE]
    on_ok = np.where(b_on[0,nonjnctermsMapToE]>BPRIME_UNCERTAINTY)[0]
    off_ok = np.where(b_off[0,nonjnctermsMapToE]>BPRIME_UNCERTAINTY)[0]

    gammaTerm0_on = 0
    gammaTerm0_off = 0
    gammaTerm1_on = 0
    gammaTerm1_off = 0
    gammaTerm2_on = 0
    gammaTerm2_off = 0
    gammaTerm3_on = 0
    gammaTerm3_off = 0

    # actually follows a beta' distribution with parameters b*eta, sum(b*eta)
    if len(nonjnctermsMapToE[on_ok])>0:
        gammaTerm0_on -= np.sum(gammaln(b_on[0,nonjnctermsMapToE[on_ok]]*eta))
        gammaTerm1_on -= np.sum(b_on[0,nonjnctermsMapToE[on_ok]]*eta*np.log(kappa))
        gammaTerm2_on += np.sum((b_on[0,nonjnctermsMapToE[on_ok]]*eta-1)*psi(lambdap_on[nonjnctermsMapToE[on_ok]])+np.log(kappa))
        gammaTerm3_on -= np.sum(lambdap_on[nonjnctermsMapToE[on_ok]])

    if len(nonjnctermsMapToE[off_ok])>0:
        gammaTerm0_off -= np.sum(gammaln(b_off[0,nonjnctermsMapToE[off_ok]]*eta))
        gammaTerm1_off -= np.sum(b_off[0,nonjnctermsMapToE[off_ok]]*eta*np.log(kappa))
        gammaTerm2_off += np.sum((b_off[0,nonjnctermsMapToE[off_ok]]*eta-1)*psi(lambdap_off[nonjnctermsMapToE[off_ok]])+np.log(kappa))
        gammaTerm3_off -= np.sum(lambdap_off[nonjnctermsMapToE[off_ok]])
    
    jnxNoTuples = [term2junction[jnctermsNoForE[j]] for j in range(len(jnctermsNoForE))]
    jnxTuples = [term2junction[jnctermsForE[j]] for j in range(len(jnctermsForE))]
    cnt=0
    for EJncTuple in jnxNoTuples:
        term = jnctermsNoForE[cnt]
        
        onInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOn)
        offInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOff)
#         jnxp_1,jnxp_0=getJunctionProbs(onInconsistencies,offInconsistencies,EJncTuple,discrete_mapping)
        if offInconsistencies!=onInconsistencies and (onInconsistencies==0 or offInconsistencies==0):
            jnxTuples.append(EJncTuple)
            jnctermsForE.append(term)
        cnt+=1
    
    
    cnt=0
    for EJncTuple in jnxTuples:
        onInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOn)
        offInconsistencies = jnxTupleMuInconsistent(EJncTuple,muOff)
        jnxp_1,jnxp_0=getJunctionProbs(onInconsistencies,offInconsistencies,EJncTuple,discrete_mapping)

        gammaTerm0_on -= np.sum(gammaln(b_on[0,jnctermsForE[cnt]]*eta))
        gammaTerm0_off -= np.sum(gammaln(b_off[0,jnctermsForE[cnt]]*eta))
        gammaTerm1_on -= np.sum(b_on[0,jnctermsForE[cnt]]*eta*np.log(kappa))
        gammaTerm1_off -= np.sum(b_off[0,jnctermsForE[cnt]]*eta*np.log(kappa))
        gammaTerm2_on += np.sum((b_on[0,jnctermsForE[cnt]]*eta-1)*psi(lambdap_on[jnctermsForE[cnt]])+np.log(kappa))
        gammaTerm2_off += np.sum((b_off[0,jnctermsForE[cnt]]*eta-1)*psi(lambdap_off[jnctermsForE[cnt]])+np.log(kappa))
        gammaTerm3_on -= np.sum(lambdap_on[jnctermsForE[cnt]])
        gammaTerm3_off -= np.sum(lambdap_off[jnctermsForE[cnt]])
        cnt+=1

    term_1_all=gammaTerm0_on+gammaTerm1_on+gammaTerm2_on+gammaTerm3_on+(r-1+1)*np.log(muOn[e])+(s-1+0)*np.log(1-muOn[e])
    term_0_all=gammaTerm0_off+gammaTerm1_off+gammaTerm2_off+gammaTerm3_off+(r-1+0)*np.log(muOff[e])+(s-1+1)*np.log(1-muOff[e])

    return term_1_all,term_0_all

def getJunctionProbs(onInconsistencies,offInconsistencies,EJncTuple,discrete_mapping):
    if discrete_mapping:
        if onInconsistencies==0 and offInconsistencies==0:
            jnxp_1 = (1-MU_UNCERTAINTY)
            jnxp_0 = (1-MU_UNCERTAINTY)
        elif onInconsistencies>0 and offInconsistencies>0:
            jnxp_1 = MU_UNCERTAINTY
            jnxp_0 = MU_UNCERTAINTY
        elif onInconsistencies==0 and offInconsistencies>0:
            jnxp_1 = (1-MU_UNCERTAINTY)
            jnxp_0 = MU_UNCERTAINTY
        else:
            jnxp_1 = MU_UNCERTAINTY
            jnxp_0 = (1-MU_UNCERTAINTY)
    else:
        jnxp_1 = (1-MU_UNCERTAINTY)-(1-MU_UNCERTAINTY-MU_UNCERTAINTY)*(onInconsistencies/np.float(len(EJncTuple)-1))
        jnxp_0 = (1-MU_UNCERTAINTY)-(1-MU_UNCERTAINTY-MU_UNCERTAINTY)*(offInconsistencies/np.float(len(EJncTuple)-1))
        
    return (jnxp_1,jnxp_0)

def fast_expectation_deterministic_b_localsearch(lambdap,mus,term2exon,term2junction,eta,V,r,s,E_beta,nonjncterms,exonToTermsJncs,jncterms,exonToTermsNonJnc):
    # compute the dirichlet part of the \mu update
    #lambdap[lambdap<ERROR_P]=ERROR_P # all for some error 
    muprobs = []
    
    # compute probabilities for each mu, choose proportional to their normalized probability
    for mu in mus:
        dirichletTerm0=0
        dirichletTerm1=0
        dirichletTerm2=0
        mu[mu<MU_UNCERTAINTY]=MU_UNCERTAINTY
        mu[mu>(1-MU_UNCERTAINTY)]=1-MU_UNCERTAINTY
        for v in range(V):
            if v in term2junction:
                if jnxTupleMuInconsistent(term2junction[v],mu):
                    vprob = MU_UNCERTAINTY
                else:
                    vprob = (1-MU_UNCERTAINTY)
            else:
                if mu[term2exon[v]]>0.5:
                    vprob = 1-MU_UNCERTAINTY
                else:
                    vprob = MU_UNCERTAINTY
            
            dirichletTerm0+=vprob*eta
            dirichletTerm1+=gammaln(vprob*eta)
            dirichletTerm2+=((eta*vprob)-1)*E_beta[v]
        

        dirichletTerm0 = gammaln(dirichletTerm0)
        muprob=dirichletTerm0-dirichletTerm1+dirichletTerm2+np.sum((r-1+0)*np.log(mu[:])+(s-1+1)*np.log(1-mu[:]))
        muprobs.append(muprob)

    theargmax = np.argmax(muprobs, 0)
    return mus[theargmax]

def jnxTupleMuInconsistent(jnxTuple,mu):
    num_inconsistent=0
    for jncctr in range(len(jnxTuple)-1):
        # are they both in isoform and adjacent in isoform?
        if mu[jnxTuple[jncctr]]<EXISTANCE_THRESHOLD or mu[jnxTuple[jncctr+1]]<EXISTANCE_THRESHOLD or np.any(mu[jnxTuple[jncctr]+1:jnxTuple[jncctr+1]]>EXISTANCE_THRESHOLD):
            num_inconsistent+=1
    return num_inconsistent

def jnxTupleMuProb(jnxTuple,mu):
    total_prob=1
    for jncctr in range(len(jnxTuple)-1):
        # are they both in isoform and adjacent in isoform?
        total_prob*=mu[jnxTuple[jncctr]]*mu[jnxTuple[jncctr+1]]*np.prod(1-mu[jnxTuple[jncctr]+1:jnxTuple[jncctr+1]])
    if(total_prob<BPRIME_UNCERTAINTY):
        total_prob=BPRIME_UNCERTAINTY
    return total_prob

# exons must be tuple if if a single exon
def makeConsistentMu(exons,E):
    mu=np.zeros((E),dtype=np.float32)
    start=exons[0]
    end=exons[-1]

    injnx=0        
    for exon in range(E):
        # are they both in isoform and adjacent in isoform?
        if exon==start:
            injnx=1
            mu[exon]=1-MU_UNCERTAINTY
        elif exon==end:
            injnx=0
            mu[exon]=1-MU_UNCERTAINTY
        elif injnx==0:
            if np.random.random()<0.5:
                mu[exon]=1-MU_UNCERTAINTY
            else:
                mu[exon]=MU_UNCERTAINTY
        elif injnx==1:
            if exon in exons:
                mu[exon]=1-MU_UNCERTAINTY
            else:
                mu[exon]=MU_UNCERTAINTY
    return mu

def expectation_deterministic_b(lambdap,mu,e,term2exon,term2junction,eta,V,r,s,E_beta):
    # compute the dirichlet part of the \mu update
 
    ''' simulate beta from dirichlet(lambda) to produce a multinomial probability vector'''
    lambdap[lambdap<LAMBDAP_UNCERTAINTY]=LAMBDAP_UNCERTAINTY # all for some error 
    mu[mu<MU_UNCERTAINTY]=MU_UNCERTAINTY
    mu[mu>(1-MU_UNCERTAINTY)]=1-MU_UNCERTAINTY

    dirichletTerm0_on=0
    dirichletTerm0_off=0
    dirichletTerm0_shared=0
    dirichletTerm1_on=0
    dirichletTerm1_off=0
    dirichletTerm2_on=0
    dirichletTerm2_off=0
    for term in range(len(term2exon)):
        if term not in term2junction.keys() and term2exon[term]==e:
            dirichletTerm0_on += (1-LAMBDAP_UNCERTAINTY)*eta
            dirichletTerm0_off += LAMBDAP_UNCERTAINTY*eta
            dirichletTerm1_on += gammaln((1-LAMBDAP_UNCERTAINTY)*eta)
            dirichletTerm1_off += gammaln(LAMBDAP_UNCERTAINTY*eta)
            dirichletTerm2_on += (eta*(1-LAMBDAP_UNCERTAINTY)-1)*E_beta[term]
            dirichletTerm2_off += (eta*LAMBDAP_UNCERTAINTY-1)*E_beta[term]
        elif (term in term2junction.keys() and e in term2junction[term]):
            back = mu[e]
            mu[e]=1
            jnxp_1 = np.prod(mu[[i for i in term2junction[term]]])
            mu[e]=back
            dirichletTerm0_on += (1-LAMBDAP_UNCERTAINTY)*jnxp_1*eta
            dirichletTerm0_off += LAMBDAP_UNCERTAINTY*eta
            dirichletTerm1_on += gammaln(jnxp_1*eta)*jnxp_1+gammaln((1-(1-LAMBDAP_UNCERTAINTY)*jnxp_1)*eta)*(1-jnxp_1)
            dirichletTerm1_off += gammaln(LAMBDAP_UNCERTAINTY*eta)
            dirichletTerm2_on += (eta*(1-LAMBDAP_UNCERTAINTY)*jnxp_1-1)*E_beta[term]
            dirichletTerm2_off += (eta*LAMBDAP_UNCERTAINTY-1)*E_beta[term]
        elif term not in term2junction.keys():  #expectation
            dirichletTerm0_shared += eta*mu[term2exon[term]]
        else:
            jnxp_1 = np.prod(mu[[i for i in term2junction[term]]])
            dirichletTerm0_shared+= 1*jnxp_1*eta

    
    dirichletTerm0_on = gammaln(dirichletTerm0_on+dirichletTerm0_shared)
    dirichletTerm0_off = gammaln(dirichletTerm0_off+dirichletTerm0_shared)
    term_1_all=dirichletTerm0_on-dirichletTerm1_on+dirichletTerm2_on+(r-1+1)*np.log(mu[e])+(s-1+0)*np.log(1-mu[e])
    term_0_all=dirichletTerm0_off-dirichletTerm1_off+dirichletTerm2_off+(r-1+0)*np.log(mu[e])+(s-1+1)*np.log(1-mu[e])

    return term_1_all,term_0_all

def add_to_key(dictionary,to_add):
    newDict = dict()
    if not dictionary:
        newDict[round(to_add,2)]=0.5
    for key in dictionary.keys():
        newKey = round(key+to_add,2)
        newDict[newKey]=dictionary[key]

    return newDict

def add_to_new(dictionary,to_add_key_on,prob_on,prob_off):
    newDict = add_to_key(dictionary,to_add_key_on)
    for key in newDict.keys():
        newDict[key]*=prob_on
    for key in dictionary.keys():
        if key in newDict:
            newDict[key]+=(prob_off*dictionary[key])
        else: newDict[key]=(prob_off*dictionary[key])
        
    return newDict
    

def addIntoArray(start,end,badcntr,array_probs,array_cnts,prob,cnt,cnt_to_idx):
    probToRedistribute=0
    for idx in range(start,end):
        newProb = array_probs[idx]*prob
        if newProb < EXPECTATION_PROB_THRESHOLD:
            probToRedistribute+=newProb
            badcntr+=1
        else:
#             array_probs[idx-badcntr]=newProb
#             array_cnts[idx-badcntr]+=cnt
            newcnt=np.round(array_cnts[idx]+cnt,ROUND_SIG_FIGS)
            if newcnt in cnt_to_idx:
                badcntr+=1
                array_probs[cnt_to_idx[newcnt]]+=newProb
            else: 
                array_probs[idx-badcntr]=newProb
                array_cnts[idx-badcntr]=newcnt
                cnt_to_idx[newcnt]=idx-badcntr
    return badcntr,probToRedistribute