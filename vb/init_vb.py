'''
Initialize variables for the variational bayes algorithm
'''
import numpy as np
from math import pow
from operator import mul
from itertools import combinations
from compute_expectations import jnxTupleMuInconsistent, makeConsistentMu, EXISTANCE_THRESHOLD, BPRIME_UNCERTAINTY, MU_UNCERTAINTY
import random

def initialize_iso_simple(K,nbExons,term2exon,term2junction,r,s):
    ''' This function defines an initial set of isoforms, whic are defined by the exons they contain'''
#     assert nbExons>=2
    # this is th emaximum number of unique isoforms with at least one exon
    b_prime=np.zeros((1,nbExons),dtype=np.float32)
    b_prime[:]=1-BPRIME_UNCERTAINTY
    return b_prime

def initialize_iso(K,nbExons,term2exon,term2junction,r,s):
    ''' This function defines an initial set of isoforms, whic are defined by the exons they contain'''
#     assert nbExons>=2
    # this is th emaximum number of unique isoforms with at least one exon
    maxiso= int(pow(2,nbExons))-1
    assert K<= maxiso
    
    b_prime=np.zeros((K,nbExons),dtype=np.float32)
    b_prime[:]=BPRIME_UNCERTAINTY
    
    
    if K == maxiso:
        # if the initial number of isoforms required is the max, just enumerate all the possibilities
        # this can be when the number of exons is small enough for instance
        for k in range(1,K+1):
            a=bin(int(k))[2:].zfill(nbExons)
            for i in range(nbExons):
                b_prime[k-1,i]=int(a[i])
    else:

        # build up initial isoforms greedily, make sure each read term has a matching isoform
        mu=np.zeros((nbExons),dtype=np.float32)
        mu[:]=1-BPRIME_UNCERTAINTY
        b_prime[0]=mu[:]
        curr_mus = 1
        
        (exonsToTerms,exons) = getExonToTerms(term2exon,term2junction)
        complxJncs = getTermsComplexJncs(term2exon,term2junction)
        

        complxJncsUniq = getTermsComplexJncsUniq(term2exon,term2junction)

        # the first third, is just random exons        
        exon_inclusion_prob = r/(r+s)
        for k in range(1,np.int(np.floor(K/3))):
            mu=np.zeros((nbExons),dtype=np.float32)
            mu[:]=MU_UNCERTAINTY
            for i in range(nbExons):
                if np.random.random() < exon_inclusion_prob:
                    mu[i]=1-MU_UNCERTAINTY
            b_prime[k,:]=mu        
        # the next third is sampling a random number of random terms 
        for k in range(np.int(np.floor(K/3)),np.int(np.floor(2*K/3))):            
            mu=np.zeros((nbExons),dtype=np.float32)
            for ctr in range(np.int(np.ceil(exon_inclusion_prob*nbExons))):
                randomJnc = random.sample(complxJncsUniq,1)[0]
                for jncctr in range(len(randomJnc)-1):
                    mu[randomJnc[jncctr]]=1-MU_UNCERTAINTY
                    mu[randomJnc[jncctr+1]]=1-MU_UNCERTAINTY
                    mu[randomJnc[jncctr]+1:randomJnc[jncctr+1]]=MU_UNCERTAINTY
            for e in range(len(mu)):
                if mu[e]==0:
                    mu[e]=np.random.random_sample()
            b_prime[k,:]=mu  
        # lastly, sample in proportion to number of read terms   
        for k in range(np.int(np.floor(2*K/3)),K):
            mu=np.zeros((nbExons),dtype=np.float32)
            for ctr in range(np.int(np.ceil(exon_inclusion_prob*nbExons))):
                randomJnc = random.sample(complxJncs,1)[0]
                for jncctr in range(len(randomJnc)-1):
                    mu[randomJnc[jncctr]]=1-MU_UNCERTAINTY
                    mu[randomJnc[jncctr+1]]=1-MU_UNCERTAINTY
                    mu[randomJnc[jncctr]+1:randomJnc[jncctr+1]]=MU_UNCERTAINTY
            for e in range(len(mu)):
                if mu[e]==0:
                    mu[e]=np.random.random_sample()
            b_prime[k,:]=mu  

                
    return b_prime

def getExonToTerms(term2exon,term2junction):
    exonsToTerms = dict()
    for term in range(len(term2exon)):
        if term in term2junction:
            for exon in term2junction[term]:
                if exon not in exonsToTerms:
                    exonsToTerms[exon]=set()
                exonsToTerms[exon].add(term2junction[term])
        else:
            if term2exon[term] not in exonsToTerms:
                exonsToTerms[term2exon[term]]=set()
            exonsToTerms[term2exon[term]].add(term2exon[term])
    exons = sorted(exonsToTerms.keys())
    return (exonsToTerms,exons)

def getTermsComplexJncs(term2exon,term2junction):
    terms = []
    for term in range(len(term2exon)):
        if term in term2junction:
            if (len(term2junction[term])>2) or ((term2junction[term][0]+1)!=term2junction[term][1]):
                for exon in term2junction[term]:
                    terms.append(term2junction[term])
    return terms

def getTermsComplexJncsUniq(term2exon,term2junction):
    terms = set()
    for term in range(len(term2exon)):
        if term in term2junction:
            if (len(term2junction[term])>2) or ((term2junction[term][0]+1)!=term2junction[term][1]):
                for exon in term2junction[term]:
                    terms.add(term2junction[term])
    return terms

def initialize_iso_splicegraph(K,nbExons,term2exon,term2junction,maxisos):
    ''' This function defines an initial set of isoforms, whic are defined by the exons they contain'''
    ''' initialize only feasible isoforms from the splice graph'''
    (exonsToTerms,exons) = getExonToTerms(term2exon,term2junction)
    isoforms = []
#     isoforms.append(np.zeros((nbExons),dtype=np.float32))
    isoforms.append(np.ones((nbExons),dtype=np.float32))
    for exon in exons:
        if(len(isoforms)>maxisos):
            return None
        unalignedTerms = [];
        for term in exonsToTerms[exon]:
            numberGood=0
            for isoform in isoforms:
                if type(term) is np.int32:
                    if (isoform[term]==1):
                        numberGood+=1
                else:
                    if (jnxTupleMuInconsistent(term,isoform)==0):
                        numberGood+=1
            if numberGood == 0:
                unalignedTerms.append(term)
        for term in unalignedTerms:
            toappend=[]
            for isoform in isoforms:
                isoformcopy = np.copy(isoform)
                if type(term) is tuple:
                    for jncctr in range(len(term)-1):
                        isoformcopy[term[jncctr]]=1
                        isoformcopy[term[jncctr+1]]=1
                        isoformcopy[term[jncctr]+1:term[jncctr+1]]=0
                else:
                    isoformcopy[term]=1
                toappend.append(isoformcopy)
            for isoform in toappend:
                isoforms.append(isoform)
#     isoforms.pop(0)
    return np.asanyarray(isoforms,dtype=np.float32)
    
def check_iso(b_prime,junction_set): # TODO:FIX ME, it's wrong, throws away good isoforms
    ''' this functions checks that the present set of isoforms does accommodate the sets of junction reads '''
    nbK=b_prime.shape[0]
    nbE=b_prime.shape[1]
    new_bprime=np.zeros((nbK,nbE), dtype=np.float32)
    nb_iso_bprime=0
    toChange=range(nbK)   
    
    for junction in junction_set:
        ju=[ex for ex in junction]
        if ju[1]-ju[0] > 1:
            motif= np.asarray([1]+[0 for i in range(ju[1]-ju[0]-1)]+[1])
        else:
            motif=np.asarray([1,1])
        
        #find that motif in the right submatrix of b_prime
        submatrix=new_bprime[:,ju[0]:(ju[1]+1)]
        # first look into new bprime
        found_newbprime=False
        for j in range(nb_iso_bprime):
            if np.all(submatrix[j,:]==motif):
                found_newbprime=True
                break
                
        if found_newbprime:
            continue
        else:
            # look into the isoform of the original isoform set, that can still be transformed
            submatrix=b_prime[:,ju[0]:(ju[1]+1)]
            found=False
            for j in toChange:
                if np.all(submatrix[j,:]==motif):
                    found=True
                    # make sure this isoform is not subsequently transformed
                    toChange.remove(j)
                    # add it to the set of final isoforms
                    new_bprime[nb_iso_bprime,:] = b_prime[j,:]
                    nb_iso_bprime += 1
                    break
            
            if not(found):
                ind=toChange[0]
                #change corresponding iso to include the junction
                b_prime[ind,ju[0]:(ju[1]+1)]=motif
                toChange.remove(ind)
                new_bprime[nb_iso_bprime,:] = b_prime[ind,:]
                nb_iso_bprime += 1
    # finally add the isoforms from the original set tha thave not been yet modified
    for j in toChange:  
        new_bprime[nb_iso_bprime,:] = b_prime[j,:]
        nb_iso_bprime += 1
        
    return new_bprime


def compatibility(b,M,N,X,Kmin):
    ''' returns a compatibility matrix, given the term composition of the isoforms (in matrix b) 
    Serves in the initialization of zeta'''
    comp=[]
    for j in range(M):
        comp.append(np.zeros((max(N),Kmin),dtype=np.float32))
    for j in range(M):
        for i in range(N[j]):
            # which dictionary term is read i in individual j?
            term=X[j,i]
            # which isoforms is it compatible with? TODO:FIX ME, maybe assign reads proportional to probability?
            potential_iso=np.where(b[:,term]>EXISTANCE_THRESHOLD)[0]
            # assign those isoforms equal assignment probability for this read
            if len(potential_iso)==0:
                potential_iso=range(0,Kmin)
            comp[j][i,potential_iso]= 1/np.float32(len(potential_iso))
#             comp[j,i,:]=b[:,term]/np.sum(b[:,term])
    return comp

def compatibility_v(b,Kmin,j,C,V):
    ''' returns a compatibility matrix, given the term composition of the isoforms (in matrix b) 
    Serves in the initialization of zeta'''
    comp=np.zeros((V,Kmin),dtype=np.float32)
    for v in range(V):
        if C[j,v]>0:
            # which isoforms is it compatible with? TODO:FIX ME, maybe assign reads proportional to probability?
            potential_iso=np.where(b[:,v]>EXISTANCE_THRESHOLD)[0]
            # assign those isoforms equal assignment probability for this read
            if len(potential_iso)==0:
                potential_iso=range(0,Kmin)
            comp[v,potential_iso]= 1/np.float32(len(potential_iso))

    return comp


def initialize_z(b,M, N, X):
    '''This function is to build an assignment matrix (of reads in each individual to isoforms).
    Not used anymore in the current version of the code
    Was used in the version where phi was initialized first and then zeta'''
    z = np.zeros((M,max(N)),dtype=np.int16)
#         can_be_removed={}
#         can_be_removed_combined={}
    for j in range(M):
        for i in range(N[j]):
            # which dictionary term is read i in individual j?
            term=X[j,i]                 
            # look for all isoforms where that read in 'on'
            potential_iso=np.where(b[:,term]>0)[0]
                # choose randomly from those
            if len(potential_iso)==1:
                z[j,i]=potential_iso
            else:
                # remove one possibility
                z[j,i]=np.random.choice(potential_iso)
    return z


#### from Blei et al
def log_normalize(v):
    ''' return log(sum(exp(v)))'''

    log_max = 100.0
    if len(v.shape) == 1:
        max_val = np.max(v)
        log_shift = log_max - np.log(len(v)+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift))
        log_norm = np.log(tot) - log_shift
        v = v - log_norm
    else:
        max_val = np.max(v, 1)
        log_shift = log_max - np.log(v.shape[1]+1.0) - max_val
        tot = np.sum(np.exp(v + log_shift[:,np.newaxis]), 1)

        log_norm = np.log(tot) - log_shift
        v = v - log_norm[:,np.newaxis]

    return (v, log_norm)