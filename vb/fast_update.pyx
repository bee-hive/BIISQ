# cython: boundscheck=False
# cython: cdivision=False
# cython: wraparound=False
# cython: nonecheck=False
# cython: profile=True

from __future__ import division
import numpy as np
from cython import boundscheck, wraparound
cimport numpy as np
import sys
from ctypes import py_object
from libcpp cimport bool
from libcpp.vector cimport vector
from libc.math cimport log, exp, fabs, floor
from libc.float cimport DBL_MAX
from libc.stdio cimport printf, fprintf, stderr
import cython
# from compute_expectations import expectation_log_dirichlet,expectation_log_sigma_beta, expectation_log_beta
# from time import time
# from initializeVB import compatibility
from cython.parallel import prange, parallel

#cdef extern from "Gamma.h":
#    cdef double LogGamma(double) nogil
#    cdef double Gamma(double) nogil

DTYPEINT = np.int32
ctypedef np.int32_t DTYPEINT_t

DTYPELONG = np.int64
ctypedef np.int64_t DTYPELONG_t

DTYPEFLOAT = np.float32
ctypedef np.float32_t DTYPEFLOAT_t

DTYPEDOUBLE = np.float64
ctypedef np.float64_t DTYPEDOUBLE_t

DTYPEUINT = np.uint32
ctypedef np.uint32_t DTYPEUINT_t


def compute_likelihood_per_person(np.ndarray X,np.ndarray phi,np.ndarray zeta,np.ndarray lambdap,np.ndarray good_k,
                                  np.ndarray E_freqIso,np.ndarray N, DTYPEINT_t numthreads,DTYPEINT_t V,np.ndarray C):
    cdef np.ndarray[DTYPEFLOAT_t, ndim=3] read_assignments = np.einsum('jvl,jlk->jvk',phi,zeta)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=3] n_cnts = np.zeros((zeta.shape[0],lambdap.shape[0],lambdap.shape[1]),dtype=np.float32) # global iso x read-terms    
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] reads_per_iso = np.zeros((zeta.shape[0],lambdap.shape[0]),dtype=DTYPEFLOAT)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=1] likely=np.zeros(zeta.shape[0],dtype=np.float32)
    return compute_likelihood_per_personCdef(lambdap,good_k,read_assignments,n_cnts,X,reads_per_iso,E_freqIso,N,likely,numthreads,V,C)
 
cdef DTYPEFLOAT_t[:] compute_likelihood_per_personCdef(np.ndarray lambdap,np.ndarray good_k,np.ndarray read_assignments,
                                                       np.ndarray n_cnts, np.ndarray X,np.ndarray reads_per_iso,np.ndarray E_freqIso,
                                                       np.ndarray N, np.ndarray likely, DTYPEINT_t numthreads,DTYPEINT_t V,np.ndarray C) nogil :

    cdef DTYPEFLOAT_t[:,:] E_freqIso_view = E_freqIso
    cdef DTYPEINT_t[:] N_view = N
    cdef DTYPEFLOAT_t[:,:,:] read_assignments_view = read_assignments
    cdef DTYPEFLOAT_t[:,:] lambdap_view = lambdap
    cdef DTYPELONG_t[:] good_k_view = good_k
    cdef DTYPEFLOAT_t[:,:,:] n_cnts_view = n_cnts
    cdef DTYPEINT_t[:,:] X_view = X
    cdef DTYPEFLOAT_t[:,:] C_view = C
    cdef DTYPEFLOAT_t[:,:] reads_per_iso_view = reads_per_iso
    cdef DTYPEFLOAT_t[:] likely_view = likely
    cdef DTYPEINT_t itr, i, j, v
    cdef DTYPELONG_t k
    cdef DTYPEFLOAT_t totalReads = 0
    for j in prange(E_freqIso_view.shape[0],nogil=True,num_threads=numthreads):
        for v in range(V):
            if C_view[j,v]>0:
                for itr in range(good_k_view.shape[0]):
                    k=good_k_view[itr]
                    n_cnts_view[j,k,v]+=read_assignments_view[j,v,k]*C_view[j,v]
#     n_cnts[:,flat_X]+=read_assignments[:,:]
    for j in prange(n_cnts_view.shape[0],nogil=True,num_threads=numthreads):
        for itr in range(good_k_view.shape[0]):
            k=good_k_view[itr]
            for i in range(n_cnts_view.shape[2]):
                reads_per_iso_view[j,k] += n_cnts_view[j,k,i]

    for j in prange(n_cnts_view.shape[0],nogil=True,num_threads=numthreads):
        for itr in range(good_k_view.shape[0]):
            k=good_k_view[itr]
            if reads_per_iso_view[j,k] > 0 and E_freqIso_view[j,k]>0:
                likely_view[j]+=log(E_freqIso_view[j,k])+dirMult(n_cnts_view[j,k],reads_per_iso_view[j,k],lambdap_view[k])
#                 likely_view[j]+=dirMult(n_cnts_view[j,k],reads_per_iso_view[j,k],lambdap_view[k])
    return likely_view

cdef DTYPEFLOAT_t dirMult(DTYPEFLOAT_t[:] n_cnts, DTYPEFLOAT_t num_reads,DTYPEFLOAT_t[:] lambdap) nogil :
    cdef DTYPEFLOAT_t sum_alpha = 0, coeff = 0
    cdef DTYPEINT_t it = 0
    for it in range(lambdap.shape[0]):
        sum_alpha += lambdap[it]
    coeff=LogGamma(num_reads+1)+LogGamma(sum_alpha)-LogGamma(num_reads+sum_alpha)
    for it in range(lambdap.shape[0]):
        coeff+=LogGamma(n_cnts[it]+lambdap[it])-LogGamma(n_cnts[it]+1)-LogGamma(lambdap[it])
    return coeff

def updateMuForIsoform(int E,np.ndarray mu,np.ndarray grp,np.ndarray term2exon,term2junction,float  eta,int V,float  r,float  s,np.ndarray E_beta,np.ndarray nonjncterms,exonToTermsJncs,np.ndarray jncterms,exonToTermsNonJnc,np.ndarray bad_k,np.ndarray term2exonsCovered,np.ndarray term2exonsCoveredLengths,float MU_UNCERTAINTY,float EXISTANCE_THRESHOLD,float BPRIME_UNCERTAINTY):
    cdef float[:,:] E_beta_view = E_beta
    cdef int[:,:] term2exonsCovered_view = term2exonsCovered
    cdef int[:] term2exonsCoveredLengths_view = term2exonsCoveredLengths
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] b_on = np.zeros((1, V),dtype=np.float32)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] b_off = np.zeros((1, V),dtype=np.float32)
    cdef float[:,:] b_on_view = b_on
    cdef float[:,:] b_off_view = b_off
    cdef long[:] jnctermsNoForE_view
    cdef vector[ int ] jnctermsForE
    cdef np.ndarray[DTYPELONG_t, ndim=1] jnctermsNoForE
    cdef vector[ int[:] ] jnxNoTuples
    cdef vector[ int[:] ] jnxTuples
    cdef vector[ int ] jnxNoTuplesLengths
    cdef vector[ int ] jnxTuplesLengths
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2]  muOn
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2]  muOff
    cdef float[:,:] muOn_view
    cdef float[:,:] muOff_view
    cdef int k, jnct, idxterm, termIdx
    cdef float prob1
    cdef int maxTerm = term2exonsCovered.shape[1]
    cdef np.ndarray[DTYPEINT_t, ndim=1] termArray = np.zeros((maxTerm),dtype=np.int32)
    cdef int[:] termArray_view = termArray
    cdef np.ndarray[DTYPELONG_t, ndim=1] nonjnctermsMapToE
    cdef np.ndarray[DTYPELONG_t, ndim=1] nonjnctermsNoMapToE
    
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] muCopy = np.zeros((mu.shape[0],mu.shape[1]),dtype=np.float32)
    muCopy[:,:] = mu[:,:]
    cntr=0
    muOn=np.zeros((1,muCopy.shape[1]),dtype=np.float32)
    muOff=np.zeros((1,muCopy.shape[1]),dtype=np.float32)
    for k in grp:
        if k not in bad_k:
            for e in range(E):
                jnxNoTuples.clear()
                jnctermsForE.clear()
                jnxTuples.clear()
                jnxNoTuplesLengths.clear()
                jnxTuplesLengths.clear()
                nonjnctermsMapToE=nonjncterms[np.where(term2exon[nonjncterms]==e)]
                nonjnctermsNoMapToE=nonjncterms[np.where(term2exon[nonjncterms]!=e)]
                if e in exonToTermsJncs:
                    jnctermsForE_py=exonToTermsJncs[e].tolist()
                else:
                    jnctermsForE_py=[]
                for jnct in jnctermsForE_py:
                    jnctermsForE.push_back(jnct)
                jnctermsNoForE=jncterms[np.in1d(jncterms,jnctermsForE_py)==False]
                jnctermsNoForE_view = jnctermsNoForE
                
                for j in range(len(jnctermsNoForE_view)):
                    termArray = np.zeros((maxTerm),dtype=np.int32)
                    termArray_view=termArray
                    idxterm = jnctermsNoForE_view[j]
                    for termIdx in range(maxTerm):
                        termArray_view[termIdx]=term2exonsCovered_view[idxterm,termIdx]
#                    jnxNoTuples.push_back(term2exonsCovered_view[idxterm])
                    jnxNoTuples.push_back(termArray_view)
                    jnxNoTuplesLengths.push_back(term2exonsCoveredLengths_view[idxterm])                    
                
 #               sys.stderr.write("number of tuples " + str(jnxNoTuples.size()))
 #               sys.stderr.write("number of terms " + str(len(jnctermsNoForE_view)))
                for j in range(jnctermsForE.size()):
                    termArray = np.zeros((maxTerm),dtype=np.int32)
                    termArray_view=termArray
                    idxterm = jnctermsForE[j]
                    for termIdx in range(maxTerm):
                        termArray_view[termIdx]=term2exonsCovered_view[idxterm,termIdx]
                    jnxTuples.push_back(termArray_view)
                    jnxTuplesLengths.push_back(term2exonsCoveredLengths_view[idxterm])
                
                
                for j in range(muCopy.shape[1]):
                    muOn[0,j]=muCopy[k,j]
                    muOff[0,j]=muCopy[k,j]
                muOn_view = muOn
                muOff_view = muOff
                for j in range(V):
                    b_on_view[0,j]=BPRIME_UNCERTAINTY
                    b_off_view[0,j]=BPRIME_UNCERTAINTY
                
 #               fprintf( stderr, "before fast_expectation..." )
                prob1 = fast_expectation_deterministic_bCdef(
                muCopy[k,:],
                e,
                eta,
                V,
                r,
                s,
                E_beta_view[k],
                term2exonsCovered_view,
                term2exonsCoveredLengths_view,
                MU_UNCERTAINTY,
                EXISTANCE_THRESHOLD,
                BPRIME_UNCERTAINTY,
                b_on_view,
                b_off_view,
                nonjnctermsMapToE,
                nonjnctermsNoMapToE,
                jnctermsForE,
                jnctermsNoForE,
                jnxNoTuples,
                jnxTuples,
                jnxNoTuplesLengths,
                jnxTuplesLengths,
                muOn_view,
                muOff_view)
                
                muCopy[k,e] = prob1
                cntr+=1
                for j in range(len(muCopy[k])):
                    if muCopy[k,j]<MU_UNCERTAINTY:
                        muCopy[k,j]=MU_UNCERTAINTY
                    if muCopy[k,j]>1-MU_UNCERTAINTY:
                        muCopy[k,j]=1-MU_UNCERTAINTY
                
    return muCopy[grp]

cdef float fast_expectation_deterministic_bCdef(float[:] mu,int e,float  eta,int V,float  r,float  s,float[:] E_beta, int[:,:] term2exonsCovered,int[:] term2exonsCoveredLengths,float MU_UNCERTAINTY,float EXISTANCE_THRESHOLD,float BPRIME_UNCERTAINTY,float[:,:] b_on,float[:,:] b_off,long[:] nonjnctermsMapToE,long[:] nonjnctermsNoMapToE,vector[int] jnctermsForE,long[:] jnctermsNoForE,vector[ int[:] ] jnxNoTuples,vector[ int[:] ] jnxTuples, vector[ int ] jnxNoTuplesLengths, vector[ int ] jnxTuplesLengths,float[:,:] muOn,  float[:,:] muOff) nogil:
    
    cdef float dirichletTerm0_on = 0
    cdef float dirichletTerm1_on = 0
    cdef float dirichletTerm2_on = 0
    cdef float dirichletTerm0_off = 0
    cdef float dirichletTerm1_off = 0
    cdef float dirichletTerm2_off = 0
    cdef float dirichletTerm0_shared = 0
    cdef float term_1_all = 0
    cdef float term_0_all = 0
    cdef float x = 0
    cdef float y = 0
    cdef int cnt = 0
    cdef int term = 0
    cdef int eterm = 0
    cdef float max = 0
    cdef float result = 0
    cdef int[:] EJncTuple
    cdef int length = 0
    cdef int inti = 0
    cdef int maxTerm = term2exonsCovered.shape[1]
    muOn[0,e]=(1-MU_UNCERTAINTY)
    b_on = construct_b_from_bprimeCdefCPrims(V,muOn,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD,b_on)
    
    muOff[0,e]=MU_UNCERTAINTY
    b_off = construct_b_from_bprimeCdefCPrims(V,muOff,term2exonsCovered,term2exonsCoveredLengths,1,BPRIME_UNCERTAINTY,EXISTANCE_THRESHOLD,b_off)

    for term in range(nonjnctermsMapToE.shape[0]):
        if b_on[0][nonjnctermsMapToE[term]]>BPRIME_UNCERTAINTY:
            dirichletTerm0_on += (float)(nonjnctermsMapToE.shape[0])*(eta)
            dirichletTerm1_on += (float)(nonjnctermsMapToE.shape[0])*LogGamma(eta)
            for eterm in range(nonjnctermsMapToE.shape[0]):
                dirichletTerm2_on += (eta-1)*E_beta[nonjnctermsMapToE[eterm]]

        if b_off[0][nonjnctermsMapToE[term]]>BPRIME_UNCERTAINTY:
            dirichletTerm0_off += (float)(nonjnctermsMapToE.shape[0])*(eta)
            dirichletTerm1_off += (float)(nonjnctermsMapToE.shape[0])*LogGamma(eta)
            for eterm in range(nonjnctermsMapToE.shape[0]):
                dirichletTerm2_off += (eta-1)*E_beta[nonjnctermsMapToE[eterm]]

    for term in range(nonjnctermsNoMapToE.shape[0]):
        dirichletTerm0_shared += (eta)
    for inti in range(jnxNoTuples.size()):
        EJncTuple = jnxNoTuples[inti]
        term = jnctermsNoForE[inti]
        onInconsistencies=jnxTupleMuInconsistentCdefCPrim(term2exonsCovered[term],term2exonsCoveredLengths[term],muOn[0],EXISTANCE_THRESHOLD)
        offInconsistencies=jnxTupleMuInconsistentCdefCPrim(term2exonsCovered[term],term2exonsCoveredLengths[term],muOff[0],EXISTANCE_THRESHOLD)
        if offInconsistencies==0 and onInconsistencies==0:
            dirichletTerm0_shared+=eta
        elif onInconsistencies==offInconsistencies:
            dirichletTerm0_shared+=eta
        if (offInconsistencies==0 and onInconsistencies>0) or (offInconsistencies>0 and onInconsistencies==0):
#            jnxTuples.push_back(term2exonsCovered[cnt])
            jnxTuples.push_back(EJncTuple)
            jnxTuplesLengths.push_back(jnxNoTuplesLengths[cnt])
            jnctermsForE.push_back(term)
            
        cnt+=1
    cnt=0
    
    for term in range(jnxTuples.size()):
        EJncTuple = jnxTuples[term]
        length = jnxTuplesLengths[term]
        onInconsistencies=jnxTupleMuInconsistentCdefCPrim(term2exonsCovered[jnctermsForE[cnt]],term2exonsCoveredLengths[jnctermsForE[cnt]],muOn[0],EXISTANCE_THRESHOLD)
        offInconsistencies=jnxTupleMuInconsistentCdefCPrim(term2exonsCovered[jnctermsForE[cnt]],term2exonsCoveredLengths[jnctermsForE[cnt]],muOff[0],EXISTANCE_THRESHOLD)

        if b_on[0][jnctermsForE[cnt]]>BPRIME_UNCERTAINTY:
            dirichletTerm0_on += eta
            dirichletTerm1_on += LogGamma(eta)
            dirichletTerm2_on += (eta-1)*E_beta[jnctermsForE[cnt]]

        if b_off[0][jnctermsForE[cnt]]>BPRIME_UNCERTAINTY:
            dirichletTerm0_off += eta
            dirichletTerm1_off += LogGamma(eta)
            dirichletTerm2_off += (eta-1)*E_beta[jnctermsForE[cnt]]
        cnt+=1
    dirichletTerm0_on = LogGamma(dirichletTerm0_on+dirichletTerm0_shared)
    dirichletTerm0_off = LogGamma(dirichletTerm0_off+dirichletTerm0_shared)
    term_1_all=dirichletTerm0_on-dirichletTerm1_on+dirichletTerm2_on+(r-1+1)*log(muOn[0,e])+(s-1+0)*log(1-muOn[0,e])
    term_0_all=dirichletTerm0_off-dirichletTerm1_off+dirichletTerm2_off+(r-1+0)*log(muOff[0,e])+(s-1+1)*log(1-muOff[0,e])

    if term_1_all > term_0_all:
        max = term_1_all
    else:
        max = term_0_all
    x=exp(term_1_all-max) # to avoid underflow
    y=exp(term_0_all-max)
    result = (x/(x+y))
    return result


def fastUpdateLambda(np.ndarray grp,np.long_t M,np.long_t V,np.ndarray phi,np.ndarray zeta,np.ndarray lambdap,np.ndarray b,
                     np.float32_t eta,np.ndarray X,np.ndarray starts,np.ndarray ends,np.int32_t D,np.ndarray bad_k, np.ndarray C):
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] lambdapCopy = np.zeros((len(grp), lambdap.shape[1]),dtype=DTYPEFLOAT)
    cdef int max_t = 0
    for j from 0 <= j < M:
        if phi[j].shape[1]>max_t:
            max_t=phi[j].shape[1] 
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] phisums = np.zeros((M,max_t),dtype=DTYPEFLOAT)
    return np.asarray(fastUpdateLambdaCdef(grp,M,V,phi,zeta,lambdap,b,eta,starts,ends,D,lambdapCopy,phisums,bad_k,C))

cdef DTYPEFLOAT_t[:,:] fastUpdateLambdaCdef(np.ndarray grp,np.long_t M,np.long_t V,np.ndarray phi,np.ndarray zeta,np.ndarray lambdap,
                                            np.ndarray b,np.float32_t eta,np.ndarray starts,np.ndarray ends,np.int32_t D, 
                                            np.ndarray lambdapCopy, np.ndarray phisums,np.ndarray bad_k, np.ndarray C) nogil:
    
    cdef DTYPEFLOAT_t[:,:] lambdapCopy_view = lambdapCopy   
    cdef DTYPEFLOAT_t thislambda
    cdef DTYPEINT_t j,v,vit,it,kit,t,k,kit2,max_t,idx=0
    cdef DTYPEFLOAT_t[:,:] phisums_view = phisums
    cdef DTYPELONG_t[:] grp_view = grp
    cdef DTYPEFLOAT_t[:,:,:] phi_view = phi
    cdef DTYPEFLOAT_t[:,:,:] zeta_view = zeta
    cdef DTYPEFLOAT_t[:,:] lambdap_view = lambdap
    cdef DTYPEFLOAT_t[:,:] b_view = b
    cdef DTYPEFLOAT_t[:,:] C_view = C
    cdef DTYPEINT_t[:,:] starts_view = starts
    cdef DTYPEINT_t[:,:] ends_view = ends
    cdef DTYPELONG_t[:] bad_k_view = bad_k
    cdef DTYPEINT_t skip
    for v from 0 <= v < V:
        phisums_view[:,:] = 0
        for j from 0 <= j < M:
            for it from 0 <= it < phi_view[j].shape[1]:
                phisums_view[j,it]=phisums_view[j,it]+(phi_view[j,v,it]*C_view[j,v]) # people by isoforms
        idx=0

        for kit from 0 <= kit < grp_view.shape[0]:
            k = grp_view[kit]
            skip=0
            for kit2 from 0 <= kit2 < bad_k_view.shape[0]:
                if k==bad_k_view[kit2]:
                    skip = 1
                    break
            if skip>0:
                idx+=1
                continue
            
            thislambda=0
            for j from 0 <= j < M:
                for t from 0 <= t < zeta_view[j].shape[0]:
                    thislambda += zeta_view[j,t,k]*phisums_view[j,t]
#             lambdapCopy[k,v]=b[k,v]*(eta+thislambda)
            lambdapCopy_view[idx,v]=b_view[k,v]*(eta+D*thislambda)
            idx+=1
    return lambdapCopy_view

def construct_b_from_bprime(np.long_t V,np.ndarray iso,np.ndarray term2exonsCovered,np.ndarray term2exonsCoveredLengths, np.int_t discrete_mapping, DTYPEFLOAT_t BRPIME_UNCERTAINTY, DTYPEFLOAT_t EXISTANCE_THRESHOLD):
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] b = np.zeros((iso.shape[0], V),dtype=DTYPEFLOAT)
    b+=BRPIME_UNCERTAINTY
    return np.asarray(construct_b_from_bprimeCdef(V,iso,term2exonsCovered,term2exonsCoveredLengths,discrete_mapping,BRPIME_UNCERTAINTY,EXISTANCE_THRESHOLD,b))

cdef DTYPEFLOAT_t[:,:] construct_b_from_bprimeCdef(np.long_t V,DTYPEFLOAT_t[:,:] iso_view,DTYPEINT_t[:,:] term2exonsCovered_view,DTYPEINT_t[:] term2exonsCoveredLengths_view, np.int_t discrete_mapping, DTYPEFLOAT_t BRPIME_UNCERTAINTY,  DTYPEFLOAT_t EXISTANCE_THRESHOLD, DTYPEFLOAT_t[:,:] b_view) nogil:
    cdef DTYPEINT_t nbIso=iso_view.shape[0] 
#    cdef DTYPEFLOAT_t[:,:] b_view = b
    cdef DTYPEINT_t jncctr,num_inconsistent,v,inner_ctr
    
    
    # for all terms that maps within a single exon, assign the value form the exon selector vector corresponding to that exon
    for i in range(nbIso): # for all isoforms
        for v in range(V):
            if term2exonsCoveredLengths_view[v]==1:
                if iso_view[i,term2exonsCovered_view[v,0]]>EXISTANCE_THRESHOLD: # if the exon ex exists in isoform i 
                    if(discrete_mapping):
                        b_view[i,v]=1-BRPIME_UNCERTAINTY
                    else:
                        b_view[i,v]=iso_view[i,term2exonsCovered_view[v,0]]
            else:
                # for now assume not read paired... when read pair we need to be smart how we assign compatible reads
                #if np.all(iso_view[i,list(combinations(term2junction[jnc],2))]>EXISTANCE_THRESHOLD).all(): # if the exon ex exists in isoform i 
                num_inconsistent=0
                
                for jncctr in range(term2exonsCoveredLengths_view[v]-1):
                    # are they both in isoform and adjacent in isoform?
                    if iso_view[i,term2exonsCovered_view[v,jncctr]]<EXISTANCE_THRESHOLD or iso_view[i,term2exonsCovered_view[v,jncctr+1]]<EXISTANCE_THRESHOLD:
                        num_inconsistent+=1
                    else:
                        for inner_ctr in range(term2exonsCovered_view[v,jncctr]+1,term2exonsCovered_view[v,jncctr+1]):
                            if(iso_view[i,inner_ctr]>EXISTANCE_THRESHOLD):
                                num_inconsistent+=1
                                break
                if(discrete_mapping):
                    if num_inconsistent>0:
                        b_view[i,v]=BRPIME_UNCERTAINTY
                    else:
                        b_view[i,v]=1-BRPIME_UNCERTAINTY
                else:
                    if iso_view[i,term2exonsCovered_view[v][term2exonsCoveredLengths_view[v]-1]]<EXISTANCE_THRESHOLD:
                        num_inconsistent+=1
                    b_view[i,v]=(1-BRPIME_UNCERTAINTY)-(1-BRPIME_UNCERTAINTY)*((float)(num_inconsistent))/((float)(term2exonsCoveredLengths_view[v]))
    return b_view
    
cdef float[:,:] construct_b_from_bprimeCdefCPrims(np.long_t V,float[:,:] iso_view,int[:,:] term2exonsCovered_view,int[:] term2exonsCoveredLengths_view, np.int_t discrete_mapping, float BRPIME_UNCERTAINTY,  float EXISTANCE_THRESHOLD, float[:,:] b_view) nogil :
    cdef int nbIso=iso_view.shape[0] 
#    cdef float[:,:] b_view = b
    cdef int jncctr,num_inconsistent,v,inner_ctr
    
    # for all terms that maps within a single exon, assign the value form the exon selector vector corresponding to that exon
    for i in range(nbIso): # for all isoforms
        for v in range(V):
            if term2exonsCoveredLengths_view[v]==1:
                if iso_view[i,term2exonsCovered_view[v,0]]>EXISTANCE_THRESHOLD: # if the exon ex exists in isoform i 
                    if(discrete_mapping):
                        b_view[i,v]=1-BRPIME_UNCERTAINTY
                    else:
                        b_view[i,v]=iso_view[i,term2exonsCovered_view[v,0]]
            else:
                # for now assume not read paired... when read pair we need to be smart how we assign compatible reads
                #if np.all(iso_view[i,list(combinations(term2junction[jnc],2))]>EXISTANCE_THRESHOLD).all(): # if the exon ex exists in isoform i 
                num_inconsistent=0
                
                for jncctr in range(term2exonsCoveredLengths_view[v]-1):
                    # are they both in isoform and adjacent in isoform?
                    if iso_view[i,term2exonsCovered_view[v,jncctr]]<EXISTANCE_THRESHOLD or iso_view[i,term2exonsCovered_view[v,jncctr+1]]<EXISTANCE_THRESHOLD:
                        num_inconsistent+=1
                    else:
                        for inner_ctr in range(term2exonsCovered_view[v,jncctr]+1,term2exonsCovered_view[v,jncctr+1]):
                            if(iso_view[i,inner_ctr]>EXISTANCE_THRESHOLD):
                                num_inconsistent+=1
                                break
                if(discrete_mapping):
                    if num_inconsistent>0:
                        b_view[i,v]=BRPIME_UNCERTAINTY
                    else:
                        b_view[i,v]=1-BRPIME_UNCERTAINTY
                else:
                    if iso_view[i,term2exonsCovered_view[v][term2exonsCoveredLengths_view[v]-1]]<EXISTANCE_THRESHOLD:
                        num_inconsistent+=1
                    b_view[i,v]=(1-BRPIME_UNCERTAINTY)-(1-BRPIME_UNCERTAINTY)*((float)(num_inconsistent))/((float)(term2exonsCoveredLengths_view[v]))
    return b_view

# make into cython function
def initLambda(DTYPEINT_t K,DTYPEINT_t V,DTYPELONG_t[:] nonjnxkeys,DTYPELONG_t[:] jnxkeys,DTYPEFLOAT_t[:,:] mu,DTYPEFLOAT_t eta,
               DTYPEFLOAT_t existence_threshold,DTYPEFLOAT_t BPRIME_UNCERTAINTY,DTYPEFLOAT_t LAMBDAP_UNCERTAINTY, DTYPEINT_t numthreads,
               DTYPEINT_t[:,:] term2exonsCovered, DTYPEINT_t[:] term2exonsCoveredLengths):
    initlambdap=np.zeros((K,V), dtype=np.float32)
    cdef DTYPEFLOAT_t[:,:] initlambdap_view=initlambdap
    cdef DTYPEINT_t num_inconsistent,term,k,itr
    cdef DTYPELONG_t v
    for k in prange(K,nogil=True,num_threads=numthreads):
        for itr in range(jnxkeys.shape[0]):
            v = jnxkeys[itr]
            num_inconsistent=0
#                 jnx = term2junction[v]
            num_inconsistent=jnxTupleMuInconsistentCdef(term2exonsCovered[v],term2exonsCoveredLengths[v],mu[k],existence_threshold)
#                 num_inconsistent=jnxTupleMuInconsistent(jnx,mu[k])
#                 consistentProb=jnxTupleMuProb(jnx,mu[k])
            if num_inconsistent==0:
                initlambdap_view[k,v] = eta
            else:
                initlambdap_view[k,v] = LAMBDAP_UNCERTAINTY
#                     lambdap[k,v] = 1-np.float(num_inconsistent)/np.float(length)
#                     lambdap[k,v]=consistentProb
        for itr in range(nonjnxkeys.shape[0]):
            num_inconsistent=0
            v=nonjnxkeys[itr]
            term = term2exonsCovered[v,0]
            if mu[k][term]<existence_threshold:
                num_inconsistent=1
#                 consistentProb=mu[k][jnx]
            if num_inconsistent==0:
                initlambdap_view[k,v] = eta
            else:
                initlambdap_view[k,v] = LAMBDAP_UNCERTAINTY
                
    initlambdap[initlambdap<LAMBDAP_UNCERTAINTY]=LAMBDAP_UNCERTAINTY
    return initlambdap

# cdef DTYPEFLOAT_t[:,:] jnxTupleMuInconsistentCdef(DTYPEINT_t[:] jnxTuple_view, DTYPEINT_t tupleSize,DTYPEFLOAT_t[:] mu_view,DTYPEFLOAT_t EXISTANCE_THRESHOLD) nogil:


def jnxTupleMuInconsistent(np.ndarray jnxTuple, DTYPEINT_t tupleSize,np.ndarray mu,DTYPEFLOAT_t EXISTANCE_THRESHOLD):
    return jnxTupleMuInconsistentCdef(jnxTuple,tupleSize,mu,EXISTANCE_THRESHOLD)

cdef DTYPEINT_t jnxTupleMuInconsistentCdef(DTYPEINT_t[:] jnxTuple_view, DTYPEINT_t tupleSize,DTYPEFLOAT_t[:] mu_view,DTYPEFLOAT_t EXISTANCE_THRESHOLD) nogil:
    cdef DTYPEINT_t jncctr,inner_ctr,num_inconsistent=0
#    cdef DTYPEINT_t[:] jnxTuple_view = jnxTuple
#    cdef DTYPEFLOAT_t[:] mu_view = mu
    for jncctr from 0 <= jncctr < tupleSize-1:
        # are they both in isoform and adjacent in isoform?
        if mu_view[jnxTuple_view[jncctr]]<EXISTANCE_THRESHOLD or mu_view[jnxTuple_view[jncctr+1]]<EXISTANCE_THRESHOLD:
            num_inconsistent+=1
        else:
            for inner_ctr from jnxTuple_view[jncctr]+1 <= inner_ctr < jnxTuple_view[jncctr+1]:
                if mu_view[inner_ctr]>EXISTANCE_THRESHOLD:
                    num_inconsistent+=1
                    break
    return num_inconsistent
    
cdef int jnxTupleMuInconsistentCdefCPrim(int[:] jnxTuple_view, int tupleSize,float[:] mu_view,float EXISTANCE_THRESHOLD) nogil:
    cdef int jncctr,inner_ctr,num_inconsistent=0
#    cdef int[:] jnxTuple_view = jnxTuple
#    cdef float[:] mu_view = mu
    for jncctr from 0 <= jncctr < tupleSize-1:
        # are they both in isoform and adjacent in isoform?
        if mu_view[jnxTuple_view[jncctr]]<EXISTANCE_THRESHOLD or mu_view[jnxTuple_view[jncctr+1]]<EXISTANCE_THRESHOLD:
            num_inconsistent+=1
        else:
            for inner_ctr from jnxTuple_view[jncctr]+1 <= inner_ctr < jnxTuple_view[jncctr+1]:
                if mu_view[inner_ctr]>EXISTANCE_THRESHOLD:
                    num_inconsistent+=1
                    break
    return num_inconsistent
    
cdef double Gamma(double x) nogil:
    if (x <= 0.0):
        return 0

    cdef double gamma 
    gamma = 0.577215664901532860606512090

    if (x < 0.001):
        return 1.0/(x*(1.0 + gamma*x))

    cdef double y,num,den,z,result,temp
    cdef int n, i
    cdef bint arg_was_less_than_one
    cdef double[8] p, q
    if (x < 12.0):
        y = x
        n = 0
        arg_was_less_than_one = (y < 1.0)
        
        if (arg_was_less_than_one):
            y += 1.0
        else:
            n = ( (int)(floor(y)) )
            n -= 1
            y -= n

        p[0] = -1.71618513886549492533811E+0
        p[1] = 2.47656508055759199108314E+1
        p[2] = -3.79804256470945635097577E+2
        p[3] = 6.29331155312818442661052E+2
        p[4] = 8.66966202790413211295064E+2
        p[5] = -3.14512729688483675254357E+4
        p[6] = -3.61444134186911729807069E+4
        p[7] = 6.64561438202405440627855E+4

        q[0] = -3.08402300119738975254353E+1
        q[1] = 3.15350626979604161529144E+2
        q[2] = -1.01515636749021914166146E+3
        q[3] = -3.10777167157231109440444E+3
        q[4] = 2.25381184209801510330112E+4
        q[5] = 4.75584627752788110767815E+3
        q[6] = -1.34659959864969306392456E+5
        q[7] = -1.15132259675553483497211E+5

        num = 0.0
        den = 1.0
        z = y - 1
        for i in range(8):
            num = (num + p[i])*z
            den = den*z + q[i]

        result = num/den + 1.0

        if (arg_was_less_than_one):
            result /= (y-1.0)
        else:
            for i in range(n):
                result *= y
                y+=1
        return result

    if (x > 171.624):
        temp = DBL_MAX
        return temp*2.0

    return exp( LogGamma(x) )

cdef double LogGamma(double x) nogil:
    if (x <= 0.0):
        return 0

    if (x < 12.0):
        return log(fabs(Gamma(x)))

    cdef double z, sum, series, logGamma, halfLogTwoPi
    cdef double[8] c
    c[0] =1.0/12.0
    c[1] =-1.0/360.0
    c[2] =1.0/1260.0
    c[3] =-1.0/1680.0
    c[4] =1.0/1188.0
    c[5] =-691.0/360360.0
    c[6] =1.0/156.0
    c[7] =-3617.0/122400.0

    z = 1.0/(x*x)
    sum = c[7]
    for i in range(6,-1,-1):
        sum *= z
        sum += c[i]
    series = sum/x

    halfLogTwoPi = 0.91893853320467274178032973640562
    logGamma = (x - 0.5)*log(x) - x + halfLogTwoPi + series
    return logGamma

""" for updateIndividualPlate """
def updateIndividualPlate(\
                        np.ndarray grp                  , \
                        np.ndarray X                    , \
                        np.ndarray N                    , \
                        bool update_gamma               , \
                        bool update_zeta                , \
                        bool update_phi                 , \
                        bool update_a                   , \
                        bool update_lambda              , \
                        np.ndarray gamma1               , \
                        np.ndarray gamma2               , \
                        np.ndarray phi                  , \
                        np.ndarray zeta                 , \
                        np.ndarray a1                   , \
                        np.ndarray a2                   , \
                        float omega                     , \
                        np.ndarray E_psi                , \
                        np.ndarray E_beta               , \
                        float alpha                     , \
                        np.ndarray E_t                  , \
                        int M                           , \
                        int K                           , \
                        np.ndarray lambdap              , \
                        int V, np.ndarray mu            , \
                        np.ndarray term2exon            , \
                        term2junction                   , \
                        float eta                       , \
                        int stochastic                  , \
                        float converge                  , \
                        float INDIVIDUAL_PLATE_CONVERGE , \
                        int INDIVIDUAL_PLATE_MAXIT      \
                        ):
    # initialize buffers
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] gamma1Copy = np.zeros((gamma1.shape[0], gamma1.shape[1]), dtype=gamma1.dtype)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=2] gamma2Copy = np.zeros((gamma2.shape[0], gamma2.shape[1]), dtype=gamma2.dtype)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=3] zetaCopy = np.zeros((zeta.shape[0], zeta.shape[1], zeta.shape[2]), dtype=zeta.dtype)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=3] lastZeta = np.zeros((zeta.shape[0], zeta.shape[1], zeta.shape[2]), dtype=zeta.dtype)
    cdef np.ndarray[DTYPEFLOAT_t, ndim=3] phiCopy = np.zeros((phi.shape[0], phi.shape[1], phi.shape[2]), dtype=phi.dtype)
    
    cdef int T = zeta.shape[1] # assume T is the same for each individual
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] zeta_buf = np.zeros((T, K), dtype=DTYPEDOUBLE)
    cdef np.ndarray[DTYPEDOUBLE_t, ndim=2] phi_buf = np.zeros((np.max(N),T), dtype=DTYPEDOUBLE)

    cdef int i, k, t, h, it, idx
    cdef int N_j
    cdef long jidx, gidx
    cdef vector[ long ] gamma1_gzero_idx

    # cdef double gamma1_sum, gamma2_sum
    
    # cdef float dot_sum=0.0
    # cdef double amax=0.0, normalize_sum=0.0

    # cdef int dim2 = E_beta.shape[0]
    # cdef int goodLocalIso_num
    cdef bint check_converge
    
    # create memory views
    cdef DTYPELONG_t[:]         grp_view = grp
    cdef DTYPEUINT_t[:]         N_view = N
    cdef DTYPEINT_t[:, :]       X_view = X
    cdef DTYPEFLOAT_t[:, :]     gamma1_view = gamma1
    cdef DTYPEFLOAT_t[:, :, :]  phi_view = phi

    cdef DTYPEFLOAT_t[:, :]     gamma1Copy_view = gamma1Copy
    cdef DTYPEFLOAT_t[:, :]     gamma2Copy_view = gamma2Copy
    cdef DTYPEFLOAT_t[:, :, :]  zetaCopy_view = zetaCopy
    cdef DTYPEFLOAT_t[:, :, :]  lastZeta_view = lastZeta
    cdef DTYPEFLOAT_t[:, :, :]  phiCopy_view = phiCopy

    cdef DTYPEDOUBLE_t[:, :]    zeta_buf_view = zeta_buf
    cdef DTYPEDOUBLE_t[:, :]    phi_buf_view = phi_buf

    cdef DTYPEDOUBLE_t[:]       E_t_view    = E_t
    cdef DTYPEFLOAT_t[:,:]      E_beta_view = E_beta
    cdef DTYPEFLOAT_t[:, :]     E_psi_view = E_psi

    ''' loop for local parameters to converge, now only 2 for reasons of speed '''
    it = 0
    while(1):
        for idx in range(grp_view.shape[0]):
            jidx = grp_view[idx]
            # T = zetaCopy_view[jidx].shape[0]
            N_j = N_view[jidx]
            # words = X[j, :N[j]]

            if update_gamma:
                '''update gamma1,gamma2: var param for psi'''
                gamma1Copy_view[jidx, :] = f_update_gamma1_cdef(\
                                                                T                       , \
                                                                gamma1_view[jidx,:]     , \
                                                                phi_view[jidx,:,:]      , \
                                                                gamma1Copy_view[jidx,:] \
                                                                )
                gamma2Copy_view[jidx, :] = f_update_gamma2_cdef(\
                                                                T                       , \
                                                                gamma1_view[jidx,:]     , \
                                                                phi_view[jidx,:,:]      , \
                                                                alpha                   , \
                                                                gamma2Copy_view[jidx,:] \
                                                                )


            if update_zeta:
                '''update zeta: var param for c'''
                if it == 0:
                    zeta_buf_view = f_update_zeta_cdef(\
                                                        phi_view[jidx,:,:]      , \
                                                        E_beta_view             , \
                                                        E_t_view                , \
                                                        X_view[jidx,:]          , \
                                                        T                       , \
                                                        K                       , \
                                                        N_j                     , \
                                                        gamma1_view[jidx,:]     , \
                                                        zeta_buf_view           \
                                                        )
                else:
                    zeta_buf_view = f_update_zeta_cdef(\
                                                        phiCopy_view[jidx,:,:]  , \
                                                        E_beta_view             , \
                                                        E_t_view                , \
                                                        X_view[jidx,:]          , \
                                                        T                       , \
                                                        K                       , \
                                                        N_j                     , \
                                                        gamma1_view[jidx,:]     , \
                                                        zeta_buf_view           \
                                                        )

                    
                # copy and round zeta
                for t in range(T):
                    for k in range(K):
                        zetaCopy_view[jidx, t, k] = zeta_buf_view[t, k]

            if update_phi:
                '''update phi: var param for z'''
                gamma1_gzero_idx.clear()
                for gidx in range(gamma1_view.shape[1]):
                    if(gamma1_view[jidx, gidx] > 0):
                        gamma1_gzero_idx.push_back(gidx)
                # goodLocalIso_num = int(gamma1_gzero_idx.size())
                phi_buf_view = f_update_phi_cdef(\
                                                zetaCopy_view[jidx,:,:] , \
                                                E_psi_view[jidx,:]      , \
                                                E_beta_view             , \
                                                X_view[jidx,:]          , \
                                                T                       , \
                                                N_j                     , \
                                                gamma1_gzero_idx        , \
                                                phi_buf_view            \
                                                )


                # copy and round phi
                for h in range(N_j):
                    for t in range(T):
                        phiCopy_view[jidx, h, t] = phi_buf_view[h, t]
        it += 1
        check_converge = f_check_converge_cdef(jidx, T, K, zetaCopy_view, lastZeta_view, INDIVIDUAL_PLATE_CONVERGE)      
        if ( ((it > 0)&(stochastic == 0))|(check_converge == 1)|(it > INDIVIDUAL_PLATE_MAXIT) ):
            break

        lastZeta_view = f_deep_copy_cdef(\
                                            zetaCopy_view     , \
                                            lastZeta_view     \
                                            ) 
        # for i in range(zetaCopy_view.shape[0]):
        #     for t in range(zetaCopy_view.shape[1]):
        #         for k in range(zetaCopy_view.shape[2]):
        #             lastZeta_view[i, t, k] = zetaCopy_view[i, t, k]

    # wrap up return variables
    gamma1Copy = np.asarray(gamma1Copy_view)
    gamma2Copy = np.asarray(gamma2Copy_view)
    zetaCopy = np.asarray(zetaCopy_view)
    phiCopy = np.asarray(phiCopy_view)
    return ([gamma1Copy[i] for i in grp], [gamma2Copy[i]  for i in grp], [zetaCopy[i] for i in grp], [phiCopy[i] for i in grp])

cdef DTYPEFLOAT_t[:, :, :] f_deep_copy_cdef(\
                                            DTYPEFLOAT_t[:, :, :] zetaCopy_view     , \
                                            DTYPEFLOAT_t[:, :, :] lastZeta_view     \
                                            ) nogil:
    cdef int i, t, k
    for i in range(zetaCopy_view.shape[0]):
        for t in range(zetaCopy_view.shape[1]):
            for k in range(zetaCopy_view.shape[2]):
                lastZeta_view[i, t, k] = zetaCopy_view[i, t, k]
    return lastZeta_view

cdef bint f_check_converge_cdef(\
                                int jidx , int T, int K, \
                                DTYPEFLOAT_t[:, :, :]  zeta_copy_view  , \
                                DTYPEFLOAT_t[:, :, :]  last_zeta_view  , \
                                float               converge_thr    \
                                ) nogil:
    cdef bint converge = 1
    cdef float temp_abs_diff = 0.0
    cdef int t, k
    for t in range(T):
        for k in range(K):
            temp_abs_diff = fabs(zeta_copy_view[jidx, t, k]-last_zeta_view[jidx, t, k])
            if(temp_abs_diff >= converge_thr):
                converge = 0
                break
    return converge

cdef DTYPEFLOAT_t[:] f_update_gamma1_cdef(\
                                            int                 T                 , \
                                            DTYPEFLOAT_t[:]     gamma1_j_view     , \
                                            DTYPEFLOAT_t[:, :]  phi_j_view        , \
                                            DTYPEFLOAT_t[:]     gamma1Copy_j_view  \
                                            ) nogil:
    cdef int t, k
    cdef double gamma1_sum
    for t in range(T):
        # compute sum terms
        if(gamma1_j_view[t] > 0):
            gamma1_sum = 0.0
            for k in range(phi_j_view.shape[0]):
                gamma1_sum += phi_j_view[k, t]
            gamma1Copy_j_view[t] = 1. + gamma1_sum
    return gamma1Copy_j_view

cdef DTYPEFLOAT_t[:] f_update_gamma2_cdef(\
                                            int                 T                  , \
                                            DTYPEFLOAT_t[:]     gamma1_j_view      , \
                                            DTYPEFLOAT_t[:, :]  phi_j_view         , \
                                            float               alpha              , \
                                            DTYPEFLOAT_t[:]     gamma2Copy_j_view  \
                                            ) nogil:
    cdef int t, k, h
    cdef double gamma2_sum

    for t in range(T):
        # compute sum terms
        if(gamma1_j_view[t] > 0):
            gamma2_sum = 0.0
            for k in range(phi_j_view.shape[0]):
                for h in range(t+1, phi_j_view.shape[1]):
                    gamma2_sum += phi_j_view[k, h]
            gamma2Copy_j_view[t] = alpha + gamma2_sum
    return gamma2Copy_j_view


cdef DTYPEDOUBLE_t[:,:] f_update_zeta_cdef(\
                                            DTYPEFLOAT_t[:,:]   phi_j_view      , \
                                            DTYPEFLOAT_t[:,:]   E_beta_view     , \
                                            DTYPEDOUBLE_t[:]    E_t_view        , \
                                            DTYPEINT_t[:]       X_j_view        , \
                                            int                 T               , \
                                            int                 K               , \
                                            int                 N_j             , \
                                            DTYPEFLOAT_t[:]     gamma1_j_view   , \
                                            DTYPEDOUBLE_t[:,:]  zeta_buf_view    \
                                            ) nogil:
    cdef int t, k, h
    cdef float dot_sum=0.0
    cdef double amax=0.0, normalize_sum=0.0
    for t in range(T):
        for k in range(K):
            zeta_buf_view[t, k] = 0.0

    for t in range(T):
        if(gamma1_j_view[t] != 0):
            for k in range(K):
                dot_sum = 0.0
                for h in range(N_j):
                    # if it == 0:
                    dot_sum += phi_j_view[h, t] * E_beta_view[k, X_j_view[h]]
                    # else:
                    #     dot_sum += phiCopy_j_view[h, t] * E_beta_view[k, X_view[h]]
                zeta_buf_view[t, k] = E_t_view[k] + dot_sum

            amax = 0.0
            for k in range(K):
                if(zeta_buf_view[t, k] != 0):
                    if(k == 0):
                        amax = zeta_buf_view[t, k]
                    else:
                        if(zeta_buf_view[t, k] > amax):
                            amax = zeta_buf_view[t, k]

            normalize_sum = 0.0
            for k in range(K):
                if(zeta_buf_view[t, k] != 0):
                    zeta_buf_view[t, k] = exp(zeta_buf_view[t, k] - amax)
                    normalize_sum += zeta_buf_view[t, k]

            for k in range(K):
                if(zeta_buf_view[t, k] != 0):
                    zeta_buf_view[t, k] /= normalize_sum
    return zeta_buf_view


cdef DTYPEDOUBLE_t[:,:] f_update_phi_cdef(\
                                            DTYPEFLOAT_t[:,:]   zetaCopy_j_view     , \
                                            DTYPEFLOAT_t[:]     E_psi_j_view        , \
                                            DTYPEFLOAT_t[:,:]   E_beta_view         , \
                                            DTYPEINT_t[:]       X_j_view            , \
                                            int                 T                   , \
                                            int                 N_j                 , \
                                            vector[ long ]      gamma1_gzero_idx    , \
                                            DTYPEDOUBLE_t[:,:]  phi_buf_view        \
                                            ) nogil:
    cdef int i, k, h, t
    cdef int goodLocalIso_num = int(gamma1_gzero_idx.size())
    cdef int dim2= E_beta_view.shape[0]
    cdef float dot_sum=0.0
    cdef double amax=0.0, normalize_sum=0.0

    for h in range(N_j):
        for t in range(T):
            phi_buf_view[h, t] = 0.0

    for h in range(N_j):
        amax=0.0
        for k in range(goodLocalIso_num):
            dot_sum = 0.0
            for i in range(dim2):
                dot_sum += E_beta_view[i, X_j_view[h]]*zetaCopy_j_view[gamma1_gzero_idx[k], i]
            phi_buf_view[h, gamma1_gzero_idx[k]] = E_psi_j_view[gamma1_gzero_idx[k]] + dot_sum
            if(k == 0):
                amax = phi_buf_view[h, gamma1_gzero_idx[k]]
            else:
                if(phi_buf_view[h, gamma1_gzero_idx[k]] > amax):
                    amax = phi_buf_view[h, gamma1_gzero_idx[k]]

        normalize_sum=0.0
        for k in range(goodLocalIso_num):
            phi_buf_view[h, gamma1_gzero_idx[k]] = exp(phi_buf_view[h, gamma1_gzero_idx[k]] - amax)
            normalize_sum += phi_buf_view[h, gamma1_gzero_idx[k]]

        for k in range(goodLocalIso_num):
            phi_buf_view[h, gamma1_gzero_idx[k]] /= normalize_sum
    return phi_buf_view

