'''
Updates for the variational bayes algorithm
'''
import numpy as np

def softmax(w):
    w = np.array(w)
 
    maxes = np.amax(w)
    #maxes = maxes.reshape(maxes.shape[0], 1)
    e = np.exp(w - maxes)
    dist = e / np.sum(e)
    if not( np.sum(dist)==1 ):
        dist[-1]=1-np.sum(dist[:-1])
    assert np.sum(dist)==1
    return dist

''' for individual j, phi and words are indexed by j'''
def f_update_zeta(phi, E_beta, E_t, T,K,Nj,gamma1,C):
    zeta=np.zeros((T,K),dtype=np.float64)
    for l in range(T):
        if gamma1[l]!=0:
            for k in range(K):
                #print np.exp(E_t[k])
                #print np.exp(phi[:,l])
                #print np.exp(E_beta[k,words])
                #print np.exp(E_t[k] + np.dot(phi[:,l],E_beta[k,words]))
                zeta[l,k]= E_t[k] + np.dot(phi[:,l]*C[:],E_beta[k])
            # normalize
            #zeta[l,:] =  softmax(zeta[l,:])
            zeta[l][zeta[l]!=0] = np.exp(zeta[l][zeta[l]!=0] - np.max(zeta[l][zeta[l]!=0]))            
            zeta[l][zeta[l]!=0] = zeta[l][zeta[l]!=0]/np.sum(zeta[l][zeta[l]!=0])

#     assert np.allclose(np.sum(zeta,1),1) 
    return zeta

# defined for a particular person j in the outer iteration
def f_update_phi(zeta, E_psi, E_beta, T, Nj,goodLocalIso,C,Cbool):
#     phi=np.zeros((Nj,T),dtype=np.float64)
#     phi[:,goodLocalIso] = E_psi[goodLocalIso] + np.dot(zeta[goodLocalIso,:] , E_beta[:,words]).T
    phi=np.zeros((len(C),T),dtype=np.float64)
    phi[:,goodLocalIso] = E_psi[goodLocalIso] + np.dot(zeta[goodLocalIso,:], E_beta[:,:]).T
    # normalize
    # phi[i,:] = softmax(phi[i,:])
    phi[:,goodLocalIso] = np.exp(phi[:,goodLocalIso].T - np.amax(phi[:,goodLocalIso],axis=1)).T
    phi[:,goodLocalIso] = (phi[:,goodLocalIso].T/np.sum(phi[:,goodLocalIso],axis=1)).T
    phi[:,goodLocalIso]=(phi[:,goodLocalIso].T*Cbool).T
    assert np.allclose(np.sum(phi,1)[np.nonzero(Cbool)],1)

    
    return phi
            
            
            