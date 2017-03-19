'''
Opens result files from the VB algorithm and draws plots
'''

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import transformations as tf
import os
from mpl_toolkits.mplot3d import Axes3D
from itertools import *



COLOR_MAP = plt.get_cmap("Accent")
NUM_INDIVIDUALS_X = 5
NUM_INDIVIDUALS_Y = 10
PROPS = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

##############################################
# Ground truth
# K_true: number of true isoforms
##############################################
def ground_truth(DATA_DIR, N_exons, V, term2exon, term2junction,true_isoforms,count_term_iso_ind, M, K_true, genes):

    
    prop_per_ind=np.sum(count_term_iso_ind,1)/float(1000)
    
    fig, ax = plt.subplots(NUM_INDIVIDUALS_X, NUM_INDIVIDUALS_Y, sharex=True, sharey=True)
    
    plt.xlim(0,K_true)
    plt.ylim(0,1)    
    create_axes_title('isoform','normalized read count','true read counts by isoform for each individual',fig)
    
    for i in range(NUM_INDIVIDUALS_X):
        for j in range(NUM_INDIVIDUALS_Y):
            ax[i, j].bar(np.linspace(0.6,K_true-0.4,K_true),prop_per_ind[i*NUM_INDIVIDUALS_Y+j,:])
            ax[i, j].set_xticks(range(1,K_true+1))
            ax[i, j].set_xticklabels(range(1,K_true+1))
            ax[i, j].set_xbound(lower=0,upper=K_true+1)

    fig.text(0.02, 0.96,'iso:<e0> <e1>...\n'+tf.get_string_from_array(true_isoforms),  fontsize=14,verticalalignment='top', bbox=PROPS)
    fig.savefig('ground_truth.pdf')
    

##############################################
# variational bayes results
##############################################
def variational_bayes_zeta(N_exons, V, term2exon, term2junction, M, N, K, E, zeta, it_num=-1):
    
    # results are indexed by <number of iterations> 
    zeta0=zeta[it_num,:,:,:]

    fig, ax = plt.subplots(NUM_INDIVIDUALS_X, NUM_INDIVIDUALS_Y, sharex=True, sharey=True)
    create_axes_title('global isoform distribution index','person isoform distribution index','zeta heat map for all individuals for iteration ' + `it_num`,fig)
        
    for i in range(NUM_INDIVIDUALS_X):
        for j in range(NUM_INDIVIDUALS_Y):
            mesh = ax[i, j].pcolormesh(zeta0[i*NUM_INDIVIDUALS_Y+j,:,:])
            ax[i, j].set_xticks(np.linspace(0.5,zeta0.shape[1]-0.5,zeta0.shape[1]))
            ax[i, j].set_xticklabels(range(1,zeta0.shape[1]+1))
            ax[i, j].set_yticks(np.linspace(0.5,zeta0.shape[2]-0.5,zeta0.shape[2]))
            ax[i, j].set_yticklabels(range(1,zeta0.shape[2]+1))

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(mesh, cax=cbar_ax)
    isoform_string = tf.get_isoform_string(K, E)
    fig.text(0.02, 0.96, isoform_string,  fontsize=14,verticalalignment='top', bbox=PROPS)
    fig.savefig('zeta_iteration_'+str(it_num)+'.pdf')


def variational_bayes_a(N_exons, V, term2exon, term2junction, M, N, K, a1, a2):
    # E(t_k), global isoform proportions
    num_iterations=a1.shape[0]
    E_t = np.zeros((num_iterations,K),dtype=np.float64)
    tmp=a1/(a1+a2)
    E_t[:,0]= tmp[:,0]
    prod=1-tmp[:,0]
    for k in range(0,K):
        E_t[:,k]= tmp[:,k]*prod
        prod=prod*(1-tmp[:,k])
    
    # convergence of global isoform proportions
    fig = plt.figure()
    fig.hold(True)
    for k in range(0,K):
        plt.plot(E_t[:,k], linewidth=3.0)
    plt.legend(range(1,K+1))
    create_axes_title('iteration','global isoform proportion','global isoform proportions over all iterations',fig)
    fig.savefig('a.pdf')
    fig.hold(False)

def variational_bayes_phi(N_exons, V, term2exon, term2junction, M, N, K, E, phi, person_index, it_num=-1):
    fig = plt.figure()
    fig.hold(True)
    phi0=phi[it_num,person_index,:,:]
    for k in range(0,K):
        plt.plot(phi0[:,k],'o')
    plt.legend(range(1,K+1))
    create_axes_title('read','\phi_{j='+`person_index`+',i} = P(z_{ij}=c_{jl})','phi for all reads',fig)
    fig.savefig('phi_iteration_'+str(it_num)+'.pdf')
    fig.hold(False)
    
    
def variational_bayes_phi_zeta(N_exons, V, term2exon, term2junction, M, N, K, E, phi, zeta, person_index=-2,it_num=-1):
    fig = plt.figure()
    fig.hold(True)
    zeta0=zeta[it_num,:,:,:]
    phi0=phi[it_num,:,:,:]
    P_zjik = compute_phi_zeta_prob(M, N, K, phi0, zeta0)

    if person_index==-2:
        person_index = range(0,M) # override default
    
    if isinstance(person_index,(int,long)):
        plt.plot(np.sum(P_zjik[person_index],1).T/max(N),'o')
    else: 
        for k in person_index:
            plt.plot(np.sum(P_zjik[k],1).T/max(N),'o')
        plt.legend(person_index)
    
    create_axes_title('read','sum(phi[j,i,:]*zeta[j,:,k])','P_zjik for iteration ' + `it_num` + ' and people: ' + `person_index`,fig)
    fig.savefig('phi_zeta_iteration_'+str(it_num)+'.pdf')
    fig.hold(False)
    

def variational_bayes_pzjik(N_exons, V, term2exon, term2junction, M, N, K, E, phi, zeta, it_num=-1):
    zeta0=zeta[it_num,:,:,:]
    phi0=phi[it_num,:,:,:]
    P_zjik = compute_phi_zeta_prob(M, N, K, phi0, zeta0)
    fig, ax = plt.subplots(NUM_INDIVIDUALS_X, NUM_INDIVIDUALS_Y, sharex=True, sharey=True)
    create_axes_title('isoforms','P_zjik','P_zjik for all individuals for iteration ' + `it_num`,fig)
        
    for i in range(NUM_INDIVIDUALS_X):
        for j in range(NUM_INDIVIDUALS_Y):
            ax[i, j].bar(range(K),(np.sum(P_zjik,1).T/max(N))[:,i*NUM_INDIVIDUALS_Y+j])
            ax[i, j].set_ylim(0,1)
            ax[i, j].set_xticks(np.linspace(0.5,K-0.5,K))
            ax[i, j].set_xticklabels(range(1,K+1))
            ax[i, j].set_xbound(lower=0,upper=K+1)
       
    fig.savefig('pzjik_iteration_'+str(it_num)+'.pdf')
    
def variational_bayes_pzjik_by_reads(N_exons, V, term2exon, term2junction, M, N, K, E, phi, zeta, person_index=0, it_num=-1):
    zeta0=zeta[it_num,:,:,:]
    phi0=phi[it_num,:,:,:]
    P_zjik = compute_phi_zeta_prob(M, N, K, phi0, zeta0)
    fig = plt.figure()
    plt.plot(P_zjik[0,:,:],'o')
    plt.legend(range(1,K+1))
    create_axes_title('reads','P(z_{ji}=k)','P(z_{ji}=k) for person '+`person_index`+' and all isoforms',fig)
    fig.savefig('pzjik_by_reads_iteration_'+str(it_num)+'.pdf')


#def variational_bayes_rho1(N_exons, V, term2exon, term2junction, M, N, K, E, rho1):
#def variational_bayes_rho2(N_exons, V, term2exon, term2junction, M, N, K, E, rho2):
def variational_bayes_mu(N_exons, V, term2exon, term2junction, M, N, K, E, mu):
    # quels exons contient chaque isoforme
    # initial average exon composition
    fig = plt.figure()
    plt.subplot(1,2,1)
    plt.pcolormesh(mu[0,:,:])
    plt.xlabel('exon')
    plt.ylabel('isoform')
    plt.title('exon composition parameter mu for first iteration')
    plt.xticks(0.5+np.arange(3))
    plt.xticks(0.5+np.arange(3))
    plt.gca().set_xticklabels(range(1,4))
    plt.gca().set_yticklabels(range(1,K+1))
    # final exon composition
    plt.subplot(1,2,2)
    plt.pcolor(mu[-1,:,:])
    plt.colorbar()
    plt.xlabel('exon')
    plt.ylabel('isoform')
    plt.title('exon composition parameter mu for last iteration')
    plt.gca().set_xticks(np.linspace(0.5, 2.5, 3))
    plt.gca().set_yticks(np.linspace(0.5, 6.5, K))
    plt.gca().set_xticklabels(range(1,4))
    plt.gca().set_yticklabels(range(1,K+1))
    fig.savefig('mu_iteration.pdf')
    
def variational_bayes_beta(N_exons, V, term2exon, term2junction, M, N, K, E, lambdap):
    num_iterations = lambdap.shape[0]
    E_beta=np.zeros((num_iterations,K,V),dtype=np.float64)
    for l in range(num_iterations):
        x=lambdap[l,:,:]
        E_beta[l,:,:] = (x.T/x.sum(axis=1)).T
    
    # view convergence from 0 to 100 iterations
    fig = plt.figure()
    plt.hold(True)
    plt.plot(E_beta[-1,:,:].T,'o')
    plt.plot(E_beta[0,:,:].T,'x')
    create_axes_title('Dictionary read terms','Topic composition','lambdap for all isoforms and aligned read positions',fig)
    fig.savefig('beta.pdf')
    
def variational_bayes_lambdap(N_exons, V, term2exon, term2junction, M, N, K, E, lambdap):
    fig = plt.figure()
    plt.plot(((lambdap[-1,:,:]>0)*np.arange(1,K+1)[:,None]).T,'o')
    create_axes_title('aligned read positions','lambdap','lambdap for all isoforms and aligned read positions',fig)
    plt.ylim(0,K+1)
    fig.savefig('lambdap.pdf')

def print_4d_array(ind1, ind2, array_to_print):
    print_2d_array(array_to_print[ind1,ind2,:,:])

def print_3d_array(ind1, array_to_print):
    print_2d_array(array_to_print[ind1,:,:])

def print_2d_array(array_to_print):
    pretty_print_string = ''
    for i in range(array_to_print.shape[0]):
        for j in range(array_to_print.shape[1]):
            pretty_print_string = pretty_print_string + `array_to_print[i,j]` + "\t"
        pretty_print_string = pretty_print_string[:-1]
        pretty_print_string = pretty_print_string + "\n"
    pretty_print_string = pretty_print_string[:-1]
    print pretty_print_string

def summarize_array(fixed_indices, array_to_print, function_to_apply):
    dimensions = range(len(array_to_print.shape))
    return_dimensions = list()
    indices = list()
    variable_indices = list(dimensions)
    variable_ranges = list()

    for i in dimensions:
        if i in fixed_indices:
            indices.append(slice(array_to_print.shape[i]))
            variable_indices.remove(i)
        else: 
            indices.append(0)
            variable_ranges.append(range(array_to_print.shape[i]))
    for i in variable_indices:
        return_dimensions.append(array_to_print.shape[i])
    the_product = product(*variable_ranges)
    to_return = np.zeros(return_dimensions,dtype=np.float64)
    for i in list(the_product):
        cnt = 0
        for j in variable_indices:
            indices[j] = i[cnt]
            cnt = cnt + 1
        to_return[i]=function_to_apply(array_to_print[indices])
        
    return to_return
        
def plot_array(array,axisx,axisy,title):
    fig = plt.figure()
    fig.hold(True)
    for k in range(0,array.shape[0]):
        plt.plot(array[k,:], linewidth=3.0)
    plt.legend(range(1,array.shape[0]+1))
    create_axes_title(axisx,axisy,title,fig)
    fig.savefig(title+'.pdf')
    fig.hold(False)
    
    
def plot_lambda(lambdap,iteration):
    # update lambda only
    K = lambdap.shape[1]
    fig = plt.figure()
    plt.hold(True)
    for k in range(K):
        plt.subplot(K,1,k)
        plt.plot(lambdap[iteration,k,:].T)
    #plt.legend(range(0,lambdap.shape[0],10))
    create_axes_title('aligned read locations','lambda','lambda value for iteration ' + `iteration` + ' and isoforms ' + `range(K)`,fig)
    fig.savefig('lambda_iteration_'+str(iteration)+'.pdf')
    plt.hold(False)
    
#     # update mu only
#     plt.plot(((lambdap[iteration,:,:]>0)*np.arange(1,K+1)[:,None]).T,'o')
#     plt.ylim(0,K+1)
#     create_axes_title('aligned read locations','isoforms','lambda value for iteration ' + `iteration` + ' and isoforms ' + `range(K)`,fig)
#     plt.show()


def create_axes_title(x_axis_string, y_axis_string, title, fig):
    #fig.text(0.5,0.95,'true read counts by isoform for each individual', ha='center', va='center', size='x-large')
    fig.suptitle(title, fontsize=14, fontweight='bold')
    fig.text(0.5, 0.06, x_axis_string, ha='center', va='center', size='large') 
    fig.text(0.1, 0.5, y_axis_string, ha='center', va='center', rotation='vertical', size='large')  
    
def compute_phi_zeta_prob(M,N,K,phi0,zeta0):
    P_zjik = np.zeros((M,max(N),K),dtype=np.float32)
    for j in range(M):
        for i in range(N[j]):
            for k in range(K):
                P_zjik[j,i,k] = np.sum(phi0[j,i,:]*zeta0[j,:,k])
    return P_zjik