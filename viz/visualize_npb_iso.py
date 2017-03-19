# !/tigress/lifangc/anaconda/bin/python
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import numpy as np
import os
from scipy.special import psi
import pickle, gzip

EXON_COMP_THR = 0.4

def visualize_iso_results(\
                            result_file                 , \
                            output_dir                  , \
                            exon_len_file               , \
                            flag_plot_all_iter=0        , \
                            flag_compute_alt=0          , \
                            flag_skip_plotting=0        , \
                            flag_discrete_read_map=0    , \
                            flag_save_html=False        \
                            ):
    iter_format = '%04d'

    # data = np.load(result_file)
    print 'loading result file: ' + result_file
    if result_file[-4:]=='.pkz':  # zipped pickle
        (mu_final,lambda_final,a1_final,a2_final,phi_final,zeta_final,gamma1_final,gamma2_final) = pickle.load(gzip.open(result_file, 'rb'))
    elif result_file[-4:]=='.pkl': # pickle
        (mu_final,lambda_final,a1_final,a2_final,phi_final,zeta_final,gamma1_final,gamma2_final) = pickle.load(file.open(result_file, 'rb'))
    elif result_file[-4:]=='.npz': # compressed numpy (last it)
        data=np.load(result_file)
        mu_final=[]
        mu_final.append(data['mu'])
        lambda_final=[]
        lambda_final.append(data['lambdap'])
        a1_final=[]
        a1_final.append(data['a1'])
        a2_final=[]
        a2_final.append(data['a2'])
        phi_final=[]
        phi_final.append(data['phi'])
        zeta_final=[]
        zeta_final.append(data['zeta'])
        gamma1_final=[]
        gamma1_final.append(data['gamma1'])
        gamma2_final=[]
        gamma2_final.append(data['gamma2'])
        
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    print 'output directory for figures: ' + output_dir

    iter_num        = len(mu_final)
    exon_num        = mu_final[-1].shape[1]
    init_iso_num    = mu_final[-1].shape[0]
    idv_num         = len(zeta_final[-1])
    read_num        = len(lambda_final[-1][0])
    # iter_num        = data['zeta'].shape[0]
    # idv_num         = data['zeta'].shape[1] # M, index by j
    # init_iso_num    = data['mu'].shape[1] # K, index by k
    # exon_num        = data['mu'].shape[2] # E, index by e
    # read_num        = data['lambdap'].shape[2] # V, index by v

    # access the last iteration to get final results
    t_final     = compute_t(init_iso_num, a1_final[-1], a2_final[-1])
    c_final     = compute_c(idv_num, zeta_final[-1])
    z_final     = compute_z(idv_num, phi_final[-1])
    psi_final   = compute_psi(idv_num, init_iso_num, gamma1_final[-1], gamma2_final[-1])
    b_final     = compute_b(init_iso_num, exon_num, mu_final[-1])
    # t_final     = compute_t(init_iso_num, data['a1'][iter_num-1, :], data['a2'][iter_num-1, :])
    # c_final     = compute_c(idv_num, data['zeta'][iter_num-1, :, :, :])
    # z_final     = compute_z(idv_num, data['phi'][iter_num-1, :])
    # psi_final   = compute_psi(idv_num, init_iso_num, data['gamma1'][iter_num-1, :], data['gamma2'][iter_num-1, :])
    # pi_final    = compute_pi(init_iso_num, data['rho1'][iter_num-1, :], data['rho2'][iter_num-1, :])
    # b_final     = compute_b(init_iso_num, exon_num, data['mu'][iter_num-1, :, :])
    # # beta_final   = compute_beta(iso_num=iso_num, read_num=read_num, lam=data['lambdap'][iter_num-1, :, :])
    # np.savez(output_dir+'param_final.npz', t=t_final,z=z_final,c=c_final,psi=psi_final,pi=pi_final,b=b_final)

    # merge duplicated isoforms
    # merge_iso_tab: init_iso_num x 1
    # new_iso_tab: new iso_num x 1
    new_iso_num, new_iso_tab, merge_iso_tab, merge_iso_str_tab = merge_iso(init_iso_num, exon_num, b_final)
    ind_read_iso_mappings = np.einsum('jil,jlk->jik',phi_final[-1],zeta_final[-1])
    ind_read_iso_mappings_merged = np.zeros((idv_num,ind_read_iso_mappings.shape[1],new_iso_num),dtype=ind_read_iso_mappings.dtype)
    for k in range(init_iso_num):
        ind_read_iso_mappings_merged[:,:,merge_iso_tab[k]]+=ind_read_iso_mappings[:,:,k]

    # print merge_iso_tab, merge_iso_str_tab
    # print new_iso_num, new_iso_tab
    new_b_final = merge_b(new_iso_num, exon_num, b_final, merge_iso_tab)

    # plot the last iteration
    prefix = 'iter_' + (iter_format % (iter_num-1)) + '_'

    # compute global & individual proportions
    idv_read_mapping = compute_read_mapping(idv_num, new_iso_num, exon_num, z_final, c_final, merge_iso_tab)
    # print idv_read_mapping
    global_iso_prop, idv_iso_prop = compute_iso_prop(idv_num, init_iso_num, new_iso_num, exon_num, merge_iso_tab, c_final, t_final, psi_final, flag_compute_alt)
    
    # output results of global & individual proportions
    f = open(exon_len_file, 'rb')
    exon_len = np.int_(f.readlines())
    assert exon_len.shape[0] == exon_num
    # print exon_len
    print_isoform_results(iter_num, idv_num, init_iso_num, new_iso_num, exon_num, new_b_final, global_iso_prop, idv_iso_prop, idv_read_mapping, exon_len, ind_read_iso_mappings_merged, flag_discrete_read_map)
    output_isoform_results(iter_num, idv_num, init_iso_num, new_iso_num, exon_num, new_b_final, global_iso_prop, idv_iso_prop, idv_read_mapping, exon_len, prefix, output_dir, ind_read_iso_mappings_merged, flag_discrete_read_map)    
            
    if not flag_skip_plotting:
        # # decide global colormap for exons and isoforms:
        # index 0:(exon_num-1) - exon 1 ~ exon E
        # index exon_num:(exon_num+iso_num-1) - isoform 1 ~ isoform K
        global_color = plt.cm.gnuplot(np.linspace(0,1,exon_num+new_iso_num))
        exon_color = global_color[0:exon_num, :]
        iso_color = global_color[exon_num:(exon_num+new_iso_num), :]

        prefix = 'final_iter_' + (iter_format % (iter_num-1)) + '_'
        
        # plot the last iteration
        # plot exon composition of isoforms    
        plot_isoform2exon(new_iso_num, exon_num, new_b_final, exon_color, prefix, output_dir)
        plot_isoform2exon2(new_iso_num, exon_num, new_b_final, exon_color, prefix, output_dir)
        plot_top_isoform2exon2(new_iso_num, exon_num, new_b_final, global_iso_prop, exon_color, prefix, output_dir)

        # plot the results of global & individual proportions
        plot_global_isoform_dist(new_iso_num, global_iso_prop, iso_color, iter_format, prefix, output_dir)
        plot_all_isoform_dist(idv_num, new_iso_num, exon_num, global_iso_prop, idv_iso_prop, iso_color, prefix, output_dir)
        for j in range(idv_num):
            plot_idv_isoform_dist(j, new_iso_num, idv_iso_prop, iso_color, iter_format, prefix, output_dir)

        # output one html file to view all results
        html_title = 'Final Results'
        sec_title = [   'exon composition (view 1)', \
                        'exon composition (view 2)', \
                        'exon composition for top isoforms (view 2)', \
                        'isoform distribution (global and individual)', \
                        'isoform distribution (global)' ]
        sec_prefix = [  prefix + 'isoform2exon' , \
                        prefix + 'isoform2exon2', \
                        prefix + 'top_isoform2exon2', \
                        prefix + 'isoform_dist_all', \
                        prefix + 'isoform_dist_global', \
                         ]
        sec_iter = []
        for x in range(len(sec_prefix)):
            sec_iter.append([])
        for j in range(idv_num):
            sec_title.append('isoform distribution for individual ' + str(j))
            sec_prefix.append((prefix + 'isoform_dist_idv_' + (iter_format % j)))
            sec_iter.append([])
        # for j in range(idv_num):
        #     sec_title.append('read-to-isoform distribution for individual ' + str(j))
        #     sec_prefix.append((prefix + 'read2isoform_dist_idv_' + (iter_format % j)))
        #     sec_iter.append([])
        output_html(html_title, sec_title, sec_prefix, sec_iter, iter_format, 'png', '', './', output_dir)
    
        # if we want to plot multiple iteration
        if(flag_plot_all_iter == 1):
            for iter_idx in range(0, iter_num):
                curr_iso_num    = mu_final[iter_idx].shape[0]

                a1      = a1_final[iter_idx]
                a2      = a2_final[iter_idx]
                phi     = phi_final[iter_idx]
                zeta    = zeta_final[iter_idx]
                gamma1  = gamma1_final[iter_idx]
                gamma2  = gamma2_final[iter_idx]
                mu      = mu_final[iter_idx]

                curr_t      = compute_t(curr_iso_num, a1, a2)
                curr_c      = compute_c(idv_num, zeta)
                curr_z      = compute_z(idv_num, phi)
                curr_psi    = compute_psi(idv_num, curr_iso_num, gamma1, gamma2)
                curr_b      = compute_b(curr_iso_num, exon_num, mu)

                curr_new_iso_num, curr_new_iso_tab, curr_merge_iso_tab, curr_merge_iso_str_tab = merge_iso(curr_iso_num, exon_num, curr_b)
                # print merge_iso_tab, merge_iso_str_tab
                # print new_iso_num, new_iso_tab
                curr_new_b = merge_b(curr_new_iso_num, exon_num, curr_b, curr_merge_iso_tab)

                global_color = plt.cm.gnuplot(np.linspace(0,1,exon_num+curr_new_iso_num))
                exon_color = global_color[0:exon_num, :]
                iso_color = global_color[exon_num:(exon_num+curr_new_iso_num), :]

                prefix = 'iter_' + (iter_format % iter_idx) + '_'

                plot_isoform2exon(curr_new_iso_num, exon_num, curr_new_b, exon_color, prefix, output_dir)
                plot_isoform2exon2(curr_new_iso_num, exon_num, curr_new_b, exon_color, prefix, output_dir)
                
                curr_idv_read_mapping = compute_read_mapping(idv_num, curr_new_iso_num, exon_num, curr_z, curr_c, curr_merge_iso_tab)
                # print idv_read_mapping

                # compute global & individual proportions
                curr_global_iso_prop, curr_idv_iso_prop = compute_iso_prop(idv_num, curr_iso_num, curr_new_iso_num, exon_num, curr_merge_iso_tab, curr_c, curr_t, curr_psi, flag_compute_alt)
                plot_top_isoform2exon2(curr_new_iso_num, exon_num, curr_new_b, curr_global_iso_prop, exon_color, prefix, output_dir)

                # output results of global & individual proportions
                print '@ iteration : ' + str(iter_idx)
                print_isoform_results(iter_num, idv_num, curr_iso_num, curr_new_iso_num, exon_num, curr_new_b, curr_global_iso_prop, curr_idv_iso_prop, idv_read_mapping, exon_len)
            
                # plot the results of global & individual proportions
                plot_global_isoform_dist(curr_new_iso_num, curr_global_iso_prop, iso_color, iter_format, prefix, output_dir)
                plot_all_isoform_dist(idv_num, curr_new_iso_num, exon_num, curr_global_iso_prop, curr_idv_iso_prop, iso_color, prefix, output_dir)
                for j in range(idv_num):
                    plot_idv_isoform_dist(j, curr_new_iso_num, curr_idv_iso_prop, iso_color, iter_format, prefix, output_dir)
                    
            # output one html file to view changes of all iterations for each item
            html_title_list = [ 'exon composition (view 1)', \
                                'exon composition (view 2)', \
                                'exon composition for top isoforms (view 2)', \
                                'isoform distribution (global and individual)', \
                                'isoform distribution (global)', \
                                'individual isoform distribution' ]
            title_prefix = [    'isoform2exon' , \
                                'isoform2exon2', \
                                'top_isoform2exon2', \
                                'isoform_dist_all', \
                                'isoform_dist_global', \
                                'isoform_dist_idv' ]
            for hidx in range(len(html_title_list)):
                sec_title = []
                sec_prefix = []
                sec_iter = []
                for iter_idx in range(iter_num):
                    sec_title.append('iteration ' + str(iter_idx))
                    sec_prefix.append('iter_' + (iter_format % iter_idx) + '_' + title_prefix[hidx])
                    if(hidx > 4):
                        sec_iter.append(range(idv_num))
                    else:
                        sec_iter.append([])
                output_html(html_title_list[hidx], sec_title, sec_prefix, sec_iter, iter_format, 'png', (title_prefix[hidx] + '_'), './', output_dir)            


def compute_z(idv_num, phi):
    assert len(phi) == idv_num
    z = []
    for j in range(idv_num):
        # identify how many valid reads
        valid_read = 0
        for i in range(phi[j].shape[0]):
            if(np.sum(phi[j][i, :]) == 0):
                pass
            else:
                valid_read += 1
        # print valid_read
        # record the index of the most likely individual distribution
        z.append(np.argmax(phi[j], axis=1)[0:valid_read])
    return z

def compute_c(idv_num, zeta):
    assert len(zeta) == idv_num
    # record all distributions first
    c = []
    for j in range(idv_num):
        # record the index of the most likely individual distribution
        # print 'checking c: individual ' + str(j)
        # print zeta[j]
        # print np.argmax(zeta[j], axis=1)
        c.append(np.argmax(zeta[j], axis=1)) 
    return c

def compute_psi(idv_num, iso_num, gamma1, gamma2):
    all_psi = np.zeros((idv_num, iso_num))
    assert len(gamma1) == idv_num
    assert len(gamma2) == idv_num
#     assert len(gamma1[0]) == iso_num
#     assert len(gamma2[0]) == iso_num

    for j in range(0, idv_num):
        for l in range(0, len(gamma1[j])):
            if gamma1[j][l]>0:
                e1 = psi(gamma1[j][l]) - psi(gamma1[j][l] + gamma2[j][l])
                e2 = 0.0
                for x in range(0, l):
                    if gamma1[j][x]>0:
                        e2 += psi(gamma2[j][x]) - psi(gamma1[j][x] + gamma2[j][x])
                all_psi[j, l] = np.exp(e1 + e2)
        all_psi[j, :] = all_psi[j, :]/np.sum(all_psi[j, :])
    return all_psi


def compute_t(iso_num, a1, a2):
    t = np.zeros((iso_num))
    assert t.shape[0] == a1.shape[0]
    assert t.shape[0] == a2.shape[0]

    for k in range(iso_num):
        if a1[k]==0:
            continue
        e1 = psi(a1[k]) - psi(a1[k]+a2[k])
        e2 = 0.0
        for x in range(0, k):
            if a1[x]==0:
                continue
            e2 += psi(a2[x]) - psi(a1[x] + a2[x])
        t[k] = np.exp(e1 + e2)
    # normalize to have sume to one (?)
    t = t/np.sum(t)
    return t


def compute_pi(iso_num, rho1, rho2):
    all_pi = np.zeros((iso_num))
    assert rho1.shape[0] == iso_num
    assert rho2.shape[0] == iso_num
    for k in range(iso_num):
        all_pi[k] = np.exp(psi(rho1[k]) - psi(rho1[k] + rho2[k]))
    return all_pi

def compute_b(iso_num, exon_num, mu):
    assert mu.shape == (iso_num, exon_num)
    return mu

# merge k (isoform index)
def merge_iso(init_iso_num, exon_num, b):
    assert b.shape == (init_iso_num, exon_num)
    new_iso_num = 0
    new_iso_tab = []
    merge_iso_str_tab = []
    merge_iso_tab = np.zeros((init_iso_num,))

    exon_comp_thr = EXON_COMP_THR
    for k in range(init_iso_num):
        exon_comp = b[k, :]
        valid_exon_idx = np.where(b[k, :] > exon_comp_thr)[0]
        exon_str = ''
        for e in valid_exon_idx:
            exon_str += str(e) + ','
        exon_str = exon_str[0:-1]
        if exon_str not in merge_iso_str_tab:
            new_iso_tab.append(k)
            merge_iso_str_tab.append(exon_str)
            merge_iso_tab[k] = new_iso_num
            new_iso_num += 1
        else:
            merge_iso_tab[k] = merge_iso_str_tab.index(exon_str)

    new_iso_tab = np.asarray(new_iso_tab)
    merge_iso_tab = np.asarray(merge_iso_tab,dtype=np.int)
    merge_iso_str_tab = np.asarray(merge_iso_str_tab)
    return new_iso_num, new_iso_tab, merge_iso_tab, merge_iso_str_tab


def merge_b(new_iso_num, exon_num, b, merge_iso_tab):
    new_b = np.zeros((new_iso_num, exon_num))
    for k in range(new_iso_num):
        new_b[k, :] = np.mean(b[np.where(merge_iso_tab == k)[0], :], axis=0)
    return new_b


def compute_read_mapping(idv_num, new_iso_num, exon_num, z, c, merge_iso_tab):
    new_z = list(z)
    for j in range(idv_num):
        new_z[j] = merge_iso_tab[ c[j][ z[j] ] ]
        # print c[j], z[j], new_z[j]
    return new_z


def compute_beta(iso_num, read_num, lam):
    assert lam.shape == (iso_num, read_num)
    beta = np.zeros((iso_num, read_num))
    for k in range(iso_num):
        for v in range(read_num):
            beta[k, v] = psi(lam[k, v]) - np.sum(psi(lam[k, :]))
        beta[k, v] = np.exp(beta[k, v])
    return beta


def compute_iso_prop(idv_num, init_iso_num, new_iso_num, exon_num, merge_iso_tab, c, t, psi, flag_compute_alt):
    if(flag_compute_alt == 0):
        global_iso_prop = np.zeros((new_iso_num, ))
        for k in range(init_iso_num):
            global_iso_prop[merge_iso_tab[k]] += t[k]

        idv_iso_prop = np.asarray([])
        for j in range(idv_num):
            curr_idv_iso_prop = np.zeros((1, new_iso_num))
            for k in range(init_iso_num):
                for l in range(len(c[j])):
                    if(c[j][l] == k):
                        curr_idv_iso_prop[0, merge_iso_tab[k]] += psi[j, l]
            if(j == 0):
                idv_iso_prop = curr_idv_iso_prop
            else:
                idv_iso_prop = np.vstack((idv_iso_prop, curr_idv_iso_prop))

    elif(flag_compute_alt == 1):
        idv_iso_prop = np.asarray([])
        for j in range(idv_num):
            curr_idv_iso_prop = np.zeros((1, new_iso_num))
            for k in range(init_iso_num):
                for l in range(init_iso_num):
                    if(c[j][l] == k):
                        curr_idv_iso_prop[0, merge_iso_tab[k]] += psi[j, l]
            if(j == 0):
                idv_iso_prop = curr_idv_iso_prop
            else:
                idv_iso_prop = np.vstack((idv_iso_prop, curr_idv_iso_prop))
        
        global_iso_prop = np.zeros((new_iso_num, ))
        for k in range(init_iso_num):
            global_iso_prop[merge_iso_tab[k]] += t[k]
        global_iso_prop = global_iso_prop*np.sum(idv_iso_prop, axis=0)
        global_iso_prop = global_iso_prop/np.sum(global_iso_prop)

    else:
        raise NotImplementedError

    return global_iso_prop, idv_iso_prop


def print_isoform_results(iter_num, idv_num, init_iso_num, new_iso_num, exon_num, b, global_iso_prop, idv_iso_prop, idv_read_mapping, exon_len, ind_read_iso_mappings, flag_discrete_read_map):
    read_count = np.zeros((idv_num, new_iso_num))
    
    if(flag_discrete_read_map):
        for j in range(idv_num):
            for k in range(new_iso_num):
                read_count[j, k] = len(np.where(idv_read_mapping[j] == k)[0])
    else:        
        read_count=np.sum(ind_read_iso_mappings,axis=1)

    print '-----------------------------------------------------------------------------'
    print '# of iterations: ' + str(iter_num)
    print '# of individuals: ' + str(idv_num)
    print '# of exons: ' + str(exon_num)
    print '# of isoforms (before merging): ' + str(init_iso_num)
    print '# of isoforms (after merging): ' + str(new_iso_num)
    print '-----------------------------------------------------------------------------'

    exon_comp_thr = EXON_COMP_THR

    sorted_iso_idx = np.argsort(global_iso_prop)[::-1]
    for k in sorted_iso_idx:
        if global_iso_prop[k]>0:
            exon_comp = b[k, :]
            valid_exon_idx = np.where(b[k, :] > exon_comp_thr)[0]
            exon_str = ''
            for e in valid_exon_idx:
                exon_str += str(e) + ','
            exon_str = exon_str[0:-1]
            rpkm = compute_rpkm(np.sum(read_count[:, k]), np.sum(read_count), b[k, :], exon_len)
            print 'isoform ' + str(k) + ': (' + ('%3.2f' % (100.0*global_iso_prop[k])) + '%): ' + exon_str + '\t' + 'RPKM = ' + ('%.4f' % rpkm)
    print '-----------------------------------------------------------------------------'
    
    # print isoforms and proportions for each individual
    assert idv_iso_prop.shape[0] == idv_num
    for j in range(idv_num):
        print 'isoform distribution for individual ' + str(j)
        sorted_idv_iso_idx = np.argsort(idv_iso_prop[j, :])[::-1]
        for k in sorted_idv_iso_idx:
            if(idv_iso_prop[j, k] > 0):
                exon_comp = b[k, :]
                valid_exon_idx = np.where(b[k, :] > exon_comp_thr)[0]
                exon_str = ''
                for e in valid_exon_idx:
                    exon_str += str(e) + ','
                exon_str = exon_str[0:-1]
                rpkm = compute_rpkm(read_count[j, k], np.sum(read_count[j, :]), b[k, :], exon_len)
                print 'isoform ' + str(k) + ': (' + ('%3.2f' % (100.0*idv_iso_prop[j, k])) + '%): ' + exon_str + '\t' + 'RPKM = ' + ('%.4f' % rpkm)
    print '-----------------------------------------------------------------------------'

def compute_rpkm(read_count, total_read_count, iso_b, exon_len):
    exon_comp_thr = EXON_COMP_THR
    norm_count = total_read_count/np.power(10., 6.)
    iso_len = 0.
    valid_exon_idx = np.where(iso_b > exon_comp_thr)[0]
    if(len(valid_exon_idx) > 0):
        iso_len = np.sum(exon_len[valid_exon_idx])
    iso_len = iso_len/1000.
    if norm_count==0 or iso_len==0:
        return 0
    else:
        rpkm = read_count/(norm_count * iso_len)
        return rpkm


def output_isoform_results(iter_num, idv_num, init_iso_num, new_iso_num, exon_num, b, global_iso_prop, idv_iso_prop, idv_read_mapping, exon_len, prefix, output_dir, ind_read_iso_mappings, flag_discrete_read_map):
    filename = output_dir + 'results.txt'
    # print 'output results to tab csv file: ' + filename
    read_count = np.zeros((idv_num, new_iso_num))
    if(flag_discrete_read_map):
        for j in range(idv_num):
            for k in range(new_iso_num):
                read_count[j, k] = len(np.where(idv_read_mapping[j] == k)[0])
    # print read_count
    else:
        read_count=np.sum(ind_read_iso_mappings,axis=1)

    f = open(filename, 'wb')
    f.write('type')
    f.write('\t\t')
    f.write('isoform')
    f.write('\t\t')
    f.write('proportions')
    f.write('\t\t')
    f.write('exons')
    f.write('\t\t')
    f.write('RPKM/FPKM')
    f.write('\n')

    exon_comp_thr = EXON_COMP_THR

    # print global isoforms and proportions
    sorted_iso_idx = np.argsort(global_iso_prop)[::-1]
    for k in sorted_iso_idx:
        exon_comp = b[k, :]
        valid_exon_idx = np.where(b[k, :] > exon_comp_thr)[0]
        exon_str = ''
        for e in valid_exon_idx:
            exon_str += str(e) + ','
        exon_str = exon_str[0:-1]
        f.write('global')
        f.write('\t\t')
        f.write(str(k))
        f.write('\t\t')
        f.write(('%3.4f' % (100.0*global_iso_prop[k])))
        f.write('\t\t')
        f.write(exon_str)
        f.write('\t\t')
        rpkm = compute_rpkm(np.sum(read_count[:, k]), np.sum(read_count), b[k, :], exon_len)
        f.write(('%.4f' % rpkm))
        f.write('\n')
    
    # print isoforms and proportions for each individual
    for j in range(idv_num):
        sorted_idv_iso_idx = np.argsort(idv_iso_prop[j, :])[::-1]
        for k in sorted_idv_iso_idx:
            if(idv_iso_prop[j, k] > 0):
                exon_comp = b[k, :]
                valid_exon_idx = np.where(b[k, :] > exon_comp_thr)[0]
                exon_str = ''
                for e in valid_exon_idx:
                    exon_str += str(e) + ','
                exon_str = exon_str[0:-1]
                f.write('idv' + str(j))
                f.write('\t\t')
                f.write(str(k))
                f.write('\t\t')
                f.write(('%3.4f' % (100.0*idv_iso_prop[j, k])))
                f.write('\t\t')
                f.write(exon_str)
                f.write('\t\t')
                rpkm = compute_rpkm(read_count[j, k], np.sum(read_count[j, :]), b[k, :], exon_len)
                f.write(('%.4f' % rpkm))
                f.write('\n')
    f.close()


def plot_isoform2exon(iso_num, exon_num, b, exon_color, prefix, output_dir):
    bar_height = 0.35
    ind = range(iso_num)

    fig_width = 20.0
    fig_height = min(2.0 + 2.0*(iso_num + 1), 40)
    plt.figure(figsize=(fig_width, fig_height))
    for e in range(exon_num):
        plt.barh(ind, b[0:iso_num, e], height=bar_height, align='center', left=np.sum(b[0:iso_num, 0:e], axis=1), color=exon_color[e], label=('exon ' + str(e)))
    
    iso_tick = []
    for k in range(iso_num):
        iso_tick.append(('isoform ' + str(k)))
    plt.yticks(ind, iso_tick, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim((0.0, exon_num+0.3))
    plt.xlabel('exon number', fontsize=16)
    plt.title('Global isoform to exon composition', fontsize=20)
    # plt.legend()
    lgd = plt.legend(fontsize=14, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    # plt.savefig(output_dir + prefix + 'isoform2exon' + '.eps', format='eps')
    plt.savefig(output_dir + prefix + 'isoform2exon' + '.png', format='png', bbox_extra_artists=(lgd,))
    plt.close()


def plot_isoform2exon2(iso_num, exon_num, b, exon_color, prefix, output_dir):
    bar_height = 0.35
    ind = range(iso_num)

    fig_width = 20.0
    fig_height = min(2.0 + 2.0*(iso_num + 1), 40)
    plt.figure(figsize=(fig_width, fig_height))
    for e in range(exon_num):
        plt.barh(ind, b[0:iso_num, e], height=bar_height, align='center', left=e*np.ones((iso_num)), color=exon_color[e], label=('exon ' + str(e)))
    
    iso_tick = []
    for k in range(iso_num):
        iso_tick.append(('isoform ' + str(k)))
    plt.yticks(ind, iso_tick, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim((0.0, exon_num+0.3))
    plt.xlabel('exon index', fontsize=16)
    plt.title('Global isoform to exon composition', fontsize=20)
    # plt.legend()
    lgd = plt.legend(fontsize=14, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    # plt.savefig(output_dir + prefix + 'isoform2exon2' + '.eps', format='eps')
    plt.savefig(output_dir + prefix + 'isoform2exon2' + '.png', format='png', bbox_extra_artists=(lgd,))
    plt.close()    


def plot_top_isoform2exon2(iso_num, exon_num, b, global_iso_prop, exon_color, prefix, output_dir):
    plot_iso_num = len(np.where(global_iso_prop > 0)[0])
    plot_iso_num = min(5, plot_iso_num)

    dist_all = np.atleast_2d(global_iso_prop)
    top_iso_idx = np.argsort(global_iso_prop)[::-1]
    bar_height = 0.35
    ind = range(plot_iso_num)

    fig_width = 20.0
    fig_height = min(2.0 + 2.0*(plot_iso_num + 1), 40)
    plt.figure(figsize=(fig_width, fig_height))
    for e in range(exon_num):
        plt.barh(ind, b[top_iso_idx[0:plot_iso_num], e], height=bar_height, align='center', left=e*np.ones((plot_iso_num)), color=exon_color[e], label=('exon ' + str(e)))
    
    iso_tick = []
    for k in range(plot_iso_num):
        # print 'top ' + str(k) + ' isoform k = ' + str(top_iso_idx[k]) + ' proportion = ' + str(t[top_iso_idx[k]])
        # print 'exon composition = ' + repr(b[top_iso_idx[k], :])
        iso_tick.append(('isoform ' + str(top_iso_idx[k]) + ': ' + ('%2.2f' % global_iso_prop[top_iso_idx[k]])))

    plt.yticks(ind, iso_tick, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim((0.0, exon_num+0.3))
    plt.xlabel('exon index', fontsize=16)
    plt.title('Global isoform to exon composition for top ' + str(plot_iso_num) + ' isoforms', fontsize=20)
    # plt.legend()
    lgd = plt.legend(fontsize=14, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    # plt.savefig(output_dir + prefix + 'top_isoform2exon2' + '.eps', format='eps')
    plt.savefig(output_dir + prefix + 'top_isoform2exon2' + '.png', format='png', bbox_extra_artists=(lgd,))
    plt.close()    


def plot_all_isoform_dist(idv_num, iso_num, exon_num, global_iso_prop, idv_iso_prop, iso_color, prefix, output_dir):
    dist_all = np.atleast_2d(global_iso_prop)
    dist_all = np.vstack((dist_all, idv_iso_prop))

    bar_height = 0.35
    ind = range(idv_num + 1)

    fig_width = 20.0
    fig_height = min(20.0 + 1.0*(idv_num + 1), 40)
    plt.figure(figsize=(fig_width, fig_height))
    for k in range(iso_num):
        plt.barh(ind, dist_all[:, k], bar_height, align='center', left=np.sum(dist_all[:, 0:k], axis=1), color=iso_color[k], label=('isoform ' + str(k)))
    
    idv_tick = ['Global']
    for j in range(idv_num):
        idv_tick.append(('subject ' + str(j)))
    plt.yticks(ind, idv_tick, fontsize=14)
    plt.xticks(fontsize=14)
    plt.xlim((0.0, 1.2))
    plt.xlabel('isoform proportion', fontsize=16)
    plt.title('Isoform distribution (tk and tk*cjl)', fontsize=20)
    # plt.legend()
    lgd = plt.legend(fontsize=14, bbox_to_anchor=(1.02, 1), loc=2, borderaxespad=0.)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)   
    # plt.savefig(output_dir + prefix + 'isoform_dist_all' + '.eps', format='eps')
    plt.savefig(output_dir + prefix + 'isoform_dist_all' + '.png', format='png', bbox_extra_artists=(lgd,))
    plt.close()       


def plot_global_isoform_dist(iso_num, global_iso_prop, iso_color, iter_format, prefix, output_dir):
    fig_width = min(5.0*(iso_num), 80)
    fig_height = 30.0
    bar_width = 0.35

    iso_tick = []
    for k in range(iso_num):
        iso_tick.append(('isoform ' + str(k)))

    plt.figure(figsize=(fig_width, fig_height))
    for k in range(iso_num):
        plt.bar([k], global_iso_prop[k], bar_width, align='center', color=iso_color[k], label=('Isoform ' + str(k)))
    plt.xticks(range(iso_num), iso_tick, fontsize=16)
    plt.ylim((0.0, 1.0))
    plt.title('Global distribution over k isoforms')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # plt.savefig(output_dir + prefix + 'isoform_dist_idv_' + (iter_format % idv_idx) + '.eps', format='eps')
    plt.savefig(output_dir + prefix + 'isoform_dist_global.png', format='png')
    plt.close() 


def plot_idv_isoform_dist(idv_idx, iso_num, idv_iso_prop, iso_color, iter_format, prefix, output_dir):
    # num_col = 3
    # num_row = np.ceil(c_idv.shape[0]/np.float(num_col)) 
    fig_width = min(5.0*(iso_num), 80)
    fig_height = 30.0
    bar_width = 0.35

    iso_tick = []
    for k in range(iso_num):
        iso_tick.append(('isoform ' + str(k)))

    plt.figure(figsize=(fig_width, fig_height))
    for k in range(iso_num):
        plt.bar([k], idv_iso_prop[idv_idx, k], bar_width, align='center', color=iso_color[k], label=('Isoform ' + str(k)))
    plt.xticks(range(iso_num), iso_tick, fontsize=16)
    plt.ylim((0.0, 1.0))
    # if(l == np.argmax(psi)):
    #     plt.title('Individual ' + str(idv_idx) + ' distribution (cjl), l = ' + str(l), color='r')
    # else:
    plt.title('Individual ' + str(idv_idx) + ' distribution over k isoforms')

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # plt.savefig(output_dir + prefix + 'isoform_dist_idv_' + (iter_format % idv_idx) + '.eps', format='eps')
    plt.savefig(output_dir + prefix + 'isoform_dist_idv_' + (iter_format % idv_idx) + '.png', format='png')
    plt.close()   


# def plot_idv_read2isoform_dist(idv_idx, iso_num, z, c_idv, iso_color, iter_format, prefix, output_dir):
#     read_num = len(z)
#     read_tick = []
#     for i in range(read_num):
#         read_tick.append(('read ' + str(i)))

#     # dist_all = np.atleast_2d(np.asarray([]))
#     for i in range(read_num):
#         read_dist = c_idv[z[i], :]
#         read_dist = read_dist/sum(read_dist)
#         if(i == 0):
#             dist_all = np.atleast_2d(read_dist)
#         else:
#             dist_all = np.vstack((dist_all, read_dist))

#     bar_width = 0.35
#     ind = range(read_num)

#     fig_width = 2.0 + 1.0*(read_num)
#     fig_height = 8.0
#     plt.figure(figsize=(fig_width, fig_height))
#     for k in range(iso_num):
#         plt.bar(ind, dist_all[:, k], bar_width, align='center', bottom=np.sum(dist_all[:, 0:k], axis=1), color=iso_color[k], label=('isoform ' + str(k)))
#     plt.title('Individual ' + str(idv_idx) + ' read-to-isoform distribution (zij)')
#     plt.xticks(ind, read_tick, fontsize=14)
#     plt.ylim((0.0, 1.2))
#     plt.legend()
#     if not os.path.exists(output_dir):
#         os.makedirs(output_dir)
#     # plt.savefig(output_dir + prefix + 'read2isoform_dist_idv_' + (iter_format % idv_idx) + '.eps', format='eps')
#     plt.savefig(output_dir + prefix + 'read2isoform_dist_idv_' + (iter_format % idv_idx) + '.png', format='png')
#     plt.close() 


def output_html(html_title, sec_title, sec_prefix, sec_iter, iter_format, fig_type, prefix, input_dir, output_dir):
    pixel_num = 256
    fig_num_per_row = 5

    output_file = open(output_dir + prefix + 'summary.html', 'w')
    # header and title
    output_file.write('<html>\n')
    output_file.write('\n')
    output_file.write('<head>\n')
    output_file.write('\t<title>\n')
    output_file.write('\t\t' + html_title + '\n')
    output_file.write('\t</title>\n')
    output_file.write('</head>\n\n')
    output_file.write('<body bgcolor="#ffffff">\n\n')

    output_file.write('<center>\n')
    output_file.write('<h2>\n')
    output_file.write('\t' + html_title + '\n')
    output_file.write('</h2>\n')
    output_file.write('</center>\n\n')

    # iterate through sections and 
    for sidx in range(len(sec_title)):
        output_file.write('<h3>' + sec_title[sidx] + '</h3>\n\n')
        output_file.write('<table>\n')

        if(len(sec_iter[sidx]) > 0):
            count = 0
            while(count < len(sec_iter[sidx])):
                output_file.write('<table>\n')
                output_file.write('\t<tr>\n')
                for f in range(fig_num_per_row):
                    curr_fig_index = sec_iter[sidx][count]
                    
                    output_file.write('\t\t%s' % ('<td><a href="' + input_dir))
                    output_file.write('%s' % (sec_prefix[sidx] + '_'))
                    output_file.write((iter_format % curr_fig_index))
                    output_file.write('%s' % ('.' + fig_type))
                    output_file.write('">\t\t')
                    
                    output_file.write('%s' % ('<img height=' + str(pixel_num) + ' src="' + input_dir))
                    output_file.write('%s' % (sec_prefix[sidx] + '_'))
                    output_file.write((iter_format % curr_fig_index))
                    output_file.write('%s' % ('.' + fig_type))
                    output_file.write('"></a></td>\n')

                    count = count + 1
                    if(count == (len(sec_iter[sidx]))):
                        break
                output_file.write('\t</tr>\n')
                output_file.write('</table>\n')
        else:
            output_file.write('<table>\n')
            output_file.write('\t<tr>\n')
            output_file.write('\t\t%s' % ('<td><a href="' + input_dir))
            output_file.write('%s' % (sec_prefix[sidx]))
            output_file.write('%s' % ('.' + fig_type))
            output_file.write('">\t\t')
            output_file.write('%s' % ('<img height=' + str(pixel_num) + ' src="' + input_dir))
            output_file.write('%s' % (sec_prefix[sidx]))
            output_file.write('%s' % ('.' + fig_type))
            output_file.write('"></a></td>\n')
            output_file.write('\t</tr>\n')
            output_file.write('</table>\n')

    output_file.close()
