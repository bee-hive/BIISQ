#!/tigress/lifangc/anaconda/bin/python
import sys
import time
import os.path
import numpy as np
from visualize_npb_iso import *
#from parse_and_print_gff import *
#from compare_iso import *

# Check command line arguments
CMD_NUM = 7
if len(sys.argv) <  CMD_NUM:
    print 'Usage: ' + sys.argv[0] + "\n" \
            '                 <root_result_dir/>\n' + \
            '                 <exon length file>\n' + \
            '                 <flag plot all iterations (0/1)>\n' + \
            '                 <flag compute proportions (integer: 0 or 1 for now)>\n' + \
            '                 <flag skip plotting (0/1)>\n' + \
            '                 <flag discrete read mapping (0/1)>\n'
    exit (1)

root_result_dir = sys.argv[1]
# correct for trailing path separator
if root_result_dir[-1]!="\\" and root_result_dir[-1]!="/" and root_result_dir[-1]!=os.sep:
    root_result_dir+=root_result_dir+os.sep

result_file = root_result_dir + 'outfile.pkz' # zipped pickle
if not os.path.isfile(result_file):
    result_file = root_result_dir + 'outfile.pkl' # pickle
    if not os.path.isfile(result_file):
        result_file = root_result_dir + 'outfile.npz' # compressed numpy (last it)

exon_len_file = sys.argv[2]
flag_plot_all_iter = np.int(sys.argv[3])
flag_compute_alt = np.int(sys.argv[4])
flag_skip_plotting = np.int(sys.argv[5])
flag_discrete_read_map = np.int(sys.argv[6])

t0 = time.time()
visualize_iso_results(\
                        result_file             , \
                        root_result_dir         , \
                        exon_len_file           , \
                        flag_plot_all_iter      , \
                        flag_compute_alt        , \
                        flag_skip_plotting      , \
                        flag_discrete_read_map  \
                        )
# parse_and_print_gff(gene_id, root_gff_dir, gff_prefix, gff_num)
# compare_iso_reconstruction(root_result_dir, root_truth_dir)
tdiff = time.time()-t0
print 'Finished all jobs; elapsed time: ' + str(tdiff) + ' seconds.'