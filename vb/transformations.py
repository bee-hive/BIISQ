'''
Created on Jan 14, 2015

@author: Ouroboros
'''
import numpy as np
import init_vb as ivb
import itertools
from scipy.special.basic import comb
import sys
from sets import Set

def get_vector_from_bincounts(bc):
        """ Converts aggregated category counts into a vector of individual samples. """ # TODO this is so inefficient
        v = np.empty(np.sum(bc,dtype=np.uint32),dtype=np.uint32)
        n = 0
        for i in range(bc.size):
            for l in range(int(bc[i])):
                v[n] = i
                n+=1
        return v

def read_BED_ref(bed_file,ref,FIRST_EXON_ENCODING):
    ref_bed=dict()
    ref_to_curr_exon=dict()
    with open(bed_file) as f:
        for line in f:
            match=0
            isexon=0
            gene=None
            parts=line.strip().split()
            for idx in range(3,len(parts)):
                if parts[idx]=="gene_id" and (ref is None or ref==parts[idx+1].replace("\"", '').replace(";", '')):
                    match=1
                    gene=parts[idx+1].replace("\"", '').replace(";", '')
                elif parts[idx]=="exon_number":
                    isexon=1
            if match and isexon:
                if gene not in ref_to_curr_exon:
                    ref_to_curr_exon[gene]=FIRST_EXON_ENCODING
                row=np.zeros((4),dtype=np.int)
                row[0]=np.int(ref_to_curr_exon[gene])
                row[1]=np.int(parts[1])
                row[2]=np.int(parts[2])-np.int(parts[1])+1
                if ref_to_curr_exon[gene]>FIRST_EXON_ENCODING:
                    ref_bed[gene][ref_to_curr_exon[gene]-2][3]=row[1]-(ref_bed[gene][ref_to_curr_exon[gene]-2][1]+ref_bed[gene][ref_to_curr_exon[gene]-2][2])
                row[3]=-1
                if gene not in ref_bed:
                    ref_bed[gene]=[]
                ref_bed[gene].append(row)
                ref_to_curr_exon[gene]+=1
                
    return ref_bed

def read_GTF_ref(gtf_file,ref,FIRST_EXON_ENCODING):
    ref_gtf=dict()
    ref_to_curr_exon=dict()
    with open(gtf_file) as f:
        for line in f:
            if line[0]=="#":
                continue
            match=0
            isexon=0
            gene=None
            parts=line.strip().split()
            gene=parts[0]
            typefield=parts[2]
            if gene==ref and typefield=="exon":
                match=1
                isexon=1
            if match and isexon:
                if gene not in ref_to_curr_exon:
                    ref_to_curr_exon[gene]=FIRST_EXON_ENCODING
                row=np.zeros((4),dtype=np.int)
                row[0]=np.int(ref_to_curr_exon[gene])
                row[1]=np.int(parts[3])
                row[2]=np.int(parts[4])-np.int(parts[3])+1
                if ref_to_curr_exon[gene]>FIRST_EXON_ENCODING:
                    ref_gtf[gene][ref_to_curr_exon[gene]-2][3]=row[1]-(ref_gtf[gene][ref_to_curr_exon[gene]-2][1]+ref_gtf[gene][ref_to_curr_exon[gene]-2][2])
                row[3]=-1
                if gene not in ref_gtf:
                    ref_gtf[gene]=[]
                ref_gtf[gene].append(row)
                ref_to_curr_exon[gene]+=1
                
    return ref_gtf
        
def term_exon_junction_init(term2exon_read, collapse_jnx_tuples=0, exon_correction=0):
    max_v = 0
    term2exon=np.empty(len(term2exon_read),dtype=np.int32)
    isjunction=np.zeros(len(term2exon_read),dtype=np.int32)
    term2junction={}
    term2starts={}
    term2ends={}
    exonset=Set()
    for l in range(len(term2exon_read)):
        exonset.clear()
        if ":" in term2exon_read[l].strip(): # position contained in read
            starts=[]
            ends=[]
            exons_covered=[]
            reads=term2exon_read[l].strip().split(';')
            all_exons_added_this_read=[]
            for read in reads:
                
                this_read_exons_covered=[]
                read_term_parts=read.split(':') # position:exon1,exon2...
                
                if len(read_term_parts)==3:
                    ex=read_term_parts[2]
                else:
                    ex=read_term_parts[1]
                    
                ex=ex.split(',')
                sorted_exons = sorted([int(x) for x in ex])
                sorted_exons = [exon - exon_correction for exon in sorted_exons] 
                term2exon[l]=int(sorted_exons[0])
                for exon in sorted_exons: # junctions need to be sorted for the rest of the code to work
                    if exon not in this_read_exons_covered:
                        this_read_exons_covered.append(exon)
#                 for exon in this_read_exons_covered:
#                     if exon not in all_exons_added_this_read:
                
                insertidx = len(exons_covered)
                if len(exons_covered)>0 and sorted_exons[0]<exons_covered[0][0]:
                    insertidx = 0
                for insertidx_iter in range(1,len(exons_covered)):
                    if sorted_exons[0]>exons_covered[insertidx_iter-1][0] and sorted_exons[0]<exons_covered[insertidx_iter][0]:
                        insertidx=insertidx_iter
                    
                starts.insert(insertidx,np.int64(read_term_parts[0]))
                if len(read_term_parts)==3:
                    ends.insert(insertidx,np.int64(read_term_parts[1]))
                else:
                    ends.insert(insertidx,np.int64(read_term_parts[0]))
                exons_covered.insert(insertidx,tuple(this_read_exons_covered))
                [all_exons_added_this_read.append(exon) for exon in this_read_exons_covered]
                for exon in this_read_exons_covered:
                    exonset.add(exon)

            if len(exonset)>1:
                isjunction[l]=1
                ex=tuple(sorted([int(x) for x in exonset]))
                if collapse_jnx_tuples:
                    term2junction[l]=tuple(this_read_exons_covered)
                else:
                    if len(exons_covered)==1:
                        term2junction[l]=tuple(exons_covered,)
                    else:
                        term2junction[l]=tuple(exons_covered)
            if len(exonset)>max_v:
                max_v=len(exonset)
            term2starts[l]=tuple(starts)
            term2ends[l]=tuple(ends)
        else:
            v=1
            ex=term2exon_read[l].strip().split(',')
            term2exon[l]=int(ex[0])-exon_correction
            ex=tuple(sorted([int(x) for x in ex])) # junctions need to be sorted for the rest of the code to work
            ex=[exon - exon_correction for exon in ex] 
            if len(ex)>1:
                isjunction[l]=1
                term2junction[l]=tuple(ex)
                v=len(ex)
            if v>max_v:
                max_v=v
            
    return term2exon,isjunction,term2junction,term2starts,term2ends,max_v

def fix_paired_reads(isjunction,term2junction,max_v,refexons,term2starts,term2ends,INSERT_LENGTH_MEAN,INSERT_LENGTH_STDDEV,READ_LENGTH):
    # always sorted increasing number of exons
    # but the start positions could be any direction, increasing is forward direction, decreasing negative direction
    stopped_abruptly=0

    goodreads=0
    totalreads=0
    goodpairs=0
    badpairs=0
    for idx in term2junction.keys():
        if idx in term2starts:
            exons_covered = term2junction[idx]
            new_exons_in_jnc = []
            totalreads+=1
            goodreads+=1
            for exon in exons_covered[0]:
                new_exons_in_jnc.append(exon)
            for read_idx in range(1,len(exons_covered)):
                totalreads+=1
                firstreadexongroup = exons_covered[read_idx-1]
                secondreadexongroup = exons_covered[read_idx]
                
                notcontained=0
                for x in secondreadexongroup:
                    if x not in firstreadexongroup:
                        notcontained+=1
                if notcontained==0:
                    for exon in exons_covered[read_idx]:
                        if exon not in new_exons_in_jnc:
                            new_exons_in_jnc.append(exon)
                    goodreads+=1
                    continue
                notcontained=0
                for x in firstreadexongroup:
                    if x not in secondreadexongroup:
                        notcontained+=1
                if notcontained==0:
                    for exon in exons_covered[read_idx]:
                        if exon not in new_exons_in_jnc:
                            new_exons_in_jnc.append(exon)
                    goodreads+=1
                    continue
                
                firstreadstart=term2starts[idx][read_idx-1]
                firstreadend=term2ends[idx][read_idx-1]
                if firstreadend<firstreadstart:
                    temp=firstreadend
                    firstreadend=firstreadstart
                    firstreadstart=temp
                    
                secondreadstart=term2starts[idx][read_idx]
                secondreadend=term2ends[idx][read_idx]
                if secondreadend<secondreadstart:
                    temp=secondreadend
                    secondreadend=secondreadstart
                    secondreadstart=temp
                
                # reference direction for reference exons
                if refexons[firstreadexongroup[0]][1]<refexons[secondreadexongroup[0]][1]:
                    forwardstrand=1
                else:
                    forwardstrand=0

                if forwardstrand:
                    bases_known_to_exist_between_reads=refexons[firstreadexongroup[-1]][1]+refexons[firstreadexongroup[-1]][2]-firstreadend
                    bases_known_to_exist_between_reads+=secondreadstart-refexons[secondreadexongroup[0]][1]
                    fromexon=firstreadexongroup[-1]+1 
                    toexon=secondreadexongroup[0]
                else:
#                     bases_known_to_exist_between_reads=refexons[firstreadexongroup[-1]][1]+refexons[firstreadexongroup[-1]][2]-firstreadstart
                    bases_known_to_exist_between_reads=firstreadstart-refexons[firstreadexongroup[-1]][1]
                    bases_known_to_exist_between_reads+=refexons[secondreadexongroup[0]][1]+refexons[secondreadexongroup[0]][2]-secondreadend
                    fromexon=firstreadexongroup[-1]+1 
                    toexon=secondreadexongroup[0]

#                 if toexon+1<fromexon:
#                     print 'here'
                in_between_exons = range(fromexon,toexon) # exon numbering from 1
                if not in_between_exons:
                    for exon in exons_covered[read_idx]:
                        if exon not in new_exons_in_jnc:
                            new_exons_in_jnc.append(exon)
                    goodreads+=1
                    continue


                base_distance=bases_known_to_exist_between_reads
                best_comb=tuple()
                best_diff=np.abs(base_distance-INSERT_LENGTH_MEAN)
#                 if base_distance>1000 or base_distance<-1000:
#                     print "poor read"
                # read direction 
                
#                 if term2starts[idx][read_idx-1]>term2starts[idx][read_idx]: # read must be going other direction
# 
#                     endexon = refexons[exon_group_right[0]] # exon_no start length
#                     exonstart = endexon[1]
#                     exonlength = endexon[2]
#                     readstart = term2starts[idx][read_idx]
#                     readend = term2ends[idx][read_idx]
#                     if readend>readstart:
#                         print 'assert false'
#                         temp=readend
#                         readend=readstart
#                         readstart=temp
#                     else :
#                         print 'assert true'
#                         
#                         # do we just switch readend and readstart, and exons_covered?
# 
#                     rest_of_right_exon = exonstart+exonlength-readend
#                     bases_known_to_exist_between_starts = rest_of_right_exon
#                     startexon = refexons[exon_group_left[-1]] 
#                     exonstart = startexon[1]
#                     exonlength = startexon[2]
#                     readstart = term2starts[idx][read_idx-1]
#                     readend = term2ends[idx][read_idx-1]
#                     if readend>readstart:
#                         print 'assert false'
#                         temp=readend
#                         readend=readstart
#                         readstart=temp
#                     else :
#                         print 'assert true'
#                     rest_of_left_exon = readstart-exonstart
#                     bases_known_to_exist_between_starts+=rest_of_left_exon
#                     
#                     base_distance=bases_known_to_exist_between_starts
#                     best_comb=tuple()
#                     best_diff=np.abs(base_distance-INSERT_LENGTH_MEAN)
#                 else:
#                     endexon = refexons[exon_group_left[-1]] # exon_no start length
#                     exonstart = endexon[1]
#                     exonlength = endexon[2]
#                     readstart = term2starts[idx][read_idx-1]
#                     readend = term2ends[idx][read_idx-1]
#                     if readend<readstart:
#                         print 'assert false'
#                         temp=readend
#                         readend=readstart
#                         readstart=temp
#                     else :
#                         print 'assert true'
#                     rest_of_left_exon = exonstart+exonlength-readend
#                     bases_known_to_exist_between_end_and_start = rest_of_left_exon
#                     startexon = refexons[exon_group_right[0]] 
#                     exonstart = startexon[1]
#                     readstart = term2starts[idx][read_idx]
#                     readend = term2ends[idx][read_idx]
#                     if readend<readstart:
#                         print 'assert false'
#                         temp=readend
#                         readend=readstart
#                         readstart=temp
#                     else :
#                         print 'assert true'
#                     rest_of_right_exon = readstart-exonstart
#                     bases_known_to_exist_between_end_and_start+=rest_of_right_exon
#                     
#                     base_distance=bases_known_to_exist_between_end_and_start
#                     best_comb=tuple()
#                     best_diff=np.abs(base_distance-INSERT_LENGTH_MEAN)
                    
                # compare dis in reads to read length + insert size
                inf_cntr = 0
                for num_exons_between in range(1,len(in_between_exons)+1):
                    no_new_best_all_positive_distance=1
                    for comb in itertools.combinations(in_between_exons, num_exons_between):
                        # calculate the distance assuming comb exons were in the isoform
                        distance = base_distance
                        for exon in comb:
                            distance+=refexons[exon][2]
                        diff=np.abs(distance-INSERT_LENGTH_MEAN)
                        if diff<best_diff:
                            no_new_best_all_positive_distance=0
                            best_comb=comb
                            best_diff=diff     
                        inf_cntr+=1
                    if no_new_best_all_positive_distance or inf_cntr>20000000:
                        stopped_abruptly+=1
                        best_diff=100000000
                        break;  

                if abs(best_diff)>INSERT_LENGTH_STDDEV:
#                     print >> sys.stderr, 'difference outside standard deviation range ' + str(best_diff)
                    badpairs+=1
                    continue
                goodreads+=1
                goodpairs+=1
                
                exon_group_this=[]
                for exon in best_comb:
                    exon_group_this.append(exon)
                for exon in exons_covered[read_idx]:
                    exon_group_this.append(exon)
                for exon in exon_group_this:
                    if exon not in new_exons_in_jnc:
                        new_exons_in_jnc.append(exon)

#             print exons_covered
#             print new_exons_in_jnc
            if len(new_exons_in_jnc)==1:
                isjunction[idx]=0
                if idx in term2junction:
                    del term2junction[idx]
            else:
                isjunction[idx]=1
                term2junction[idx]=tuple(new_exons_in_jnc)
                if len(new_exons_in_jnc)>max_v:
                    max_v=len(new_exons_in_jnc)
    
    if totalreads>0:
        print >> sys.stderr, "total number of reads, total good, % good " + str(totalreads) + "\t"  + str(goodreads) + "\t" +  str((float(goodreads)/float(totalreads))) + "\t" + str(goodpairs)  + "\t" + str(badpairs)
        print >> sys.stderr, "number stopped abruptly: " + str(stopped_abruptly)
    return (isjunction,term2junction,max_v)

def fix_paired_reads_old(isjunction,term2junction,max_v,refexons,term2starts,INSERT_LENGTH_MEAN,INSERT_LENGTH_STDDEV,READ_LENGTH):
    goodreads=0
    totalreads=0
    for idx in term2junction.keys():
        if idx in term2starts:
            exons_covered = term2junction[idx]
            new_exons_in_jnc = []
            for exon in exons_covered[0]:
                new_exons_in_jnc.append(exon)
            for read_idx in range(1,len(exons_covered)):
                totalreads+=1
                exon_group_prev = exons_covered[read_idx-1]
                last_exon_prev = exon_group_prev[-1]
                first_exon_this = exons_covered[read_idx][0]
                in_between_exons = range(last_exon_prev+1,first_exon_this)
                if not in_between_exons:
                    for exon in exons_covered[read_idx]:
                        if exon not in new_exons_in_jnc:
                            new_exons_in_jnc.append(exon)
                    goodreads+=1
                    continue
                
                if term2starts[idx][read_idx-1]>term2starts[idx][read_idx]: # read must be going other direction
                    print "bad assumption"
                
                start_group_prev = term2starts[idx][read_idx-1]
                if (start_group_prev+READ_LENGTH)<refexons[exon_group_prev[0]][1]:
                    print >> sys.stderr, 'read start + read length starts well before exon: start ' + str(start_group_prev) + ' with exon start ' + str(refexons[exon_group_prev[0]][1])
                    continue 
                
                start_group_this = term2starts[idx][read_idx]
                if (start_group_this+READ_LENGTH)<refexons[first_exon_this][1]:
                    print >> sys.stderr, 'read start + read length starts well before exon: start ' + str(start_group_this) + ' with exon start ' + str(refexons[first_exon_this][1])
                    continue 
                
#                 if start_group_prev<refexons[exon_group_prev[0]][1] or  start_group_prev>(refexons[exon_group_prev[0]][1]+refexons[exon_group_prev[0]][2]):
#                     print >> sys.stderr, 'found bad read: ' + str(start_group_prev) + ' does not fit into [' + str(refexons[exon_group_prev[0]][1]) + ',' + str(refexons[exon_group_prev[0]][1]+refexons[exon_group_prev[0]][2]) + ']'
#                     continue
                
#                 if start_group_this<refexons[first_exon_this][1] or  start_group_this>(refexons[first_exon_this][1]+refexons[first_exon_this][2]):
#                     print >> sys.stderr, 'found bad read: ' + str(start_group_this) + ' does not fit into [' + str(refexons[first_exon_this][1]) + ',' + str(refexons[first_exon_this][1]+refexons[first_exon_this][2]) + ']'
#                     continue
                corrected_READ_LENGTH=READ_LENGTH
#                 rest_of_exon = 
                start_of_read_in_exon_offset = (start_group_prev-refexons[exon_group_prev[0]][1])
                if start_of_read_in_exon_offset < 0: 
                    # started in intron
                    corrected_READ_LENGTH+=start_of_read_in_exon_offset
                    start_of_read_in_exon_offset=0
                    
                    
                rest_of_read = corrected_READ_LENGTH-start_of_read_in_exon_offset-(refexons[exon_group_prev[0]][2])
                
                for exonidx in range(1,len(exon_group_prev)):
                    exon = exon_group_prev[exonidx]
                    rest_of_read=rest_of_read-refexons[exon][2]
                
                if rest_of_read < 0:
                    rest_of_exon=-1*rest_of_read
                    rest_of_read=0
                elif rest_of_read>=0: # some of the read is not accounted for, spills into next exon or intron?
#                     print str(rest_of_read)
                    print >> sys.stderr, 'found bad read: ' + str(start_group_this) + ' spills over into intron or next exon'
                    continue
#                     rest_of_exon=0
#                     rest_of_read=0
                covered_part_of_next_exon = start_group_this-refexons[first_exon_this][1]

                base_distance=rest_of_exon+covered_part_of_next_exon
                if base_distance<0:
                    print >> sys.stderr, 'found bad read: started ' + str(start_group_this) + ' with the aligned exon as ' + str(refexons[first_exon_this][1])
                    continue
                best_comb=tuple()
                best_diff=np.abs(base_distance-INSERT_LENGTH_MEAN)
                
                # compare dis in reads to read length + insert size
                
                for num_exons_between in range(1,len(in_between_exons)+1):
                    no_new_best_all_positive_distance=1
                    for comb in itertools.combinations(in_between_exons, num_exons_between):
                        # calculate the distance assuming comb exons were in the isoform
                        distance = base_distance
                        for exon in comb:
                            distance+=refexons[exon][2]
                        diff=np.abs(distance-INSERT_LENGTH_MEAN)
                        if distance<(INSERT_LENGTH_MEAN):
                            no_new_best_all_positive_distance=0
                        if diff<best_diff:
                            no_new_best_all_positive_distance=0
                            best_comb=comb
                            best_diff=diff     
                    if no_new_best_all_positive_distance:
                        break;       
    
                # QC 
                if abs(best_diff)>2*INSERT_LENGTH_STDDEV:
                    print >> sys.stderr, 'difference outside standard deviation range ' + str(best_diff)
                    continue
                goodreads+=1
                
                exon_group_this=[]
                for exon in best_comb:
                    exon_group_this.append(exon)
                for exon in exons_covered[read_idx]:
                    exon_group_this.append(exon)
                for exon in exon_group_this:
                    if exon not in new_exons_in_jnc:
                        new_exons_in_jnc.append(exon)

#             print exons_covered
#             print new_exons_in_jnc
            if len(new_exons_in_jnc)==1:
                isjunction[idx]=0
                if idx in term2junction:
                    del term2junction[idx]
            else:
                isjunction[idx]=1
                term2junction[idx]=tuple(new_exons_in_jnc)
                if len(new_exons_in_jnc)>max_v:
                    max_v=len(new_exons_in_jnc)
                    
    print >> sys.stderr, 'total number of reads, total good, % good ' + str(totalreads) + "\t"  + str(goodreads) + "\t" +  str((float(goodreads)/float(totalreads)))
    return (isjunction,term2junction,max_v)

def transform_term_2_exon(V,term2junction,term2exon,term2dis,max_V):

    term2exonsCovered=np.zeros((V,max_V),dtype=np.int32)
    term2exonsCoveredDistances=np.zeros((V,max_V),dtype=np.int32)
    term2exonsCoveredLengths=np.zeros((V),dtype=np.int32)
    term2exonsCovered[:,:]=-1
    
#             time_logging[time_ctr]+=time()
#             time_ctr+=1
    
    for v in range(V):
        if v in term2junction:
            term2exonsCovered[v,:len(term2junction[v])]=term2junction[v]
            term2exonsCoveredLengths[v]=len(term2junction[v])
            if v in term2dis:
                term2exonsCoveredDistances[v,:len(term2dis[v])]=term2dis[v]
        else:
            term2exonsCovered[v,0]=term2exon[v]
            term2exonsCoveredLengths[v]=1
            if v in term2dis:
                term2exonsCoveredDistances[v,0]=term2dis[v][0]
    return (term2exonsCovered,term2exonsCoveredLengths,term2exonsCoveredDistances)


def transform_term_2_exon_nonpaired(V,term2junction,term2exon,term2dis,max_V):

    term2exonsCovered=np.zeros((V,max_V),dtype=np.int32)
    term2exonsCoveredLengths=np.zeros((V),dtype=np.int32)
    term2exonsCovered[:,:]=-1
    
#             time_logging[time_ctr]+=time()
#             time_ctr+=1
    
    for v in range(V):
        if v in term2junction:
            term2exonsCovered[v,:len(term2junction[v])]=term2junction[v]
            term2exonsCoveredLengths[v]=len(term2junction[v])
        else:
            term2exonsCovered[v,0]=term2exon[v]
            term2exonsCoveredLengths[v]=1
    return (term2exonsCovered,term2exonsCoveredLengths)

def get_isoform_string(K,E):    
    iso_init= ivb.initialize_iso(K,E)
    isoform_string=''
    for i in range(iso_init.shape[0]):
        isoform_string = isoform_string + 'iso-'+ `(i+1)` + ':' + np.array_str(iso_init[i]) + '\n'
        
    return isoform_string[:-1]

def get_string_from_array(array):    
    string=''
    for i in range(array.__len__()):
        string = string + `(i+1)` + ':' + array[i].replace('\t', ' ') + '\n' # replace tabs with spaces... plotting api doesn't like tabs
        
    return string[:-1]

identity = lambda x: x