import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import time
import os
import sys
import pdb
from random import shuffle

from minhashing_config import *
import minhashing
import similarity
import shingling

def main():
    num_hashes_list = [16,32,64,128,256]
    
    '''First step, create shingles'''
    
    with open('data/output/parsed_documents.txt') as f:
        parsed_documents = json.load(f)
    shuffle(parsed_documents)
    if len(sys.argv) > 2:
        parsed_documents = parsed_documents[:int(sys.argv[1])]
        num_shingles = int(sys.argv[2])
    elif len(sys.argv) > 1:
        parsed_documents = parsed_documents[:int(sys.argv[1])]
        num_shingles = 1
    else:
        num_shingles = 1

    print "Constructing %d shingles..." %num_shingles

    #parsed_documents = \
    #[x for x in parsed_documents if 'topics' in x.keys() and 'cocoa' in
    #x['topics']]

    shingled_docs, feature_names = \
        shingling.create_word_shingles(parsed_documents, num_shingles)
    #Save the shingles to data/shingles directory
    del parsed_documents
    del feature_names

    shingling.write_sparse_data_matrix_to_file(
        shingled_docs, str(num_shingles) + "_shingles")

    shingled_docs = np.transpose(shingled_docs).tocsr()
    

    '''Second step, create hashes with the shingled doc'''
    hash_creation_times = []
    print "Creating Hashes"
    for k in num_hashes_list:
        time1 = time.time()
        hashes = minhashing.perform_LSH(shingled_docs, k)
        time2 = time.time()
        hash_creation_times.append(time2-time1)
        print "Time to generate %d hashes = %.2f secs" \
            %(k, hash_creation_times[-1])
        minhashing.save_hashes(hashes)

    del hashes
    
    '''Compute Jaccard Similarity'''
    print "Computing Jaccard Similarity"
    time1 = time.time()
    jacc_sim_matrix = similarity.jacc_sim_for_doc_collection(shingled_docs)
    time2 = time.time()
    jacc_time = time2-time1
    print "Time taken to compute jaccard similarity = %d secs" %jacc_time

    del shingled_docs
    
    '''Compute k-hash similarities'''
    MSE = []
    hash_sim_times = []
    print "Computing similarity of hashed documents"
    for k in num_hashes_list:
        hashes = minhashing.load_hashes(k)
        time1 = time.time()
        k_hash_similarity = similarity.simple_sim_for_doc_collection(hashes)
        time2 = time.time() 
        hash_sim_times.append(time2-time1)
        print "Time taken to compute %d hash similarity = %.2f secs"\
            %(k, hash_sim_times[-1])
        MSE.append(similarity.compute_MSE(jacc_sim_matrix, k_hash_similarity))
        print "MSE for %d hash = %.2e" %(k, MSE[-1])

    del jacc_sim_matrix
    del k_hash_similarity

    '''Graph the results'''
    axes = plt.gca()
    plt.xscale('log')
    plt.xlabel('Number of Hashes (log scale)')
    plt.ylabel('Mean Squared Error (in thousandths)')
    formatter = FuncFormatter(thousandths)
    axes.yaxis.set_major_formatter(formatter)
    plt.xticks(num_hashes_list, num_hashes_list)
    plt.plot(num_hashes_list, MSE)
    plt.savefig('plots/MSE' + str(num_shingles) + '.png')

    plt.clf()
    plt.xscale('log')
    plt.xlabel('Number of Hashes (log scale)')
    plt.ylabel('Time Taken(in secs)')
    plt.xticks(num_hashes_list, num_hashes_list)
    plt.plot(num_hashes_list, hash_sim_times, 'b', label='k-Hash Similarity')
    plt.plot(num_hashes_list, [jacc_time]*len(num_hashes_list),
            'r', label='Jaccard Baseline')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=3,
           ncol=2, mode="expand", borderaxespad=0.)
    plt.savefig('plots/time' + str(num_shingles) + '.png')

def thousandths(x, pos):
    return '%0.2f' %(x*1000)
    
if __name__ == "__main__":
    main()
