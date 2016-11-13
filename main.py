import numpy as np
import json
import matplotlib.pyplot as plt
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
    num_shingles = 1
    num_hashes_list = [16,32,64,128,256]
    
    '''First step, create shingles'''
    
    print "Constructing %d shingles..." %num_shingles
    with open('data/output/parsed_documents.txt') as f:
        parsed_documents = json.load(f)
    shuffle(parsed_documents)
    if len(sys.argv) > 1:
        parsed_documents = parsed_documents[:int(sys.argv[1])]

    #parsed_documents = \
    #[x for x in parsed_documents if 'topics' in x.keys() and 'cocoa' in
    #x['topics']]

    shingled_docs, feature_names = \
        shingling.create_word_shingles(parsed_documents, num_shingles)

    #Save the shingles to data/shingles directory
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

    '''Compute Jaccard Similarity'''
    print "Computing Jaccard Similarity"
    time1 = time.time()
    jacc_sim_matrix = similarity.jacc_sim_for_doc_collection(shingled_docs)
    time2 = time.time()
    jacc_time = time2-time1
    print "Time taken to compute jaccard similarity = %d" %jacc_time
    filename = os.path.join(shingles_path, jaccard_similarity_file)
    np.savetxt(filename, jacc_sim_matrix, fmt = "%0.2f", delimiter = ' ')
    
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
        print "Time taken to compute %d hash similarity = %.2f"\
            %(k, hash_sim_times[-1])
        MSE.append(similarity.compute_MSE(jacc_sim_matrix, k_hash_similarity))
        print "MSE for %d hash = %.2e" %(k, MSE[-1])
        filename = os.path.join(shingles_path, str(k)+'_hash_sim.dat')
        np.savetxt(filename, k_hash_similarity, fmt = "%0.2f", delimiter = ' ')

    '''Graph the results'''
    plt.xscale('log')
    plt.xlabel('Number of Hashes (log scale)')
    plt.ylabel('Mean Squared Error (linear scale)')
    plt.xticks(num_hashes_list, num_hashes_list)
    plt.plot(num_hashes_list, MSE)
    plt.savefig('plots/MSE.png')

    plt.clf()
    plt.xscale('log')
    plt.xlabel('Number of Hashes (log scale)')
    plt.ylabel('Time Taken(linear scale)')
    plt.xticks(num_hashes_list, num_hashes_list)
    plt.plot(num_hashes_list, hash_sim_times)
    plt.savefig('plots/time.png')

if __name__ == "__main__":
    main()
