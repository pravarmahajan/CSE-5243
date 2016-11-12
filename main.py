import numpy as np
import json

import shingling
import minhashing
import similarity
import minashing_config

def main():
    num_shingles = 1
    num_hashes_list = [16, 32, 64, 128, 256]
    
    '''First step, create shingles'''
    
    print "Constructing %d shingles..." %num_shingles
    with open('data/output/parsed_documents.txt') as f:
        parsed_documents = json.load(f)

    shingled_docs, feature_names = \
        shingling.create_word_shingles(parsed_documents, num_shingles)

    #Save the shingles to data/shingles directory
    shingling.write_sparse_data_matrix_to_file(
        shingled_docs, str(num_shingles) + "_shingles")

    shingled_docs = np.transpose(shingled_docs).tocsr()
    

    '''Second step, create hashes with the shingled doc'''
    for k in num_hashes_list:
        print "Creating %d hash" %k
        time1 = time.time()
        hashes = minhashing.perform_LSH(shingled_docs, k)
        time2 = time.time()
        print "Time to generate hashes = %.2f secs" %(time2-time1)
        minhashing.save_hashes(hashes)

    '''Compute Jaccard Similarity'''
    jacc_sim_matrix = similarity.jacc_sim_for_doc_collection(shingled_docs)
    filename = os.path.join(shingles_path, jaccard_similarity_file)
    np.savetxt(filename, jacc_sim_matrix, 
                dtype = 'float', fmt = "%0.2f", delimiter = ' ')
    
    '''Compute k-hash similarities'''
    SSE = []
    for k in num_hashes_list:
        k_hash_similarity = similarity.simple_sim_for_doc_collection(hashed_docs)
        SSE.append(similarity.compute_SSE(jacc_sim_matrix, k_hash_similarity))
        filename = os.path.join(shingles_path, str(k)+'_hash_sim.dat')
        np.savetxt(filename, k_hash_similarity, 
                    dtype = 'float', fmt = "%0.2f", delimiter = ' ')
    print SSE

if __name__ == "__main__":
    main()
