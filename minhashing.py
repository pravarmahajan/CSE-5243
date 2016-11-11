import subprocess
import glob
import os
import numpy as np
import scipy.io
import scipy.sparse

import pdb

import minhashing_config

def perform_LSH(shingled_documents, k):
    num_docs = shingled_documents.shape[0]
    permutations = generate_permutations(num_docs, k)
    hashes = []
    for perm in permutations():
        perm_doc = np.nonzero(shingled_document[perm, :])
        perm_doc = perm_doc.to_csc()
        
        hashes.append(doc_hash)
    permuted_docs = np.ndarray([shingled_documents[perm, :] for perm in permutations])
    pdb.set_trace()
    hashes = np.transpose(np.nonzero(permuted_docs))[0]

def find_min_non_zero_idx(vector):
    return vector.nonzero()[0,0]
    
def generate_permutations(num_docs, k):
    return np.random.permutation(num_docs * k).reshape((k, num_docs))

def load_sparse_data_matrix_from_file(filename):
    full_filename = os.path.join(minhashing_config.shingles_path, filename)
    return scipy.io.mmread(full_filename)

if __name__ == "__main__":
    num_shingles = 1
    shingled_documents = load_sparse_data_matrix_from_file(str(num_shingles) + "_shingles")
    shingled_documents = np.transpose(shingled_documents).tocsr()
    hashes = perform_LSH(shingled_documents, num_shingles)

