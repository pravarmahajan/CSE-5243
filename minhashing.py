import subprocess
import glob
import os.path
import numpy as np
import scipy.io
import scipy.sparse
import pdb

import minhashing_config

def perform_LSH(shingled_documents, k):
    doc_length, num_docs = shingled_documents.shape
    permutations = generate_permutations(doc_length, k)
    hashes = []
    for perm in permutations:
        perm_doc = shingled_documents[perm, :]
        perm_doc = perm_doc.tocsc()
        doc_hash = [find_min_non_zero_idx(perm_doc.getcol(i)) for i in range(num_docs)]
        hashes.append(doc_hash)
    return np.array(hashes)

def find_min_non_zero_idx(vector):
    return vector.nonzero()[0][0]
    
def generate_permutations(doc_length, k):
    permutations = []
    for i in range(k):
        permutations.append(np.random.permutation(doc_length))
    return permutations


def load_sparse_data_matrix_from_file(filename):
    full_filename = os.path.join(minhashing_config.shingles_path, filename)
    return scipy.io.mmread(full_filename)

def save_hashes(hashes):
    num_hashes = hashes.shape[0]
    filename = os.path.join(minhashing_config.shingles_path,
                str(num_hashes) + '_' + minhashing_config.hashes_filename)

    np.savetxt(filename, hashes, fmt = '%d', delimiter = ' ')

def load_hashes(num_hashes):
    filename = os.path.join(minhashing_config.shingles_path,
                str(num_hashes) + '_' + minhashing_config.hashes_filename)
    return np.loadtxt(filename, dtype='int')

if __name__ == "__main__":
    num_shingles = 1
    k = 100
    shingled_documents = load_sparse_data_matrix_from_file(str(num_shingles) + "_shingles")
    shingled_documents = np.transpose(shingled_documents).tocsr()
    hashes = perform_LSH(shingled_documents, k)
    save_hashes(hashes)
