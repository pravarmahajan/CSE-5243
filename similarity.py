import numpy as np
from sklearn.metrics import mean_squared_error
import minhashing_config
import os
import scipy.io
import pdb

def jaccard_similarity(set1,set2):
    similarity = float(len(set1&set2))/len(set1|set2)
    return similarity

def jacc_sim_for_doc_collection(docs):
    docs = docs.tocsc()
    w = docs.shape[1]
    docs_as_sets = [set(docs.getcol(i).nonzero()[0]) for i in range(w)]
    sim_matrix = np.zeros((w,w))
    for i in range(w):
        for j in range(i+1, w):
            sim_matrix[i][j] = jaccard_similarity(docs_as_sets[i],
                                                  docs_as_sets[j])
    return sim_matrix

def simple_sim_for_doc_collection(docs):
    docs.transpose()
    w = docs.shape[1]
    sim_matrix = np.zeros((w,w))
    for i in range(w):
        for j in range(i+1, w):
            sim_matrix[i][j] = np.count_nonzero(np.equal(docs[:, i], docs[:, j]))/float(w)

    return sim_matrix

def compute_MSE(jacc_sim_matrix, hash_sim_matrix):
    return mean_squared_error(jacc_sim_matrix, hash_sim_matrix)*2

if __name__ == "__main__":
    hashes_filename = os.path.join(minhashing_config.shingles_path,
    minhashing_config.hashes_filename)
    hashed_docs = np.loadtxt(hashes_filename, dtype='int')
    hash_sim_matrix = simple_sim_for_doc_collection(hashed_docs)

    shingles_filename = os.path.join(minhashing_config.shingles_path,
    '1_shingles.dat')
    shingles = scipy.io.mmread(shingles_filename)

    jacc_sim_matrix = jacc_sim_for_doc_collection(shingles)

    sse = compute_SSE(jacc_sim_matrix, hash_sim_matrix)
