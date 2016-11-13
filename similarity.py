import numpy as np
from sklearn.metrics import mean_squared_error
import minhashing_config
import os
import scipy.io
import pdb

def jaccard_similarity(vector1, vector2):
    vector1_is_true = set(vector1.nonzero()[0])
    vector2_is_true = set(vector2.nonzero()[0])
    similarity = float(len(vector1_is_true & vector2_is_true)) \
            / len(vector1_is_true | vector2_is_true)
    return similarity

def simple_similarity(vector1, vector2):
    similarity = float(np.sum(vector1 == vector2))
    similarity = similarity/len(vector1)
    return similarity

def jacc_sim_for_doc_collection(docs):
    docs = docs.tocsc()
    w = docs.shape[1]
    sim_matrix = np.zeros((w,w))
    for i in range(w):
        for j in range(i+1, w):
            sim_matrix[i][j] = jaccard_similarity(docs.getcol(i),docs.getcol(j))
            sim_matrix[j][i] = sim_matrix[i][j]

    return sim_matrix

def simple_sim_for_doc_collection(docs):
    w = h = docs.shape[1]
    sim_matrix = [[0 for x in range(w)] for y in range(w)] 
    for i in range(w):
        for j in range(i+1, w):
            sim_matrix[i][j] = simple_similarity(docs[:, i], docs[:, j])
            sim_matrix[j][i] = sim_matrix[i][j]

    return sim_matrix

def compute_MSE(jacc_sim_matrix, hash_sim_matrix):
    sse = mean_squared_error(jacc_sim_matrix, hash_sim_matrix)
    return sse

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
