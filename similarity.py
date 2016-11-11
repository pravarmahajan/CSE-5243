import numpy as np
from sklearn.metrics import jaccard_similarity_score
import minhashconfig
import os
import scipy.io

def jaccard_similarity(vector1, vector2):
    similarity = 0
    similarity = jaccard_similarity_score(vector1, vector2, normalize=False)
    return similarity

def simple_similarity(vector1, vector2):
    similarity = 0
    for i in range(len(vector1)):
        if vector1[i]==vector2[i]:
            similarity += 1
    similarity = similarity/len(vector1)
    return similarity

def jacc_sim_for_doc_collection(docs):
    return sim_matrix

def simple_sim_for_doc_collection(docs):
    return sim_matrix

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
