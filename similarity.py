import numpy as np
from sklearn.metrics import jaccard_similarity_score

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

if __name__ == "__main__":
    vector1 = np.random.randint(2, size=1000)
    vector2 = np.random.randint(2, size=1000)
    jaccard_sim = jaccard_similarity(vector1, vector2)
    print(jaccard_sim)
    
    vector1 = np.random.randint(20000, size=100)
    vector2 = np.random.randint(20000, size=100)

    simple_sim = simple_similarity(vector1, vector2)
    print(simple_sim)