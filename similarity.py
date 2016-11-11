import numpy as np

def jaccard_similarity(vector1, vector2):
    similarity = 0
    return similarity

def simple_similarity(vector1, vector2):
    similarity = 0
    return similarity

if __name__ == "__main__":
    vector1 = np.random.randint(2, size=1000)
    vector2 = np.random.randint(2, size=1000)
    jaccard_sim = jaccard_similarity(vector1, vector2)
    
    vector1 = np.random.randint(20000, size=100)
    vector2 = np.random.randint(20000, size=100)

    simple_sim = simple_similarity(vector1, vector2)
