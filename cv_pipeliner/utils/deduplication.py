import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def get_minmax_order(
    embeddings: np.ndarray
) -> List:
    
    similarity_matrix = cosine_similarity(embeddings)
    min_value = 1
    for i in range(similarity_matrix.shape[0]):
        for j in range(similarity_matrix.shape[1]):
            if i != j:
                if similarity_matrix[i][j] < min_value:
                    min_value = similarity_matrix[i][j]
                    indexes = (i,j)
    items = [indexes[0],indexes[1]]
    
    while len(items) < len(similarity_matrix):
        dst = []
        for ids in items:    
            dst.append(similarity_matrix[ids])
        dst = np.array(dst)
        new_index = np.argmin(np.amax(dst, axis=0)) 
        items.append(new_index)
    return items


