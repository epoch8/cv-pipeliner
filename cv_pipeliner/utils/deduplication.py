import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List


def get_minmax_order(
    embeddings: np.ndarray
) -> List:

    similarity_matrix = cosine_similarity(embeddings)

    items = np.unravel_index(np.argmin(
                        similarity_matrix, axis=None),
                        similarity_matrix.shape)
    items = list(items)

    while len(items) < len(similarity_matrix):
        dst = []
        for ids in items:
            dst.append(similarity_matrix[ids])
        dst = np.array(dst)
        new_index = np.argmin(np.amax(dst, axis=0))
        items.append(new_index)
    return items
