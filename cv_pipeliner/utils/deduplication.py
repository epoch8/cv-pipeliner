from typing import List

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def get_minmax_order(embeddings: np.ndarray) -> List:
    similarity_matrix = cosine_similarity(embeddings)

    items = np.unravel_index(np.argmin(similarity_matrix, axis=None), shape=similarity_matrix.shape)
    items = list(items)

    while len(items) < len(similarity_matrix):
        new_index = np.argmin(np.max(similarity_matrix[items], axis=0))
        items.append(new_index)

    return items
