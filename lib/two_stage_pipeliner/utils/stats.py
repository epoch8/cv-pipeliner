# from typing import List, Any
# from collections import Counter

# import numpy as np
# import pandas as pd

# def get_class_distribution(bboxes_data: List[Any]):
#     data_labels = np.array(
#         [item.label for item in data]
#     )
#     distribtuion = Counter(data_labels).most_common(10000)
#     distribtuion.insert(0, ("Общее коли�?е�?тво", len(data_labels)))
#     df_distribtuion = pd.DataFrame(
#         distribtuion, columns=['class_name', 'count']
#     )
#     df_distribtuion["percent"] = df_distribtuion['count'] / len(data_labels)
#     if filepath:
#         df_distribtuion.to_csv(filepath, index=False)
#         logger.info(
#             f"Save class distribution to '{filepath}'."
#         )

#     return df_distribtuion
