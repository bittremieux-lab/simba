import numpy as np


class CosineDistance:

    @staticmethod
    def compute_cosine_distance(array1, array2):

        kernel = np.ones(5)

        array1_conv = np.convolve(array1, kernel)
        array2_conv = np.convolve(array2, kernel)
        dot_product = np.dot(array1_conv, array2_conv)
        norm_array1 = np.linalg.norm(array1_conv)
        norm_array2 = np.linalg.norm(array2_conv)

        return dot_product / (norm_array1 * norm_array2)
