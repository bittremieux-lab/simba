import numpy as np


class CosineSimilarity:

    @staticmethod
    def batch_cosine_similarity(batch_vec1, batch_vec2):
        """
        Compute cosine similarity between two batches of vectors.

        Parameters:
        batch_vec1 (numpy.ndarray): First batch of vectors with shape (batch_size, vector_dim).
        batch_vec2 (numpy.ndarray): Second batch of vectors with shape (batch_size, vector_dim).

        Returns:
        numpy.ndarray: Cosine similarities between the batches of vectors with shape (batch_size,).
        """
        dot_product = np.sum(batch_vec1 * batch_vec2, axis=1)
        norm_vec1 = np.linalg.norm(batch_vec1, axis=1)
        norm_vec2 = np.linalg.norm(batch_vec2, axis=1)
        similarity = dot_product / (norm_vec1 * norm_vec2)
        return similarity
    
    @staticmethod
    def batch_cosine_similarity_combinations(batch_vec1, batch_vec2):
        """
        Compute cosine similarity between every combination of vectors from two batches.

        Parameters:
        batch_vec1 (numpy.ndarray): First batch of vectors with shape (batch_size_1, vector_dim).
        batch_vec2 (numpy.ndarray): Second batch of vectors with shape (batch_size_2, vector_dim).

        Returns:
        numpy.ndarray: Cosine similarities between every combination of vectors with shape (batch_size_1, batch_size_2).
        """
        dot_product = np.dot(batch_vec1, batch_vec2.T)
        norm_vec1 = np.linalg.norm(batch_vec1, axis=1)
        norm_vec2 = np.linalg.norm(batch_vec2, axis=1)
        similarity = dot_product / np.outer(norm_vec1, norm_vec2)
        return similarity