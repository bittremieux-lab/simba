from sklearn.decomposition import PCA


class PCAEmbeddings:

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.pca = PCA(n_components=self.n_components)

    # Perform PCA
    def fit(self, data):
        self.pca.fit(data)

    def transform(self, data):
        return self.pca.transform(data)
