from sklearn.neighbors import KNeighborsClassifier


def create_model():
    """
    Create K-Nearest Neighbors classifier.
    """

    model = KNeighborsClassifier(
        n_neighbors=5
    )

    return model