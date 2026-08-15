from sklearn.linear_model import LogisticRegression


def create_model():
    """
    Create Logistic Regression classifier.
    """

    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    )

    return model