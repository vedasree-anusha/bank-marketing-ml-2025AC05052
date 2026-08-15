from sklearn.tree import DecisionTreeClassifier


def create_model():
    """
    Create Decision Tree classifier.
    """

    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=42
    )

    return model