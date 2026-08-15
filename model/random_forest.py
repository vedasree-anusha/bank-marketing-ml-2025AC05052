from sklearn.ensemble import RandomForestClassifier


def create_model():
    """
    Create Random Forest classifier.
    """

    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    return model