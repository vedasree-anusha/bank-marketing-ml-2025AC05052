import os
import joblib

from sklearn.pipeline import Pipeline

from backend.data_preprocessing import prepare_data

from model.logistic_regression import create_model as create_logistic_model
from model.decision_tree import create_model as create_tree_model
from model.knn import create_model as create_knn_model
from model.naive_bayes import create_model as create_nb_model
from model.random_forest import create_model as create_rf_model

#paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

#model configuration
MODELS = {
    "logistic_regression": create_logistic_model,
    "decision_tree": create_tree_model,
    "knn": create_knn_model,
    "naive_bayes": create_nb_model,
    "random_forest": create_rf_model
}

#train models
def train_models():

    (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        numerical_features,
        categorical_features
    ) = prepare_data()

    # Create model directory
    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    trained_models = {}

    print("\n" + "=" * 60)
    print("STARTING MODEL TRAINING")
    print("=" * 60)

    # Train every model
    for model_name, model_creator in MODELS.items():

        print(f"\nTraining: {model_name}")

        # Create classifier
        classifier = model_creator()

        # Combine preprocessing + classifier
        pipeline = Pipeline([
            (
                "preprocessor",
                preprocessor
            ),
            (
                "classifier",
                classifier
            )
        ])

        # Train
        pipeline.fit(
            X_train,
            y_train
        )

        # Save trained pipeline
        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.pkl"
        )

        joblib.dump(
            pipeline,
            model_path
        )

        trained_models[model_name] = pipeline

        print(
            f"Saved: {model_path}"
        )

    print("\n" + "=" * 60)
    print("ALL MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)

    return (
        trained_models,
        X_test,
        y_test
    )

if __name__ == "__main__":

    train_models()