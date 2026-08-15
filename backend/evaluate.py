import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib
matplotlib.use("Agg")

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

from backend.data_preprocessing import (
    load_data,
    clean_data,
    prepare_features_target,
    split_data
)


# --------------------------------------------------
# Project paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

RESULTS_DIR = os.path.join(
    BASE_DIR,
    "results"
)

CONFUSION_DIR = os.path.join(
    RESULTS_DIR,
    "confusion_matrices"
)


# --------------------------------------------------
# Create result directories
# --------------------------------------------------

os.makedirs(
    RESULTS_DIR,
    exist_ok=True
)

os.makedirs(
    CONFUSION_DIR,
    exist_ok=True
)


# --------------------------------------------------
# Models
# --------------------------------------------------

MODEL_NAMES = [
    "logistic_regression",
    "decision_tree",
    "knn",
    "naive_bayes",
    "random_forest"
]


# --------------------------------------------------
# Evaluate models
# --------------------------------------------------

def evaluate_models():

    # Load and prepare data
    df = load_data()

    df = clean_data(df)

    X, y = prepare_features_target(df)

    # IMPORTANT:
    # Use exactly the same split as training
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    results = []

    for model_name in MODEL_NAMES:

        print("\n" + "=" * 60)
        print(f"EVALUATING: {model_name}")
        print("=" * 60)

        # Load trained model
        model_path = os.path.join(
            MODEL_DIR,
            f"{model_name}.pkl"
        )

        pipeline = joblib.load(
            model_path
        )

        # Predictions
        y_pred = pipeline.predict(
            X_test
        )

        # Probability for positive class
        y_prob = pipeline.predict_proba(
            X_test
        )[:, 1]

        # --------------------------------------------------
        # Metrics
        # --------------------------------------------------

        accuracy = accuracy_score(
            y_test,
            y_pred
        )

        auc = roc_auc_score(
            y_test,
            y_prob
        )

        precision = precision_score(
            y_test,
            y_pred,
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            zero_division=0
        )

        mcc = matthews_corrcoef(
            y_test,
            y_pred
        )

        # --------------------------------------------------
        # Print metrics
        # --------------------------------------------------

        print(f"Accuracy : {accuracy:.4f}")
        print(f"AUC      : {auc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall   : {recall:.4f}")
        print(f"F1 Score : {f1:.4f}")
        print(f"MCC      : {mcc:.4f}")

        # --------------------------------------------------
        # Classification report
        # --------------------------------------------------

        print("\nClassification Report:")
        print(
            classification_report(
                y_test,
                y_pred,
                target_names=[
                    "No",
                    "Yes"
                ],
                zero_division=0
            )
        )

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------

        cm = confusion_matrix(
            y_test,
            y_pred
        )

        plt.figure(
            figsize=(6, 5)
        )

        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"]
        )

        plt.title(
            f"Confusion Matrix - {model_name.replace('_', ' ').title()}"
        )

        plt.xlabel(
            "Predicted"
        )

        plt.ylabel(
            "Actual"
        )

        plt.tight_layout()

        confusion_path = os.path.join(
            CONFUSION_DIR,
            f"{model_name}.png"
        )

        plt.savefig(
            confusion_path,
            dpi=300
        )

        plt.close()

        # --------------------------------------------------
        # Store results
        # --------------------------------------------------

        results.append({
            "ML Model": model_name.replace(
                "_",
                " "
            ).title(),

            "Accuracy": accuracy,
            "AUC": auc,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
            "MCC": mcc
        })

    # --------------------------------------------------
    # Save comparison
    # --------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    results_path = os.path.join(
        RESULTS_DIR,
        "model_comparison.csv"
    )

    results_df.to_csv(
        results_path,
        index=False
    )

    print("\n" + "=" * 80)
    print("MODEL COMPARISON")
    print("=" * 80)

    print(
        results_df.to_string(
            index=False
        )
    )

    print("\nResults saved to:")
    print(results_path)

    return results_df


# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    evaluate_models()