import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import os

BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "bank_data.csv"
)


def load_data():
    """
    Load the Bank Marketing dataset.
    """

    df = pd.read_csv(DATA_PATH, sep=";")

    return df


def clean_data(df):
    """
    Perform basic data cleaning.

    - Remove duplicate rows
    - Remove duration to avoid target leakage
    """

    # Remove duplicate rows
    df = df.drop_duplicates()

    # Remove duration because it can cause target leakage
    df = df.drop(columns=["duration"])

    return df


def prepare_features_target(df):
    """
    Separate features and target variable.
    """

    X = df.drop(columns=["y"])

    # Convert target:
    # no  -> 0
    # yes -> 1
    y = df["y"].map({
        "no": 0,
        "yes": 1
    })

    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets.

    Stratification is used because the target classes
    are imbalanced.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def create_preprocessor(X):
    """
    Create preprocessing pipeline for numerical
    and categorical features.
    """

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    # Numerical preprocessing
    numerical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        ),
        (
            "scaler",
            StandardScaler()
        )
    ])

    # Categorical preprocessing
    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            )
        )
    ])

    # Combine both
    preprocessor = ColumnTransformer([
        (
            "numerical",
            numerical_pipeline,
            numerical_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ])

    return preprocessor, numerical_features, categorical_features


def prepare_data():
    """
    Complete preprocessing workflow.

    Returns:
        X_train
        X_test
        y_train
        y_test
        preprocessor
        numerical_features
        categorical_features
    """

    # Load
    df = load_data()

    print("Original dataset shape:", df.shape)

    # Clean
    df = clean_data(df)

    print(
        "Dataset shape after cleaning:",
        df.shape
    )

    # Features and target
    X, y = prepare_features_target(df)

    print("\nNumber of features:", X.shape[1])

    print("\nTarget distribution:")
    print(y.value_counts())

    # Train/test split
    X_train, X_test, y_train, y_test = split_data(
        X,
        y
    )

    print("\nTraining samples:", len(X_train))
    print("Testing samples:", len(X_test))

    # Preprocessor
    (
        preprocessor,
        numerical_features,
        categorical_features
    ) = create_preprocessor(X)

    print("\nNumerical features:")
    print(numerical_features)

    print("\nCategorical features:")
    print(categorical_features)

    return (
        X_train,
        X_test,
        y_train,
        y_test,
        preprocessor,
        numerical_features,
        categorical_features
    )


if __name__ == "__main__":

    prepare_data()