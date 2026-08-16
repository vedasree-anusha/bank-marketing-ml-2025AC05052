import os
import pandas as pd

from backend.data_preprocessing import (
    load_data,
    clean_data,
    prepare_features_target,
    split_data
)

#paths
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

TEST_DATA_PATH = os.path.join(
    BASE_DIR,
    "test_data.csv"
)

#test_data
def create_test_data():

    # Load original dataset
    df = load_data()

    # Apply the same cleaning used during training
    df = clean_data(df)

    # Separate features and target
    X, y = prepare_features_target(df)

    # Create exactly the same train/test split
    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = split_data(X, y)

    # Convert target back to original form
    y_test_original = y_test.map({
        0: "no",
        1: "yes"
    })

    # Combine features and target
    test_df = X_test.copy()

    test_df["y"] = y_test_original

    # Save
    test_df.to_csv(
        TEST_DATA_PATH,
        index=False
    )

    print("\nTest data created successfully.")

    print(
        "Shape:",
        test_df.shape
    )

    print("\nTarget distribution:")
    print(
        test_df["y"].value_counts()
    )

    print("\nSaved to:")
    print(TEST_DATA_PATH)


if __name__ == "__main__":
    create_test_data()