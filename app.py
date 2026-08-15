import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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

#page configuration
st.set_page_config(
    page_title="Bank Marketing ML Classifier by 2025AC05052",
    page_icon="",
    layout="wide"
)


#paths
BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

#model configuration
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl"
}

#load model
@st.cache_resource
def load_model(model_file):

    model_path = os.path.join(
        MODEL_DIR,
        model_file
    )

    return joblib.load(model_path)

#evaluate model
def evaluate_model(model, X_test, y_test):

    y_pred = model.predict(X_test)

    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(
            y_test,
            y_pred
        ),

        "AUC": roc_auc_score(
            y_test,
            y_prob
        ),

        "Precision": precision_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "Recall": recall_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "F1": f1_score(
            y_test,
            y_pred,
            zero_division=0
        ),

        "MCC": matthews_corrcoef(
            y_test,
            y_pred
        )
    }

    return metrics, y_pred


#header
st.title("Bank Marketing Classification by 2025AC05052")

st.markdown(
    """
    ### Predicting Term Deposit Subscription

    This application evaluates multiple machine learning
    classification models on the **Bank Marketing dataset**.

    Upload the test dataset, select a model, and explore its
    classification performance using multiple evaluation metrics.
    """
)

st.divider()

#side bar
st.sidebar.header("Model Configuration")

selected_model = st.sidebar.selectbox(
    "Select Classification Model",
    list(MODEL_FILES.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV",
    type=["csv"]
)

st.sidebar.markdown("---")

st.sidebar.markdown(
    """
    **Models implemented**

    • Logistic Regression  
    • Decision Tree  
    • K-Nearest Neighbors  
    • Naive Bayes  
    • Random Forest
    """
)

st.sidebar.info(
    """
    The uploaded CSV must contain the target
    column `y` for model evaluation.
    """
)


#file upload
if uploaded_file is None:

    st.info(
        "Upload `test_data.csv` from the sidebar to begin."
    )

    st.subheader("Project Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Dataset",
            "Bank Marketing"
        )

    with col2:
        st.metric(
            "Features",
            "19"
        )

    with col3:
        st.metric(
            "Test Samples",
            "8,236"
        )

    st.divider()

    st.subheader("Implemented Models")

    model_table = pd.DataFrame({
        "Model": list(MODEL_FILES.keys()),
        "Type": [
            "Linear",
            "Tree-based",
            "Instance-based",
            "Probabilistic",
            "Ensemble"
        ]
    })

    st.dataframe(
        model_table,
        use_container_width=True,
        hide_index=True
    )

    st.stop()


try:

    test_data = pd.read_csv(
        uploaded_file
    )

except Exception as e:

    st.error(
        f"Unable to read CSV: {e}"
    )

    st.stop()


#validate data
if "y" not in test_data.columns:

    st.error(
        "The uploaded CSV must contain the target column `y`."
    )

    st.stop()


st.subheader("Uploaded Test Dataset")

col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Test Records",
        f"{len(test_data):,}"
    )

with col2:

    st.metric(
        "Features",
        test_data.shape[1] - 1
    )

with col3:

    st.metric(
        "Actual Subscribers",
        f"{(test_data['y'] == 'yes').sum():,}"
    )


with st.expander("Preview Test Data"):

    st.dataframe(
        test_data.head(10),
        use_container_width=True
    )

#prepare data
X_test = test_data.drop(
    columns=["y"]
)

y_test = test_data["y"].map({
    "no": 0,
    "yes": 1
})

#selected model
st.divider()

st.subheader(
    f"{selected_model} — Evaluation Results"
)

try:

    selected_pipeline = load_model(
        MODEL_FILES[selected_model]
    )

    selected_metrics, selected_predictions = evaluate_model(
        selected_pipeline,
        X_test,
        y_test
    )

except Exception as e:

    st.error(
        f"Model prediction failed: {e}"
    )

    st.stop()

#metrics
col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Accuracy",
        f"{selected_metrics['Accuracy']:.2%}"
    )

    st.metric(
        "Precision",
        f"{selected_metrics['Precision']:.2%}"
    )


with col2:

    st.metric(
        "AUC",
        f"{selected_metrics['AUC']:.2%}"
    )

    st.metric(
        "Recall",
        f"{selected_metrics['Recall']:.2%}"
    )


with col3:

    st.metric(
        "F1 Score",
        f"{selected_metrics['F1']:.2%}"
    )

    st.metric(
        "MCC",
        f"{selected_metrics['MCC']:.4f}"
    )

#confusion matrix
st.divider()

st.subheader("Confusion Matrix")

cm = confusion_matrix(
    y_test,
    selected_predictions
)

fig, ax = plt.subplots(
    figsize=(6, 4)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["No", "Yes"],
    yticklabels=["No", "Yes"],
    ax=ax
)

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(
    f"{selected_model} - Confusion Matrix"
)

st.pyplot(
    fig,
    use_container_width=False
)

plt.close(fig)

#classification report
st.subheader("📋 Classification Report")

report = classification_report(
    y_test,
    selected_predictions,
    target_names=[
        "No",
        "Yes"
    ],
    output_dict=True,
    zero_division=0
)

report_df = pd.DataFrame(
    report
).transpose()

report_df.index.name = "Class"

st.dataframe(
    report_df.round(4),
    use_container_width=True
)

#prediction summary
st.divider()

st.subheader("Prediction Summary")

prediction_counts = pd.Series(
    selected_predictions
).map({
    0: "No",
    1: "Yes"
}).value_counts()

col1, col2 = st.columns(2)

with col1:

    st.metric(
        "Predicted No",
        f"{prediction_counts.get('No', 0):,}"
    )

with col2:

    st.metric(
        "Predicted Yes",
        f"{prediction_counts.get('Yes', 0):,}"
    )

#compare models
st.divider()

st.subheader("Model Comparison")

comparison_results = []

for model_name, model_file in MODEL_FILES.items():

    try:

        model = load_model(
            model_file
        )

        metrics, _ = evaluate_model(
            model,
            X_test,
            y_test
        )

        comparison_results.append({
            "ML Model": model_name,
            "Accuracy": metrics["Accuracy"],
            "AUC": metrics["AUC"],
            "Precision": metrics["Precision"],
            "Recall": metrics["Recall"],
            "F1": metrics["F1"],
            "MCC": metrics["MCC"]
        })

    except Exception as e:

        st.warning(
            f"Could not evaluate {model_name}: {e}"
        )


comparison_df = pd.DataFrame(
    comparison_results
)


# Format for display

display_df = comparison_df.copy()

for column in [
    "Accuracy",
    "AUC",
    "Precision",
    "Recall",
    "F1"
]:

    display_df[column] = (
        display_df[column] * 100
    ).round(2).astype(str) + "%"


display_df["MCC"] = (
    display_df["MCC"]
    .round(4)
)


st.dataframe(
    display_df,
    use_container_width=True,
    hide_index=True
)

#overall assessment
st.divider()

st.subheader("Overall Model Assessment")

# Logistic Regression is selected based on
# AUC, Recall, F1 and MCC for this imbalanced dataset.

st.success(
    """
    **Overall Best Performing Model: Random Forest**

    Random Forest provides the strongest overall performance
    on this dataset. It achieves the highest AUC (80.91%),
    F1 Score (51.34%), and MCC (0.4478), while also achieving
    a strong recall of 60.78%.

    Although KNN achieves the highest accuracy (89.55%) and
    precision (56.63%), its recall is only 30.82%. Since the
    dataset is highly imbalanced, accuracy alone is not
    sufficient to select the best model.

    Random Forest therefore provides the best overall balance
    between identifying positive customers and limiting
    incorrect predictions.
    """
)

#footer
st.divider()

st.caption(
    "Machine Learning Assignment 2 | 2025AC05052 | Bank Marketing Classification"
)