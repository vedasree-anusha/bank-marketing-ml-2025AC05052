# Bank Marketing Classification using Machine Learning

## a. Problem Statement

The objective of this project is to build and evaluate multiple machine learning classification models for predicting whether a bank customer will subscribe to a term deposit as a result of a marketing campaign.

The project implements multiple classification algorithms on the Bank Marketing dataset and compares their performance using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC).

An interactive Streamlit application is also developed to allow users to upload test data, select a machine learning model, view evaluation metrics, and analyze the confusion matrix and classification report.

---

## b. Dataset Description

### Dataset

**Bank Marketing Dataset**

### Source

The dataset is publicly available through the UCI Machine Learning Repository - [DataSet Link](https://archive.ics.uci.edu/dataset/222/bank+marketing)

### Dataset Size

The original dataset contains:

- **41,188 instances**
- **21 columns**
- **20 input attributes**
- **1 target variable**

After data cleaning, the dataset contains:

- **41,176 instances**
- **19 input features**
- **1 target variable**

### Target Variable

The target variable is: `y`

It indicates whether the customer subscribed to a term deposit.

| Target | Meaning | Count | Percentage |
|---|---|---:|---:|
| no | Customer did not subscribe | 36,537 | 88.73% |
| yes | Customer subscribed | 4,639 | 11.27% |

The target variable is therefore highly imbalanced. Because of this imbalance, accuracy alone is not sufficient to assess model performance. AUC, Precision, Recall, F1 Score, and MCC are also considered.

### Input Features

The dataset contains customer demographic information, campaign information, previous contact information, and economic indicators.

The 19 features used for model training are: age, job, marital, education, default, housing, loan, contact, month, day_of_week, campaign, pdays, previous, poutcome, emp.var.rate, cons.price.idx, cons.conf.idx, euribor3m, nr.employed

The `duration` feature was excluded from model training because it represents the duration of the current campaign contact and can introduce information that would not be available before the contact takes place.

---

## c. GitHub Repository Link

**Repository:**  
[GitHub repository link](https://github.com/vedasree-anusha/bank-marketing-ml-2025AC05052)

---

## d. Machine Learning Models Used

The following classification models were implemented on the same dataset:

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Random Forest Classifier

All models use the same training and testing dataset split and the same preprocessing pipeline to ensure a fair comparison.

The dataset was divided into:

- **Training samples:** 32,940
- **Testing samples:** 8,236

A stratified split was used to preserve the class distribution in the training and testing sets.

---

## Model Performance Comparison

The following metrics were calculated for each classification model:

- Accuracy
- AUC
- Precision
- Recall
- F1 Score
- Matthews Correlation Coefficient (MCC)


| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 83.03% | 80.02% | 35.91% | **64.55%** | 46.15% | 0.3928 |
| Decision Tree | 84.63% | 62.76% | 32.72% | 34.48% | 33.58% | 0.2490 |
| KNN | **89.55%** | 74.04% | **56.63%** | 30.82% | 39.92% | 0.3667 |
| Naive Bayes | 80.82% | 77.61% | 31.99% | 62.39% | 42.29% | 0.3478 |
| Random Forest | 87.02% | **80.91%** | 44.44% | 60.78% | **51.34%** | **0.4478** |

---

## Model-wise Observations

| ML Model Name                       | Observation about model performance                                                                                                                                                                                                                                                                                                                                                                               |
| ----------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Logistic Regression**             | Logistic Regression achieved an accuracy of **83.03%** and an AUC of **80.02%**. It achieved the **highest recall of 64.55%**, indicating that it was effective at identifying customers who actually subscribed to a term deposit. Its F1 Score was **46.15%** and MCC was **0.3928**. Although its accuracy was lower than KNN and Random Forest, it performed well in identifying the minority positive class. |
| **Decision Tree**                   | Decision Tree achieved an accuracy of **84.63%**. However, its AUC of **62.76%**, F1 Score of **33.58%**, and MCC of **0.2490** were relatively low. Its recall of **34.48%** indicates that it missed a considerable number of customers who actually subscribed. Therefore, Decision Tree was not among the strongest models for this dataset.                                                                  |
| **KNN**                             | KNN achieved the **highest accuracy of 89.55%** and the **highest precision of 56.63%**. However, its recall was only **30.82%**, meaning that it missed a large proportion of actual positive customers. Since the dataset is highly imbalanced, the high accuracy should be interpreted carefully.                                                                                                              |
| **Naive Bayes**                     | Naive Bayes achieved an accuracy of **80.82%** and a recall of **62.39%**, showing good ability to identify positive-class customers. However, its precision was only **31.99%**, resulting in more false-positive predictions. Its F1 Score of **42.29%** and MCC of **0.3478** were lower than those of Random Forest and Logistic Regression.                                                                  |
| **Random Forest (Ensemble)**        | Random Forest achieved an accuracy of **87.02%** and the **highest AUC of 80.91%**. It achieved a recall of **60.78%**, F1 Score of **51.34%**, and the **highest MCC of 0.4478**. It provides the best overall balance between identifying positive customers and limiting incorrect predictions. Therefore, Random Forest was selected as the overall best-performing model.                                    |
| **Overall Winner for your dataset** | **Random Forest** is the overall winner. It achieved the highest **AUC (80.91%)**, **F1 Score (51.34%)**, and **MCC (0.4478)**, while maintaining a strong recall of **60.78%**. Although KNN achieved higher accuracy (**89.55%**), its recall was only **30.82%**. Because the dataset is highly imbalanced, Random Forest provides a more balanced and reliable overall performance.                           |


## 7. Overall Winner

### Random Forest

**Random Forest was selected as the overall best-performing model
for this dataset.**

It achieved the highest:

- AUC: **80.91%**
- F1 Score: **51.34%**
- MCC: **0.4478**

It also achieved a strong Recall of **60.78%**.

The dataset is highly imbalanced, with only 11.27% positive
observations. Therefore, accuracy alone is not sufficient to
determine the best model.

Although KNN achieved the highest accuracy (89.55%) and precision
(56.63%), its recall was only 30.82%, indicating that it missed a
substantial number of customers who actually subscribed to a term
deposit.

Logistic Regression achieved the highest recall (64.55%), but
Random Forest provided a better overall balance across AUC, Recall,
F1 Score, and MCC.

Therefore, Random Forest is selected as the overall winner for this
classification problem.

---

## How to Run
```
pip install -r requirements.txt

python -m backend.data_preprocessing
python -m backend.train
python -m backend.evaluate

streamlit run app.py 
```
---
## Streamlit Application

