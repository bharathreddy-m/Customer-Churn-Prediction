# Banking Customer Churn Prediction

## Table of Contents
- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Modeling](#modeling)
- [Results](#results)
- [Conclusion](#conclusion)
- [Requirements](#requirements)
- [How to Run](#how-to-run)

## Project Overview
The objective of this project is to predict customer churn in the banking sector using machine learning algorithms. By analyzing customer data, the goal is to identify patterns that lead to customer attrition, allowing banks to implement strategies to retain valuable customers.

## Dataset
The dataset used in this project is the Banking Customer Churn dataset. It contains various features related to customer demographics, account information, and transaction history.

## Data Preprocessing
1. **Loading Data**: The dataset was loaded and initial data types were checked.
2. **Handling Missing Values**: Checked for missing values, filled them appropriately, and handled NaN values.
3. **Removing Duplicates**: Duplicates were identified and removed.
4. **Encoding Categorical Variables**: Categorical variables were transformed using label encoding and one-hot encoding. The 'Surname' column was deleted as it was deemed unnecessary.
5. **Outlier Detection**: Outliers were detected using IQR and Z-score methods and were filled with the mean.
6. **Normality Check**: Data normality was checked using skewness.
7. **Data Visualization**: Various plots (heatmap, histplot, KDE, box plot, Q-Q plot) were used to visualize the data and assess its distribution.
8. **Feature Scaling**: StandardScaler was used for feature scaling.
9. **Data Splitting**: The dataset was split into training (80%) and testing (20%) sets.

## Exploratory Data Analysis (EDA)
- Correlation heatmap was used to analyze feature relationships.
- Histograms, KDE plots, and box plots were employed to visualize distributions and detect potential issues with data distribution.

## Modeling
Multiple machine learning models were implemented, including:
- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Classifier (SVC)
- Decision Tree
- Random Forest
- Gradient Boosting Classification

## Results

### Logistic Regression
- **Training Accuracy**: 73.09%
- **Testing Accuracy**: 72.90%
- **ROC-AUC Score**: 0.7281

**Confusion Matrix (Testing):**
```
[[1222,  372],
 [ 439,  960]]
```

**Classification Report (Testing):**
```
              precision    recall  f1-score   support

         0.0       0.74      0.77      0.75      1594
         1.0       0.72      0.69      0.70      1399

    accuracy                           0.73      2993
   macro avg       0.73      0.73      0.73      2993
weighted avg       0.73      0.73      0.73      2993
```

### KNN Classifier
- **Training Accuracy**: 86.12%
- **Testing Accuracy**: 81.49%

**Confusion Matrix (Testing):**
```
[[1262,  332],
 [ 222, 1177]]
```

**Classification Report (Testing):**
```
              precision    recall  f1-score   support

         0.0       0.85      0.79      0.82      1594
         1.0       0.78      0.84      0.81      1399

    accuracy                           0.81      2993
   macro avg       0.82      0.82      0.81      2993
weighted avg       0.82      0.81      0.82      2993
```

### SVC Classifier
- **Training Accuracy**: 91.33%
- **Testing Accuracy**: 85.33%

**Confusion Matrix (Testing):**
```
[[1325,  269],
 [ 170, 1229]]
```

**Classification Report (Testing):**
```
              precision    recall  f1-score   support

         0.0       0.89      0.83      0.86      1594
         1.0       0.82      0.88      0.85      1399

    accuracy                           0.85      2993
   macro avg       0.85      0.85      0.85      2993
weighted avg       0.86      0.85      0.85      2993
```

### Decision Tree
- **Training Accuracy**: 95.10%
- **Testing Accuracy**: 81.72%

**Confusion Matrix (Testing):**
```
[[1310,  284],
 [ 263, 1136]]
```

**Classification Report (Testing):**
```
              precision    recall  f1-score   support

         0.0       0.83      0.82      0.83      1594
         1.0       0.80      0.81      0.81      1399

    accuracy                           0.82      2993
   macro avg       0.82      0.82      0.82      2993
weighted avg       0.82      0.82      0.82      2993
```

### Random Forest
- **Training Accuracy**: 97.85%
- **Testing Accuracy**: 87.80%

**Confusion Matrix (Testing):**
```
[[1405,  189],
 [ 176, 1223]]
```

**Classification Report (Testing):**
```
              precision    recall  f1-score   support

         0.0       0.89      0.88      0.89      1594
         1.0       0.87      0.87      0.87      1399

    accuracy                           0.88      2993
   macro avg       0.88      0.88      0.88      2993
weighted avg       0.88      0.88      0.88      2993
```

### Gradient Boosting Classification
- **Training Accuracy**: 90.37%
- **Testing Accuracy**: 88.71%

**Confusion Matrix (Testing):**
```
[[1460,  134],
 [ 204, 1195]]
```

**Classification Report (Testing):**
```
              precision    recall  f1-score   support

         0.0       0.88      0.92      0.90      1594
         1.0       0.90      0.85      0.88      1399

    accuracy                           0.89      2993
   macro avg       0.89      0.89      0.89      2993
weighted avg       0.89      0.89      0.89      2993
```

## Conclusion
In this project, several models were implemented to predict customer churn. The Random Forest model performed the best, achieving an accuracy of 87.80% on the testing set. This indicates that the model can effectively identify customers likely to churn, enabling targeted retention strategies.

## Requirements
To run this project, the following Python libraries are required:
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- shap

You can install these packages using pip:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn shap
```

## How to Run
1. Clone this repository:
    ```bash
    git clone <your-github-repo-url>
    ```
2. Navigate to the project directory:
    ```bash
    cd <your-project-directory>
    ```
3. Run the Jupyter Notebook or Python script:
    ```bash
    jupyter notebook <your-notebook>.ipynb
    ```
    or
    ```bash
    python <your-script>.py
    ```
