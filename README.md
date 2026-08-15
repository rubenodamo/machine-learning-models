# Machine Learning Prediction Models

## 1 - Predicting Heart Disease
Using patient data that is given in the dataset, a machine learning model was built to predict whether a patient has heart disease, or not, based on the appropriate input features from the dataset. 

The predictions and model descriptions involve:
- exploratory analysis and pre-processing of the data
- building a decision tree model for predicting whether or not someone has heart disease
- neural networks and an ANN for the same classification data

## 2 - Iris Species and House Price Prediction
Using the UCI Iris and Real Estate Valuation datasets, four machine learning algorithms were implemented and compared for both a classification task (predicting flower species) and a regression task (predicting house price).

The predictions and model descriptions involve:
- exploratory data analysis and pre-processing of both datasets
- logistic regression (classification) and linear regression (regression) as baseline models
- a support vector machine (SVM/SVR), decision tree, and multi-layer perceptron (MLP) for each task
- stratified 5-fold cross-validation to evaluate every model, compared by accuracy for classification and mean squared error for regression

_Team: Ruben Odamo, Emmanuela Amune, Desange Nkumu, Michael Yianni, Joshua Gaynor_

## 3 - Breast Cancer Treatment Response and Survival Prediction
Using real-world clinical data covering patient tumour characteristics, biomarkers, and treatment indicators, two end-to-end pipelines were built to predict pathological complete response (PCR) to treatment and relapse-free survival (RFS) after treatment.

The predictions and model descriptions involve:
- data cleaning and preprocessing, including KNN and median/most-frequent imputation, one-hot encoding, and scaling via modular `ColumnTransformer` pipelines
- feature selection and dimensionality reduction, including decision tree importance, mutual information, PCA, LDA, and LASSO, with clinically important features (ER, HER2, Gene) forced to remain
- model development and 5-fold cross-validated comparison, with hyperparameter tuning via `GridSearchCV`
- a final logistic regression model selected for PCR (via mutual information feature selection) and a final SVR (RBF kernel) model selected for RFS (via LASSO feature selection)

_Team: Ruben Odamo, Emmanuela Amune, Desange Nkumu, Michael Yianni, Joshua Gaynor_

> Note: the clinical dataset used for this project is not included, as it isn't the author's to share.
