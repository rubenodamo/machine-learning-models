"""
TrainPCRModel.py

Trains the final pipeline for predicting pathological complete response (pCR)
using the full training dataset (TrainDataset2025.csv), and saves the trained
model to models/pcr_final_model.pkl.

Expected project structure (relative paths):
- data/TrainDataset2025.csv
- models/
- pcr/TrainPCRModel.py

Run this script from the project root, for example:
    python pcr/TrainPCRModel.py
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.base import BaseEstimator, MetaEstimatorMixin, TransformerMixin


RANDOM_STATE = 42

class ForcedInclusionSelector(BaseEstimator, MetaEstimatorMixin, TransformerMixin):
    """
    A feature selector that forces inclusion of specified vital features
    regardless of the underlying selector's choice.
    
    Parameters:
    - selector: An instance of a feature selector (e.g., SelectKBest, Select
      FromModel).
    - vital_indices: List of indices of features that must be included.
    
    Methods:
    - fit(X, y): Fits the underlying selector.
    - transform(X): Transforms the feature matrix to include vital features.
    - get_support(indices=False): Returns a boolean mask indicating which features are selected.
    
    Returns:
    - Transformed feature matrix with vital features included.
    """
    def __init__(self, selector, vital_indices):
        self.selector = selector
        self.vital_indices = vital_indices

    def fit(self, X, y=None):
        self.selector.fit(X, y)
        return self

    def transform(self, X):
        mask = self.selector.get_support().copy()
        if self.vital_indices:
            mask[self.vital_indices] = True
        return X[:, mask]

    def get_support(self, indices=False):
        mask = self.selector.get_support().copy()
        if self.vital_indices:
            mask[self.vital_indices] = True
        return mask if not indices else np.where(mask)[0]


def load_data(train_path: str = "assignment-2/data/TrainDataset2025.csv"):
    """
    Load and clean the training data for pCR prediction.

    Steps:
    - Read CSV
    - Replace 999 with NaN
    - Drop 'ID' and 'RelapseFreeSurvival (outcome)'
    - Drop rows with missing 'pCR (outcome)'
    - Return features X and target y
    """
    df = pd.read_csv(train_path)

    # Replace sentinel value with NaN
    df = df.replace(999, np.nan)

    # Drop columns not used for PCR modelling
    cols_to_drop = ['ID', 'RelapseFreeSurvival (outcome)']
    df = df.drop(columns=cols_to_drop, errors='ignore')

    # Keep only rows with observed pCR labels
    df = df.dropna(subset=["pCR (outcome)"])

    # Separate features and target
    X = df.drop(columns=["pCR (outcome)"])
    y = df["pCR (outcome)"]

    return X, y


def build_pipeline(X_sample: pd.DataFrame) -> Pipeline:
    """
    Build the final preprocessing + MI + Logistic Regression pipeline.

    The feature grouping and preprocessing must match the development notebook:
    - Numeric features: median imputation + StandardScaler
    - Nominal categorical: KNN imputation + OneHotEncoder
    - Ordinal/binary categorical: KNN imputation (kept numeric)
    - Mutual Information: SelectKBest with mutual_info_classif, k=best found
    - Classifier: Logistic Regression with tuned hyperparameters and class_weight="balanced"
    """

    # Define feature groups
    categorical_cols = [
        'ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade',
        'Proliferation', 'HistologyType', 'LNStatus', 'TumourStage', 'Gene'
    ]

    # Nominal categorical features to be one-hot encoded
    categorical_nominal_cols = ['HistologyType']

    # Ordinal/binary categorical features kept as numeric
    categorical_ordinal_cols = [
        'Proliferation', 'LNStatus', 'TumourStage',
        'Gene', 'ER', 'PgR', 'HER2', 'TrippleNegative', 'ChemoGrade'
    ]

    # All remaining features are treated as numeric
    numeric_cols = [col for col in X_sample.columns if col not in categorical_cols]

    # Numeric: median imputation + scaling
    numeric_transformer_knn = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    # Nominal categorical: KNN imputation + one-hot encoding
    categorical_nominal_transformer_knn = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=7)),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])

    # Ordinal/binary categorical: KNN imputation only
    categorical_ordinal_transformer_knn = Pipeline(steps=[
        ('imputer', KNNImputer(n_neighbors=7))
    ])

    # Column-wise preprocessing
    preprocessor_knn = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer_knn, numeric_cols),
            ('cat_nom', categorical_nominal_transformer_knn, categorical_nominal_cols),
            ('cat_ord', categorical_ordinal_transformer_knn, categorical_ordinal_cols)
        ]
    )

    # Fit preprocessor to get feature names and vital feature indices
    preprocessor_knn.fit_transform(X_sample)
    feature_names = preprocessor_knn.get_feature_names_out()
    vital_features = ['Gene', 'ER', 'HER2']
    vital_indices = [i for i, name in enumerate(feature_names) if any(v in name for v in vital_features)]
    
    # Logistic Regression with tuned hyperparameters
    log =  LogisticRegression(
        max_iter=40000, 
        class_weight="balanced", 
        random_state=RANDOM_STATE, 
        penalty="l2", 
        C=25,
        solver="saga")

    # Full pipeline: preprocessing -> MI -> Logistic Regression
    pipeline = Pipeline(steps=[
        ("preprocess", preprocessor_knn),
        ("mi", ForcedInclusionSelector(
        selector=SelectKBest(score_func=mutual_info_classif, k=15),
        vital_indices=vital_indices
        )), 
        ("model", log)
    ])

    return pipeline


def main():
    # 1. Load cleaned training data
    X, y = load_data()

    print(f"[INFO] Loaded training data with {X.shape[0]} samples and {X.shape[1]} features.")

    # 2. Build the final pipeline
    pipeline = build_pipeline(X)

    # 3. Fit on ALL available training data (no validation split here)
    print("[INFO] Fitting final PCR pipeline on full training set...")
    pipeline.fit(X, y)
    print("[INFO] Training complete.")

    # 4. Ensure models/ directory exists and save the trained pipeline
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "pcr_final_model.pkl")

    joblib.dump(pipeline, model_path)
    print(f"[INFO] Saved trained PCR model to: {model_path}")


if __name__ == "__main__":
    main()