"""
TrainRFSModel.py

Trains the final pipeline for predicting Relapse-Free Survival (RFS)
using the full training dataset (TrainDataset2025.csv), and saves the
trained model to models/rfs_final_model.pkl.

Expected project structure (relative paths):
- data/TrainDataset2025.csv
- models/   
- rfs/TrainRFSModel.py 

Run this script from the project root, for example:
    python rfs/TrainRFSModel.py
"""

import os
import numpy as np
import pandas as pd
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.base import clone

RANDOM_STATE = 42

class MaskedFeatureSelector(BaseEstimator, TransformerMixin):
    """
    LASSO-based feature selector that ensures certain features are always kept.
    Features whose names contain any of the specified keywords will always be retained,
    regardless of their LASSO coefficients.

    Parameters:
    - feature_names: array-like of shape (n_features,)
        Names of the features corresponding to the columns of the input matrix X.
    - alpha: float, default=0.1
        Regularization strength for LASSO.
    - random_state: int, default=42
        Random seed for reproducibility.
    - must_keep_keywords: list of str, default=["ER", "HER2", "Gene"]
        Keywords to identify features that must always be kept.
    
    Methods:
    - fit(X, y): Fits the LASSO model and determines which features to keep
    - transform(X): Transforms the input matrix X by selecting the chosen features

    Returns:
    - Transformed feature matrix with selected features only.
    """

    def __init__(
        self,
        feature_names,
        alpha=0.1,
        random_state=42,
        must_keep_keywords=None,
    ):
        # feature_names is an array of strings corresponding to the columns of the preprocessed feature matrix.
        self.feature_names = np.array(feature_names)
        self.alpha = alpha
        self.random_state = random_state
        if must_keep_keywords is None:
            must_keep_keywords = ["ER", "HER2", "Gene"]
        self.must_keep_keywords = must_keep_keywords

    def fit(self, X, y=None):
        # Fit LASSO on the preprocessed features
        lasso = Lasso(
            alpha=self.alpha,
            max_iter=10000,
            random_state=self.random_state,
        )
        lasso.fit(X, y)

        coef_abs = np.abs(lasso.coef_)
        # Basic LASSO selection: non-zero coefficients
        mask = coef_abs != 0

        # If LASSO zeroes everything, keep all features
        if not mask.any():
            mask = np.ones_like(mask, dtype=bool)

        # Force ER / HER2 / Gene-related features to be kept
        for i, name in enumerate(self.feature_names):
            if any(keyword in name for keyword in self.must_keep_keywords):
                mask[i] = True

        self.mask = mask
        return self

    def transform(self, X):
        # Apply stored boolean mask
        return X[:, self.mask]


def load_data(train_path: str = "../data/TrainDataset2025.csv"):
    """
    Load and prepare the training data for RFS prediction.

    Steps:
    - Read CSV with 999 treated as missing values
    - Extract target: 'RelapseFreeSurvival (outcome)'
    - Drop 'ID' and both outcome columns from features
    - Return X (features) and y (target)
    """
    # Treat 999 as missing
    df = pd.read_csv(train_path, na_values=999)

    # Target: Relapse-Free Survival
    target_col = "RelapseFreeSurvival (outcome)"
    y = df[target_col]

    # Features: drop ID and both outcomes
    X = df.drop(columns=["ID", "pCR (outcome)", "RelapseFreeSurvival (outcome)"])

    return X, y


def build_rfs_pipeline(X_sample: pd.DataFrame) -> Pipeline:
    """
    Build the final RFS pipeline:

    1. Preprocessing
       - Numeric features: median imputation + StandardScaler
       - Categorical features: most-frequent imputation + OneHotEncoder(drop='first')

    2. L1-based feature selection with constraints
       - A custom MaskedFeatureSelector:
         * Fits LASSO(alpha=0.1) on the preprocessed features
         * Builds a mask of features with non-zero coefficients
         * Forces any feature whose name contains 'ER', 'HER2', or 'Gene'
           to be retained, as required by the brief.

    3. Support Vector Regression (SVR)
       - RBF kernel with hyperparameters:
         C=10, gamma=0.1, kernel='rbf'
       (best found during model development)

    The resulting pipeline takes raw feature columns and outputs RFS predictions.
    """

    # Categorical features (clinical / histology / gene markers)
    categorical_features = [
        "ER", "PgR", "HER2", "TrippleNegative", "ChemoGrade",
        "Proliferation", "HistologyType", "LNStatus", "TumourStage", "Gene"
    ]

    # All other columns are treated as numeric
    numeric_features = [col for col in X_sample.columns if col not in categorical_features]

    # Numeric: median imputation + scaling
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    # Categorical: most frequent imputation + one-hot encoding
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first"))
    ])

    # Column-wise preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )

    # Get feature names after preprocessing for use in feature selector
    preproc_for_names = clone(preprocessor)
    preproc_for_names.fit(X_sample)
    feature_names = preproc_for_names.get_feature_names_out()

    # LASSO-based feature selector that keeps ER / HER2 / Gene features
    feature_selector = MaskedFeatureSelector(
        feature_names=feature_names,
        alpha=0.1,
        random_state=RANDOM_STATE,
        must_keep_keywords=["ER", "HER2", "Gene"],
    )

    # Final SVR with tuned hyperparameters from development
    svr_final = SVR(
        C=10,
        gamma=0.1,
        kernel="rbf"
    )

    # Full pipeline: raw X -> preprocess -> LASSO FS (with forced keep) -> SVR
    final_rfs_pipeline = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("feature_select", feature_selector),
        ("model", svr_final),
    ])

    return final_rfs_pipeline


def main():
    # Load training data
    X, y = load_data()
    print(f"[INFO] Loaded training data: X shape = {X.shape}, y shape = {y.shape}")

    # Build final pipeline (uses X to infer feature names for FS)
    pipeline = build_rfs_pipeline(X)

    # Fit on training data
    print("[INFO] Training final RFS pipeline on full dataset...")
    pipeline.fit(X, y)
    print("[INFO] Training completed.")

    # Ensure models directory exists
    os.makedirs("models", exist_ok=True)
    model_path = os.path.join("models", "rfs_final_model.pkl")

    # Save trained pipeline
    joblib.dump(pipeline, model_path)
    print(f"[INFO] Saved trained RFS model to: {model_path}")


if __name__ == "__main__":
    main()