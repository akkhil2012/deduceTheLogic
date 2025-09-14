# impute_demo.py
# -------------------------------------------------
# Demonstrates:
# 1) Creating a toy dataset with mixed types
# 2) Injecting missingness
# 3) Imputing with:
#    - SimpleImputer (mean/median/most_frequent)
#    - KNNImputer (k-NN based)
#    - IterativeImputer (multivariate)
# 4) Evaluating imputation error on purposely-masked cells

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder

# enable experimental IterativeImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

rng = np.random.default_rng(42)

# -----------------------------
# 1) Synthetic dataset (mixed types)
# -----------------------------
n = 500
df = pd.DataFrame({
    "age": rng.normal(35, 10, n).clip(18, 70),              # numeric
    "income": rng.lognormal(mean=10, sigma=0.5, size=n),    # numeric (skewed)
    "tenure_years": rng.integers(0, 20, n),                 # numeric (discrete)
    "city": rng.choice(["Mumbai", "Bengaluru", "Delhi", "Pune"], size=n),  # categorical
    "segment": rng.choice(["Retail", "SMB", "Enterprise"], size=n)         # categorical
})

# Keep a "clean" copy to compare with later
df_clean = df.copy()

# -----------------------------
# 2) Inject missingness at random
# -----------------------------
def inject_missing(series, missing_rate=0.15):
    mask = rng.random(len(series)) < missing_rate
    s = series.copy()
    s[mask] = np.nan
    return s, mask

missing_info = {}
df_noisy = pd.DataFrame(index=df.index)

for col in df.columns:
    df_noisy[col], mask = inject_missing(df[col], missing_rate=0.15)
    missing_info[col] = mask  # boolean mask of where we hid values

numeric_features = ["age", "income", "tenure_years"]
categorical_features = ["city", "segment"]

# -----------------------------
# 3) Three imputers: simple, KNN (numeric), iterative
#    (We’ll build pipelines with ColumnTransformer)
# -----------------------------

# 3a) SimpleImputer: mean for numeric, most_frequent for categorical
simple_ct = ColumnTransformer(
    transformers=[
        ("num", SimpleImputer(strategy="median"), numeric_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features),
    ],
    remainder="drop"
)

# 3b) KNNImputer for numeric + most_frequent for categorical
# Note: KNNImputer works only on numeric columns, so we impute cats separately.
# We’ll run two-step: first impute numeric with KNN on a copy, then plug into ColumnTransformer for cats.
def knn_impute_numeric(df_in, numeric_cols, n_neighbors=5):
    temp = df_in[numeric_cols].copy()
    knn = KNNImputer(n_neighbors=n_neighbors, weights="distance")
    imputed = knn.fit_transform(temp)
    out = df_in.copy()
    out[numeric_cols] = imputed
    return out

# 3c) IterativeImputer: multivariate numeric imputation (MICE-like) + most_frequent for categorical
def iterative_impute_numeric(df_in, numeric_cols, max_iter=15):
    it = IterativeImputer(random_state=0, max_iter=max_iter)
    imputed = it.fit_transform(df_in[numeric_cols])
    out = df_in.copy()
    out[numeric_cols] = imputed
    return out

# -----------------------------
# 4) Evaluate imputation error only on cells we masked
#    - Numeric: RMSE
#    - Categorical: accuracy on masked cells
# -----------------------------
# Helper function to print before/after comparison
def print_imputation_comparison(original, imputed, mask, column_name, max_rows=5):
    """Print comparison of original vs imputed values for a column."""
    if mask.sum() == 0:  # No missing values in this column
        print(f"\nNo missing values in column: {column_name}")
        return
        
    print(f"\n=== Imputation Comparison: {column_name} ===")
    print(f"Number of imputed values: {mask.sum()}")
    
    # Create a DataFrame for comparison
    comparison = pd.DataFrame({
        'Original': original[mask],
        'Imputed': imputed[mask]
    })
    
    # Add a flag if the imputed value is different from original
    if column_name in ['city', 'segment']:  # Categorical columns
        comparison['Changed'] = comparison['Original'] != comparison['Imputed']
    else:  # Numeric columns
        comparison['Changed'] = ~np.isclose(comparison['Original'], comparison['Imputed'], rtol=1e-5, equal_nan=True)
    
    # Print first few rows with changes
    changed = comparison[comparison['Changed']]
    if len(changed) > 0:
        print("\nSample of changed values:")
        print(changed.head(max_rows))
    else:
        print("\nNo changes in imputed values")
    
    # Print statistics for numeric columns
    if column_name not in ['city', 'segment']:
        diff = comparison['Imputed'] - comparison['Original']
        print("\nImputation Statistics:")
        print(f"  Mean change: {diff.mean():.4f}")
        print(f"  Max change:  {diff.abs().max():.4f}")
        print(f"  Min change:  {diff.abs().min():.4f}")

def eval_imputation(df_original, df_imputed, masks, numeric_cols, categorical_cols, print_comparison=True):
    metrics = {}

    # Print comparison for each numeric column
    if print_comparison:
        print("\n" + "="*50)
        print("IMPUTATION COMPARISON")
        print("="*50)
        
        # Print numeric comparisons first
        for col in numeric_cols + categorical_cols:
            if col in df_original.columns and col in df_imputed.columns:
                print_imputation_comparison(df_original[col], df_imputed[col], masks[col], col)

    # Numeric RMSE (only where masked=True)
    for col in numeric_cols:
        mask = masks[col]
        if mask.sum() > 0:  # Only calculate if there are missing values
            y_true = df_original.loc[mask, col].astype(float)
            y_pred = df_imputed.loc[mask, col].astype(float)
            mse = mean_squared_error(y_true, y_pred)
            rmse = np.sqrt(mse)  # Calculate RMSE manually
            metrics[f"RMSE[{col}]"] = rmse
        else:
            metrics[f"RMSE[{col}]"] = 0.0

    # Categorical accuracy on masked cells
    for col in categorical_cols:
        mask = masks[col]
        if mask.sum() > 0:  # Only calculate if there are missing values
            y_true = df_original.loc[mask, col].astype(str)
            y_pred = df_imputed.loc[mask, col].astype(str)
            acc = (y_true == y_pred).mean()
            metrics[f"ACC[{col}]"] = acc
        else:
            metrics[f"ACC[{col}]"] = 1.0  # Perfect accuracy if no missing values
            
    return metrics  # Add this line to return the metrics dictionary

# Helper to print metrics nicely
def print_metrics(title, metrics):
    print(f"\n=== {title} ===")
    for k, v in metrics.items():
        print(f"{k:>15}: {v:.4f}" if isinstance(v, float) else f"{k:>15}: {v}")

# -----------------------------
# A) SIMPLE IMPUTER
# -----------------------------
# Fit transform to get *imputed feature matrix* for modeling.
# But for quality evaluation on raw columns, we also create a "dataframe-imputed" view.
simple_num = SimpleImputer(strategy="median")
simple_cat = SimpleImputer(strategy="most_frequent")

df_simple = df_noisy.copy()
df_simple[numeric_features] = simple_num.fit_transform(df_simple[numeric_features])
df_simple[categorical_features] = simple_cat.fit_transform(df_simple[categorical_features])

simple_metrics = eval_imputation(df_clean, df_simple, missing_info, numeric_features, categorical_features)
print_metrics("SimpleImputer (median/mode)", simple_metrics)

# -----------------------------
# B) KNN IMPUTER (numeric) + mode for categorical
# -----------------------------
df_knn = knn_impute_numeric(df_noisy, numeric_features, n_neighbors=5)
df_knn[categorical_features] = simple_cat.fit_transform(df_knn[categorical_features])

knn_metrics = eval_imputation(df_clean, df_knn, missing_info, numeric_features, categorical_features)
print_metrics("KNNImputer (k=5) + mode for cats", knn_metrics)

# -----------------------------
# C) ITERATIVE IMPUTER (numeric) + mode for categorical
# -----------------------------
df_iter = iterative_impute_numeric(df_noisy, numeric_features, max_iter=15)
df_iter[categorical_features] = simple_cat.fit_transform(df_iter[categorical_features])

iter_metrics = eval_imputation(df_clean, df_iter, missing_info, numeric_features, categorical_features)
print_metrics("IterativeImputer + mode for cats", iter_metrics)

# -----------------------------
# 5) Example: use imputed data in a modeling pipeline
#    (e.g., predicting income from other fields)
# -----------------------------
X = df_noisy.drop(columns=["income"])
y = df_noisy["income"]

# We'll impute within pipeline to avoid leakage:
pipe = Pipeline(steps=[
    ("prep", ColumnTransformer(transformers=[
        ("num", SimpleImputer(strategy="median"), ["age", "tenure_years"]),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical_features)
    ])),
    # For demonstration, a simple linear model would go here (e.g., Ridge)
    # We'll just show the transformed shape:
])

Xt = pipe.fit_transform(X)
print("\nTransformed feature matrix shape (for modeling):", Xt.shape)
