Imputation Techniques for Missing Data: 



This document provides a visual explanation of the impute_demo.py script, which demonstrates various imputation techniques for handling missing data in a Pandas DataFrame. The script covers creating a synthetic dataset, injecting missing values, and imputing them using SimpleImputer, KNNImputer, and IterativeImputer. It also evaluates the imputation error by comparing the imputed values with the original values.





















1. Synthetic Dataset Creation



The script begins by creating a synthetic dataset using Pandas. This dataset includes a mix of numeric and categorical features:







age: Normally distributed numeric data.



income: Log-normally distributed numeric data (skewed).



tenure_years: Discrete numeric data.



city: Categorical data with choices like "Mumbai", "Bengaluru", "Delhi", and "Pune".



segment: Categorical data with choices like "Retail", "SMB", and "Enterprise".



A clean copy of the DataFrame (df_clean) is stored for later comparison.



2. Injecting Missingness



Missing values are injected randomly into the dataset to simulate real-world scenarios. The inject_missing function introduces missingness at a specified rate (defaulting to 15%). A boolean mask (missing_info) is created to track the locations of the injected missing values, which will be used later to evaluate the imputation performance.



3. Imputation Techniques



The script demonstrates three different imputation techniques:



3a. SimpleImputer



The SimpleImputer is used with different strategies for numeric and categorical features.  Median is used for numeric features, and most frequent is used for categorical features. A ColumnTransformer is used to apply different imputers to different columns.  For categorical features, a OneHotEncoder is also used to convert the categorical features into numeric features.



3b. KNNImputer



The KNNImputer imputes missing numeric values based on the values of the nearest neighbors. Since KNNImputer only works on numeric columns, categorical features are imputed separately using the "most frequent" strategy from SimpleImputer. A function knn_impute_numeric is defined to handle the KNN imputation for numeric columns.



3c. IterativeImputer



The IterativeImputer (also known as Multivariate Imputation by Chained Equations or MICE) imputes missing numeric values using an iterative process. It models each feature with missing values as a function of other features. Similar to KNNImputer, categorical features are imputed separately using the "most frequent" strategy. A function iterative_impute_numeric is defined to handle the iterative imputation for numeric columns.



4. Evaluating Imputation Error



The script evaluates the performance of each imputation technique by comparing the imputed values with the original values (from df_clean) at the locations where missingness was injected.







Numeric Features: Root Mean Squared Error (RMSE) is used to measure the difference between the imputed and original values.



Categorical Features: Accuracy is used to measure the proportion of correctly imputed values.



The eval_imputation function calculates these metrics, and the print_metrics function displays the results in a readable format.  The print_imputation_comparison function prints a sample of the imputed values and their original values for comparison.



5. Modeling Pipeline Example



The script demonstrates how imputed data can be used in a modeling pipeline. A ColumnTransformer is used to apply imputation and one-hot encoding to the features. A Pipeline is then used to combine the ColumnTransformer with a model (not explicitly defined in the script, but a linear model like Ridge is suggested). This pipeline ensures that imputation is performed within the cross-validation loop to avoid data leakage. The shape of the transformed feature matrix is printed to show the result of the preprocessing steps.
