# Rachel 07/2019
# Training LogReg model on APS dataset with IBM privlib

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import make_scorer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from diffprivlib.models.logistic_regression import LogisticRegression
import utils
import time



start_time = time.perf_counter()
df_training = pd.read_csv('data/aps_failure_training_set.csv', na_values='na')
df_test = pd.read_csv('data/aps_failure_test_set.csv', na_values='na')
df_X_eval = pd.read_csv('data/aps_failure_evaluation_set.csv', na_values='na')
y_eval = pd.read_csv('data/aps_failure_evaluation_set_target.csv')

# Convert string labels to numerical labels
label_mapping = {'neg': 0, 'pos': 1}
df_training['class'] = df_training['class'].map(label_mapping)
df_test['class'] = df_test['class'].map(label_mapping)
y_eval = y_eval['class'].map(label_mapping)

# Extract feature matrix X and target label vector Y
df_X_train = df_training.drop('class', axis=1)
y_train = df_training['class']
df_X_test = df_test.drop('class', axis=1)
y_test = df_test['class']

classes = np.unique(y_test)


def feature_idxes_with_fair_nan_ratio(df, max_nan_ratio=0.8):
    nan_ratio = df.isnull().sum() / df.shape[0]
    leq_max_nan_ratio = nan_ratio <= max_nan_ratio

    print("High NaN ratio features:", (~leq_max_nan_ratio).sum())
    return leq_max_nan_ratio.index[leq_max_nan_ratio]


def feature_idxes_with_fair_corr(df, max_corr_coef=0.99):
    corr = df.corr()
    corr_upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(np.bool))
    high_corr = (corr_upper > max_corr_coef).any()

    print("Highly correlated features:", high_corr.sum())
    return high_corr.index[~high_corr]


def feature_idxes_with_fair_var(df, min_var=1e-3):
    gt_min_var = df.var() > min_var

    print("Low variance features:", (~gt_min_var).sum())
    return gt_min_var.index[gt_min_var]


def standard_scaling(df_train, df_tests):
    scaler = StandardScaler()
    df_train_std = scaler.fit_transform(df_train)
    df_tests_std = [scaler.transform(df) for df in df_tests]

    return df_train_std, df_tests_std


def fill_missing_values(X_train, X_tests,
                        missing_values=np.nan,
                        strategy='most_frequent'):
    nan_imputer = SimpleImputer(missing_values=missing_values,
                                strategy=strategy)
    X_train_imputed = nan_imputer.fit_transform(X_train)
    X_tests_imputed = [nan_imputer.transform(X) for X in X_tests]

    print("Fill missing values with '%s'" % strategy)

    return X_train_imputed, X_tests_imputed


def sample_preprocessing(df_train, df_tests):
    # Feature standardization (remove the mean and scale to unit variance)
    X_train_std, X_tests_std = standard_scaling(df_train, df_tests)

    # Missing value imputation
    X_train_imputed, X_tests_imputed = fill_missing_values(X_train_std, X_tests_std)

    return X_train_imputed, X_tests_imputed


def preprocessing(df_X_train, df_X_tests):
    keep_cols = set(df_X_train.columns)

    # Only keep features that have reasonable amount of missing values
    fair_nan_ratio_idxes = feature_idxes_with_fair_nan_ratio(df_X_train[keep_cols], 0.8)
    keep_cols = keep_cols.intersection(fair_nan_ratio_idxes)

    # Only keep features that are not too highly correlated
    fair_corr_idxes = feature_idxes_with_fair_corr(df_X_train[keep_cols], 0.99)
    keep_cols = keep_cols.intersection(fair_corr_idxes)

    # Do not keep features with low variance
    fair_var_idxes = feature_idxes_with_fair_var(df_X_train[keep_cols], min_var=1e-3)
    keep_cols = keep_cols.intersection(fair_var_idxes)

    df_X_train_col_filtered = df_X_train[keep_cols]
    df_X_tests_col_filtered = [df[keep_cols] for df in df_X_tests]

    # Modify sample values without modifying the shape of the dataset
    X_train_processed, X_tests_processed = sample_preprocessing(df_X_train_col_filtered,
                                                                df_X_tests_col_filtered)

    return X_train_processed, X_tests_processed, list(keep_cols)



X_train, [X_test, X_eval], keep_cols = preprocessing(df_X_train, [df_X_test, df_X_eval])
print(X_train.shape, X_test.shape)
#epsilons = [float('inf')]
epsilons = [5]#np.logspace(-2, 2, 50)
# output = open("aps_dataset_ibm_dp_data_norm_1000_70k_images.txt", "w+")
for epsilon in epsilons:
    # can't make a new scoring function, this option is not allow in IBM diffprivlib
    logreg_w_dp  = LogisticRegression(epsilon=epsilon, data_norm=500, max_iter=20)
    logreg_w_dp.fit(X_train, y_train)
    # score_dp = logreg_w_dp.score(X_test, y_test)
    # print('Accuracy = ', score_dp)
    accuracy, recall, precision, auc = utils.predict_score(logreg_w_dp, X_test, y_test)
    # output.write("%.3f \t %.3f \t %.3f \n" % (epsilon, recall, precision))
    print(accuracy, recall, precision, auc)

print('Total time = ', time.perf_counter() - start_time, ' seconds')
# output.close()