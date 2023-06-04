import optuna
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
import os

# CONFIG
print(os.getcwd())
TEST_SIZE = 0.3
RANDOM_STATE = 42
HYPER_PARAMS_TRIALS = 50
CSV_PATH = "raw_datasets/titanic3.csv"
OUTPUT_PATH = "output/model_output.pkl"
TARGET_COL = "survived"
CAT_COLS = ["sex", "cabin"]
NUM_COLS = ["age", "fare"]
df = pd.read_csv(CSV_PATH)


def optimize_params_xgb(trial):
    max_depth = trial.suggest_int("max_depth", 3, 20)
    reg_alpha = trial.suggest_float("reg_alpha", 0.1, 2.0)
    reg_lambda = trial.suggest_float("reg_lambda", 0.1, 2.0)
    n_estimators = trial.suggest_int("n_estimators", 10, 1000)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                            learning_rate=learning_rate, reg_alpha=reg_alpha,
                                                            reg_lambda=reg_lambda))])
    X_train, X_val, y_train, y_val = train_test_split(df[CAT_COLS + NUM_COLS], df[TARGET_COL],
                                                      test_size=TEST_SIZE, random_state=RANDOM_STATE)
    pipeline.fit(X_train, y_train)
    return roc_auc_score(y_val, pipeline.predict_proba(X_val)[:, 1])


num_preprocessor = SimpleImputer(strategy='median')
cat_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                   ('onehot', OneHotEncoder(handle_unknown='ignore'))])
preprocessor = ColumnTransformer(
    transformers=[('num', num_preprocessor, NUM_COLS), ('cat', cat_preprocessor, CAT_COLS)])
study = optuna.create_study(direction="maximize")
study.optimize(optimize_params_xgb, n_trials=HYPER_PARAMS_TRIALS)
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', XGBClassifier(**study.best_params))])
pipeline.fit(df[CAT_COLS + NUM_COLS], df[TARGET_COL])

# evaluate model scores
y_pred = pipeline.predict(df[CAT_COLS + NUM_COLS])
y_true = df[TARGET_COL]
roc_auc = roc_auc_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

# print scores
print("ROC AUC score:", roc_auc)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# save dataset_with_predictions
df_with_predictions = df.copy()
df_with_predictions['y_pred'] = y_pred
df_with_predictions.to_csv("output/dataset_with_predictions_classifier.csv", index=False)

# save model
pickle.dump(pipeline, open(OUTPUT_PATH, 'wb'))
