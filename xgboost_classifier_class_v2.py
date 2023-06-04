import optuna
import pickle
import pandas as pd
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from datetime import datetime

TEST_SIZE = 0.3
RANDOM_STATE = 42
HYPER_PARAMS_TRIALS = 50
CSV_PATH = "raw_datasets/titanic3.csv"
OUTPUT_MODEL_PATH = "output/model_output.pkl"
OUTPUT_DATA_PATH = f"output/dataset_with_predictions_classifier_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
TARGET_COL = "survived"
CAT_COLS = ["sex", "cabin"]
NUM_COLS = ["age", "fare"]

class XGBoostClassifier:
    def __init__(self, csv_path=CSV_PATH, target_col=TARGET_COL, cat_cols=CAT_COLS, num_cols=NUM_COLS,
                 output_data_path=OUTPUT_DATA_PATH):
        self.y_val = None
        self.y_train = None
        self.X_val = None
        self.X_train = None
        self.csv_path = csv_path
        self.target_col = target_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.output_data_path = output_data_path

        self.df = None
        self.preprocessor = None
        self.pipeline = None
        self.study = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)

    def optimize_params_xgb(self, trial):
        max_depth = trial.suggest_int("max_depth", 3, 20)
        reg_alpha = trial.suggest_float("reg_alpha", 0.1, 2.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 2.0)
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)

        # Configure early stopping
        early_stopping_rounds = 20
        eval_set = [(self.X_val, self.y_val)]
        eval_metric = "logloss"

        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                                     learning_rate=learning_rate, reg_alpha=reg_alpha,
                                                                     reg_lambda=reg_lambda))])

        # Train with early stopping
        self.pipeline.fit(self.X_train, self.y_train, classifier__eval_set=eval_set, classifier__early_stopping_rounds=early_stopping_rounds,
                          classifier__eval_metric=eval_metric)

        return roc_auc_score(self.y_val, self.pipeline.predict_proba(self.X_val)[:, 1])

    def train(self):
        num_preprocessor = SimpleImputer(strategy='median')
        cat_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        self.preprocessor = ColumnTransformer(transformers=[('num', num_preprocessor, self.num_cols),
                                                            ('cat', cat_preprocessor, self.cat_cols)])

        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.df[self.cat_cols + self.num_cols],
                                                          self.df[self.target_col],
                                                          test_size=TEST_SIZE, random_state=RANDOM_STATE)

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.optimize_params_xgb, n_trials=HYPER_PARAMS_TRIALS)

        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', XGBClassifier(**self.study.best_params))])

        self.pipeline.fit(self.X_train, self.y_train)

    @staticmethod
    def evaluate(y_pred, y_true):
        roc_auc = roc_auc_score(y_true, y_pred)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        print("ROC AUC score:", roc_auc)
        print("Accuracy:", accuracy)
        print("Precision:", precision)
        print("Recall:", recall)
        print("F1 Score:", f1)

    def save_predictions(self, y_pred, df=None, output_data_path=None):
        if df is None:
            df = self.df.copy()
        df['y_pred'] = y_pred

        if output_data_path is None:
            output_data_path = self.output_data_path
        df.to_csv(output_data_path, index=False)

    def load_and_predict(self, model_path, dataset_path):
        with open(model_path, 'rb') as file:
            self.pipeline = pickle.load(file)

        df = pd.read_csv(dataset_path)
        y_pred = self.pipeline.predict(df[self.cat_cols + self.num_cols])

        self.save_predictions(y_pred, df)


def main():
    classifier = XGBoostClassifier(csv_path="raw_datasets/titanic3.csv",
                                   target_col="survived",
                                   cat_cols=["sex", "cabin"],
                                   num_cols=["age", "fare"],
                                   output_data_path=OUTPUT_DATA_PATH)

    classifier.load_data()
    classifier.train()

    y_pred_train = classifier.pipeline.predict(classifier.df[classifier.cat_cols + classifier.num_cols])
    y_true_train = classifier.df[classifier.target_col]

    print("Training Set Evaluation:")
    classifier.evaluate(y_pred_train, y_true_train)

    y_pred_val = classifier.pipeline.predict(classifier.X_val)
    y_true_val = classifier.y_val

    print("Validation Set Evaluation:")
    classifier.evaluate(y_pred_val, y_true_val)

    classifier.save_predictions(y_pred_train)

    pickle.dump(classifier.pipeline, open(OUTPUT_MODEL_PATH, 'wb'))

    # classifier.load_and_predict(OUTPUT_MODEL_PATH, CSV_PATH)


if __name__ == "__main__":
    main()
