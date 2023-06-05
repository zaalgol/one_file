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
from datetime import datetime

TEST_SIZE = 0.3
RANDOM_STATE = 42
HYPER_PARAMS_TRIALS = 50
CSV_PATH = "raw_datasets/titanic3.csv"
OUTPUT_MODEL_PATH = "output/model_output.pkl"
OUTPUT_DATA_PATH = f"output/dataset_with_predictions_classifier_{datetime.now().strftime('%m-%d-%Y_%H-%M-%S')}.csv"
TARGET_COL = "survived"
CAT_COLS = ["sex", "cabin"]
NUM_COLS = ["Pclass", "age", "SibSp", "Parch", "Fare"]

class XGBoostClassifier:
    def __init__(self, csv_path=CSV_PATH, target_col=TARGET_COL, cat_cols=CAT_COLS, num_cols=NUM_COLS,
                 output_data_path=OUTPUT_DATA_PATH):
        self.csv_path = csv_path
        self.target_col = target_col
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.output_data_path = output_data_path

        self.df = None
        self.preprocessor = None
        self.pipeline = None
        self.study = None

        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    def load_data(self):
        self.df = pd.read_csv(self.csv_path)

    # def clean_data(self):
    #     pass
    #
    # def feature_engineering(self):
    #     pass

    def split_data(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.df[self.cat_cols + self.num_cols],
                                                                                self.df[self.target_col],
                                                                                test_size=TEST_SIZE, random_state=RANDOM_STATE)

    def optimize_params_xgb(self, trial):
        max_depth = trial.suggest_int("max_depth", 2, 20)
        reg_alpha = trial.suggest_float("reg_alpha", 0.1, 5.0)
        reg_lambda = trial.suggest_float("reg_lambda", 0.1, 5.0)
        n_estimators = trial.suggest_int("n_estimators", 10, 1000)
        learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
        gamma = trial.suggest_int("gamma", 0, 5)
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', XGBClassifier(n_estimators=n_estimators, max_depth=max_depth,
                                                                     learning_rate=learning_rate, reg_alpha=reg_alpha,
                                                                     gamma=gamma, reg_lambda=reg_lambda))])

        self.pipeline.fit(self.X_train, self.y_train)
        return roc_auc_score(self.y_test, self.pipeline.predict_proba(self.X_test)[:, 1])

    def train(self):
        num_preprocessor = SimpleImputer(strategy='median')
        cat_preprocessor = Pipeline(steps=[('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                                           ('onehot', OneHotEncoder(handle_unknown='ignore'))])
        self.preprocessor = ColumnTransformer(transformers=[('num', num_preprocessor, self.num_cols),
                                                            ('cat', cat_preprocessor, self.cat_cols)])

        self.study = optuna.create_study(direction="maximize")
        self.study.optimize(self.optimize_params_xgb, n_trials=HYPER_PARAMS_TRIALS)
        self.pipeline = Pipeline(steps=[('preprocessor', self.preprocessor),
                                        ('classifier', XGBClassifier(**self.study.best_params))])
        self.pipeline.fit(self.X_train, self.y_train)

    @staticmethod
    def evaluate(pred, true):
        roc_auc = roc_auc_score(true, pred)
        accuracy = accuracy_score(true, pred)
        precision = precision_score(true, pred)
        recall = recall_score(true, pred)
        f1 = f1_score(true, pred)
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
        # Load the model from the pkl file
        with open(model_path, 'rb') as file:
            self.pipeline = pickle.load(file)

        # Load the dataset
        df = pd.read_csv(dataset_path)

        # Make predictions on the dataset
        y_pred = self.pipeline.predict(df[self.cat_cols + self.num_cols])

        # Save the predictions
        self.save_predictions(y_pred, df)


def main():
    # Instantiate the XGBoostClassifier
    classifier = XGBoostClassifier(csv_path="raw_datasets/titanic3.csv",
                                   target_col="survived",
                                   cat_cols=["sex", "cabin"],
                                   num_cols=["age", "fare"],
                                   output_data_path=OUTPUT_DATA_PATH)

    # Load the data
    classifier.load_data()

    # split to train and test
    classifier.split_data()

    # Train the classifier
    classifier.train()

    y_pred = classifier.pipeline.predict(classifier.df[classifier.cat_cols + classifier.num_cols])
    y_true = classifier.df[classifier.target_col]
    y_train_pred = classifier.pipeline.predict(classifier.X_train)
    y_test_pred = classifier.pipeline.predict(classifier.X_test)

    # Evaluate the model
    print("Y_all:")
    classifier.evaluate(y_pred, y_true)
    print("y_train:")
    classifier.evaluate(y_train_pred, classifier.y_train)
    print("y_test:")
    classifier.evaluate(y_test_pred, classifier.y_test)

    # Save the dataset with predictions
    classifier.save_predictions(y_pred)

    # Save the model
    pickle.dump(classifier.pipeline, open(OUTPUT_MODEL_PATH, 'wb'))

    # run for a trained model!
    # classifier.load_and_predict(OUTPUT_MODEL_PATH, CSV_PATH)


if __name__ == "__main__":
    main()
