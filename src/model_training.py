
import os
import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression 
from src.logger import get_logger
from src.custom_exception import CustomException
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

logger = get_logger(__name__)

class ModelTrainer:

    def __init__(self, input_path: str, model_path: str):
        self.input_path = input_path
        self.model_path = model_path
        self.clf = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

        os.makedirs(self.model_path, exist_ok=True)
        logger.info("Model Trainer initialized....")

    def load_data(self):
        try:
            self.X_train = joblib.load(os.path.join(self.input_path, 'X_train.pkl'))
            self.X_test = joblib.load(os.path.join(self.input_path, 'X_test.pkl'))
            self.y_train = joblib.load(os.path.join(self.input_path, 'y_train.pkl'))
            self.y_test = joblib.load(os.path.join(self.input_path, 'y_test.pkl'))

            logger.info("Data preprocess loaded successfully for model training.")
        
        except Exception as e:
            logger.error(f"Error in loading data for model training {e}")
            raise CustomException("Error in loading data for model training", e)
        
    def train_model(self):
        try:
            self.clf = LogisticRegression(random_state=42, max_iter=1000)
            self.clf.fit(self.X_train, self.y_train)

            joblib.dump(self.clf, os.path.join(self.model_path, 'model.pkl'))

            logger.info("Model trained and saved successfully....")

        except Exception as e:
            logger.error(f"Error in training and saving model {e}")
            raise CustomException("Error in training and saving model", e)

    def evaluate_model(self):
        try:
            y_pred = self.clf.predict(self.X_test)

            accuracy = accuracy_score(self.y_test, y_pred)
            precision = precision_score(self.y_test, y_pred, average='weighted')
            recall = recall_score(self.y_test, y_pred, average='weighted')
            f1 = f1_score(self.y_test, y_pred, average='weighted')

            logger.info(f"Model Evaluation Metrics: Accuracy: {accuracy}, Precision: {precision}, Recall: {recall}, F1-Score: {f1}")
            logger.info("Model evaluated successfully....")

        except Exception as e:
            logger.error(f"Error in evaluating model {e}")
            raise CustomException("Error in evaluating model", e)

    def run(self):
        self.load_data()
        self.train_model()
        self.evaluate_model()

if __name__ == "__main__":
    trainer = ModelTrainer(input_path="artifacts/processed", model_path="artifacts/model")
    trainer.run()
