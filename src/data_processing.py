import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from src.logger import get_logger
from src.custom_exception import CustomException
import os

logger = get_logger(__name__)

class DataProcessor:

    def __init__(self, input_path: str, output_path: str):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.features = None

        os.makedirs(self.output_path, exist_ok=True)
        logger.info("DataProcessor initialized with the paths provided.")

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            logger.info("Data loaded successfully.")

        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Error loading data", e)
    
    def preprocess(self):
        try:
            self.df['Timestamp'] = pd.to_datetime(self.df['Timestamp'], errors='coerce')

            categorical_cols = ['Operation_Mode', 'Efficiency_Status']
            for col in categorical_cols:
                self.df[col] = self.df[col].astype('category')

            self.df['Month'] = self.df['Timestamp'].dt.month
            self.df['Day'] = self.df['Timestamp'].dt.day
            self.df['Year'] = self.df['Timestamp'].dt.year
            self.df['Hour'] = self.df['Timestamp'].dt.hour
    
            self.df.drop(columns=['Timestamp', 'Machine_ID'], inplace=True)

            columns_to_encode = ['Efficiency_Status', 'Operation_Mode']
            for col in columns_to_encode:
                label_encoder = LabelEncoder()
                self.df[col] = label_encoder.fit_transform(self.df[col])

            logger.info("Basic data preprocessing data DONE ....")

        except Exception as e:
            logger.error(f"Error in preprocessing: {e}")
            raise CustomException("Error in preprocessing", e)   

    def split_and_scale_and_save(self):
        try:  
            features = ['Operation_Mode', 'Temperature_C', 'Vibration_Hz',
                'Power_Consumption_kW', 'Network_Latency_ms', 'Packet_Loss_%',
                'Quality_Control_Defect_Rate_%', 'Production_Speed_units_per_hr',
                'Predictive_Maintenance_Score', 'Error_Rate_%', 'Month', 'Day', 'Year', 'Hour'
            ]
            X = self.df[features]
            y = self.df['Efficiency_Status']

            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)

            logger.info("Input data scaled successfully.")

            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            logger.info("Split into train and test sets successfully.")

            joblib.dump(scaler, os.path.join(self.output_path, 'scaler.pkl'))
            joblib.dump(X_train, os.path.join(self.output_path, 'X_train.pkl'))
            joblib.dump(X_test, os.path.join(self.output_path, 'X_test.pkl'))
            joblib.dump(y_train, os.path.join(self.output_path, 'y_train.pkl'))
            joblib.dump(y_test, os.path.join(self.output_path, 'y_test.pkl'))

            logger.info("Scaler and datasets saved successfully.")

        except Exception as e:
            logger.error(f"Error in train-test split and scaling: {e}")
            raise CustomException("Error in train-test split and scaling", e)

    def run(self):
        self.load_data()
        self.preprocess()
        self.split_and_scale_and_save()
        
if __name__ == "__main__":
    processor = DataProcessor(input_path='artifacts/raw/data.csv', output_path='artifacts/processed')
    processor.run()
