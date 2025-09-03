from src.data_processing import DataProcessor
from src.model_training import ModelTrainer

if __name__ == "__main__":
    processor = DataProcessor(input_path='artifacts/raw/data.csv', output_path='artifacts/processed/')
    processor.run()
    trainer = ModelTrainer(input_path='artifacts/processed/', model_path='artifacts/models/')
    trainer.run()