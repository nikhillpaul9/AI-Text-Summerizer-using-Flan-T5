import sys
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_trainer import ModelTrainer  # Fixed 'conponents' typo
from textSummarizer.logging import logger

class ModelTrainerTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>> stage Model Training started <<<<<")
            config = ConfigurationManager()
            
            # Fetch config
            model_trainer_config = config.get_model_trainer_config()
            
            # Initialize the component (Fixed variable shadowing bug here)
            model_trainer = ModelTrainer(config=model_trainer_config)
            
            # Execute the training sequence
            model_trainer.train()
            
            logger.info(">>>>> stage Model Training completed <<<<<\n\nx==========x")
            
        except Exception as e:
            logger.exception("An error occurred during the Model Training pipeline stage.")
            raise e

if __name__ == '__main__':
    try:
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)