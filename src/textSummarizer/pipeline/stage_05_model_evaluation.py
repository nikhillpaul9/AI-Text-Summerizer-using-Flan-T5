import sys
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.model_evaluation import ModelEvaluation  # Fixed 'conponents' typo
from textSummarizer.logging import logger

class ModelEvaluationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>> stage Model Evaluation started <<<<<")
            config = ConfigurationManager()
            
            # Fetch config
            model_evaluation_config = config.get_model_evaluation_config()
            
            # Initialize the component (Fixed variable shadowing bug here)
            model_evaluation = ModelEvaluation(config=model_evaluation_config)
            
            # Execute the evaluation sequence
            model_evaluation.evaluate()
            
            logger.info(">>>>> stage Model Evaluation completed <<<<<\n\nx==========x")
            
        except Exception as e:
            logger.exception("An error occurred during the Model Evaluation pipeline stage.")
            raise e

if __name__ == '__main__':
    try:
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)