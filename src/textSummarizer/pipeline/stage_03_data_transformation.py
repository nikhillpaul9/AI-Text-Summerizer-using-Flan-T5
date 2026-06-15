import sys
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_transformation import DataTransformation  # Fixed 'conponents' typo
from textSummarizer.logging import logger

class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>> stage Data Transformation started <<<<<")
            config = ConfigurationManager()
            
            # Fetch config and initialize the transformation component
            data_transformation_config = config.get_data_transformation_config()
            data_transformation = DataTransformation(config=data_transformation_config)
            
            # Execute the tokenization and saving process
            data_transformation.convert()
            
            logger.info(">>>>> stage Data Transformation completed <<<<<\n\nx==========x")
            
        except Exception as e:
            logger.exception("An error occurred during the Data Transformation pipeline stage.")
            raise e

if __name__ == '__main__':
    try:
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)