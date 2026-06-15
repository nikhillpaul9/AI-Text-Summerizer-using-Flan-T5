import sys
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_validation import DataValidation  # Fixed 'conponents' and 'DataValiadtion' typos
from textSummarizer.logging import logger

class DataValidationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>> stage Data Validation started <<<<<")
            config = ConfigurationManager()
            data_validation_config = config.get_data_validation_config()
            
            # Instantiating the corrected DataValidation class
            data_validation = DataValidation(config=data_validation_config)
            data_validation.validate_all_files_exist()
            
            logger.info(">>>>> stage Data Validation completed <<<<<\n\nx==========x")
            
        except Exception as e:
            logger.exception("An error occurred during the Data Validation pipeline stage.")
            raise e

if __name__ == '__main__':
    try:
        pipeline = DataValidationTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)