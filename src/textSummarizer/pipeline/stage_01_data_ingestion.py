import sys
from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.components.data_ingestion import DataIngestion
from textSummarizer.logging import logger

class DataIngestionTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        try:
            logger.info(">>>>> stage Data Ingestion started <<<<<")
            config = ConfigurationManager()
            data_ingestion_config = config.get_data_ingestion_config()
            data_ingestion = DataIngestion(config=data_ingestion_config)
            
            # Trigger the new Hugging Face native download method
            data_ingestion.download_dataset()
            
            # Note: extract_zip_file() has been completely removed from this pipeline step 
            # because the dataset is now natively saved to disk in Arrow format.
            
            logger.info(">>>>> stage Data Ingestion completed <<<<<\n\nx==========x")
            
        except Exception as e:
            logger.exception("An error occurred during the Data Ingestion pipeline stage.")
            raise e

if __name__ == '__main__':
    try:
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
    except Exception as e:
        logger.exception(e)
        sys.exit(1)