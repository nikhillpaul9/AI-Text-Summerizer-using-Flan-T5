import os
from textSummarizer.logging import logger
from textSummarizer.entity import DataValidationConfig

class DataValidation:  # Fixed class name typo (DataValiadtion -> DataValidation)
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_files_exist(self) -> bool:
        try:
            validation_status = True  # Start optimistic
            
            # Use dynamic path from config instead of hardcoded 'samsum_dataset'
            dataset_dir = self.config.unzip_data_dir 
            
            if not os.path.exists(dataset_dir):
                validation_status = False
                logger.error(f"Validation failed: Directory '{dataset_dir}' does not exist.")
            else:
                # Get the list of items in the ingested dataset directory
                all_files = os.listdir(dataset_dir)

                # Iterate through what we REQUIRE, rather than what is present
                for required_file in self.config.ALL_REQUIRED_FILES:
                    if required_file not in all_files:
                        validation_status = False
                        logger.error(f"Validation failed: Missing required file/folder '{required_file}'")
                        break  # Fail fast: stop checking if we already know it's invalid

            # Write the final status to the file ONCE, after all checks are complete
            os.makedirs(os.path.dirname(self.config.STATUS_FILE), exist_ok=True)
            with open(self.config.STATUS_FILE, 'w') as f:
                f.write(f"Validation status: {validation_status}")

            logger.info(f"Data validation phase completed. Status: {validation_status}")
            return validation_status
        
        except Exception as e:
            logger.exception("Exception occurred during data validation.")
            raise e