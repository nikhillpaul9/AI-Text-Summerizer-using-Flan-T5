import os
from textSummarizer.logging import logger
from textSummarizer.utils.common import get_size
from pathlib import Path
from textSummarizer.entity import DataIngestionConfig
from datasets import load_dataset


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config
    
    def download_dataset(self):
        """
        Downloads the specified dataset and configuration split directly from HuggingFace.
        Saves the raw DatasetDict structurally to disk for pipeline continuity.
        """
        output_dir = Path(self.config.unzip_dir)
        
        # Check if the folder exists and is not empty to avoid redundant downloads
        if not os.path.exists(output_dir) or not os.listdir(output_dir):
            logger.info(f"Initiating download for dataset path: '{self.config.source_URL}'")
            
            # Dynamically fetch the dataset (e.g., source_URL="big_patent", split_config="d")
            # If split_config is not provided in your entity, it defaults to None
            split_config = getattr(self.config, "split_config", None)
            
            if split_config:
                logger.info(f"Using specific dataset sub-configuration subset: '{split_config}'")
                dataset = load_dataset(self.config.source_URL, split_config,
                                       split={
                                            "train": "train[:5000]",          
                                            "validation": "validation[:500]", 
                                            "test": "test[:500]"              
                                        }, trust_remote_code=True)
            else:
                dataset = load_dataset(self.config.source_URL,trust_remote_code=True)
                
            logger.info(f"Download complete. Serializing raw dataset artifact to disk: {output_dir}")
            os.makedirs(output_dir, exist_ok=True)
            dataset.save_to_disk(output_dir)
            
        else:
            logger.info(f"Raw dataset artifact already exists on disk at: {output_dir}")

    def extract_zip_file(self):
        """
        Maintained for structural pipeline interface compatibility. 
        Acts as a safe pass-through since data is now natively downloaded and unzipped.
        """
        logger.info("Skipping zip extraction: Dataset downloaded natively in raw Arrow format.")
        return None