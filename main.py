import sys
from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_validation import DataValidationTrainingPipeline
from textSummarizer.pipeline.stage_03_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_04_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_05_model_evaluation import ModelEvaluationTrainingPipeline

def main():
    # Registry of all pipeline stages in execution order
    PIPELINE_STAGES = {
        "Data Ingestion": DataIngestionTrainingPipeline,
        "Data Validation": DataValidationTrainingPipeline,
        "Data Transformation": DataTransformationTrainingPipeline,
        "Model Trainer": ModelTrainerTrainingPipeline,
        "Model Evaluation": ModelEvaluationTrainingPipeline
    }

    logger.info("========== SUMMARIZATION PIPELINE EXECUTION INITIATED ==========")

    for stage_name, pipeline_class in PIPELINE_STAGES.items():
        try:
            logger.info(f"*******************")
            logger.info(f">>>>>> Stage: {stage_name} <<<<<<")
            
            # Instantiate and run the pipeline stage
            pipeline_instance = pipeline_class()
            pipeline_instance.main()
            
            logger.info(f">>>>>> Stage {stage_name} completed successfully <<<<<<\n")
            
        except Exception as e:
            logger.exception(f"Pipeline execution halted! Critical failure at: {stage_name}")
            sys.exit(1)  # Stop the entire pipeline immediately if any stage fails

    logger.info("========== ALL PIPELINE STAGES EXECUTED SUCCESSFULLY ==========")

if __name__ == '__main__':
    main()