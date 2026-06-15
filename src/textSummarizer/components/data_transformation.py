import os
from textSummarizer.logging import logger
from transformers import AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import DataTransformationConfig

class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def convert_examples_to_features(self, example_batch):
        text_column = self.config.text_column
        summary_column = self.config.summary_column

        prefix = "summarize: "
        inputs = [prefix + doc for doc in example_batch[text_column]]

        # Tokenize Inputs (Article)
        input_encodings = self.tokenizer(
            inputs, 
            max_length=1024, 
            truncation=True,
            padding=False
        )
        
        # Tokenize Targets (Abstract)
        target_encodings = self.tokenizer(
            example_batch[summary_column], 
            max_length=256, 
            truncation=True,
            padding=False
        )
            
        return {
            'input_ids': input_encodings['input_ids'],
            'attention_mask': input_encodings['attention_mask'],
            'labels': target_encodings['input_ids']
        }
    
    def filter_aiml_papers(self, example):
        """
        Filter to isolate AI/ML papers based on vocabulary in the article.
        """
        keywords = ["machine learning", "neural network", "deep learning", 
                    "transformer", "artificial intelligence", "large language model", "llm",
                    "embedding", "supervised learning", "inference",
                    "attention mechanism", "neural network", "backpropagation", 
                    "gradient descent", "convolutional", "encoder", "decoder", "embedding", "latent space",
                    "unsupervised learning", "reinforcement learning", "self-supervised", "transfer learning",
                    "fine-tuning", "regression", "clustering","generative ai", "diffusion model", "rag", 
                    "retrieval augmented generation", "prompt engineering", "fine-tuned", "zero-shot",
                    "few-shot", "alignment", "rlhf","rouge", "bleu", "perplexity", "accuracy", "f1-score",
                    "convergence", "overfitting", "regularization"]
        
        text = str(example[self.config.text_column]).lower()
        return any(keyword in text for keyword in keywords)

    def convert(self):
        logger.info(f"Loading pre-truncated dataset from: {self.config.data_path}")
        dataset = load_from_disk(self.config.data_path)
        
        # 1. HEURISTIC FILTERING: Keep only AI/ML related papers
        logger.info("Applying AI/ML heuristic filter...")
        filtered_dataset = dataset.filter(self.filter_aiml_papers, num_proc=2)
        
        # 2. PARALLEL TOKENIZATION
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        logger.info("Launching parallel tokenization...")
        
        # We assume the dataset has at least one split (e.g., 'train')
        columns_to_remove = filtered_dataset['train'].column_names 
        
        dataset_pt = filtered_dataset.map(
            self.convert_examples_to_features, 
            batched=True,
            num_proc=2,
            remove_columns=columns_to_remove 
        )
        
        output_path = os.path.join(self.config.root_dir, "transformed_dataset")
        logger.info(f"Saving tokenized PyTorch features to: {output_path}")
        dataset_pt.save_to_disk(output_path)
        logger.info("Data Transformation step completed successfully.")

        # Add this to convert()
        sample = dataset_pt['train'][0]
        logger.info(f"Sample Input starts with: {self.tokenizer.decode(sample['input_ids'][:10])}")
        logger.info("Data Transformation step completed successfully.")