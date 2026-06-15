import os
import torch
import pandas as pd
from tqdm import tqdm
import evaluate
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.logging import logger
from textSummarizer.entity import ModelEvaluationConfig

class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def generate_batch_sized_chunks(self, list_of_elements, batch_size):
        for i in range(0, len(list_of_elements), batch_size):
            yield list_of_elements[i : i + batch_size]

    def calculate_metric_on_test_ds(self, dataset, metric, model, tokenizer, 
                                    batch_size=2, device="cpu", 
                                    column_text="description", column_summary="abstract"):
        
        # Ensure we have the raw text columns
        article_batches = list(self.generate_batch_sized_chunks(dataset[column_text], batch_size))
        target_batches = list(self.generate_batch_sized_chunks(dataset[column_summary], batch_size))

        for article_batch, target_batch in tqdm(zip(article_batches, target_batches), total=len(article_batches)):
            
            inputs = tokenizer(article_batch, max_length=1024, truncation=True, 
                               padding="max_length", return_tensors="pt")
            
            with torch.no_grad():
                # Inference on CPU
                summaries = model.generate(
                    input_ids=inputs["input_ids"].to(device),
                    attention_mask=inputs["attention_mask"].to(device), 
                    length_penalty=0.8, 
                    num_beams=4, 
                    max_length=256,
                    min_length=14
                )
            
            decoded_summaries = [tokenizer.decode(s, skip_special_tokens=True, 
                                                clean_up_tokenization_spaces=True) 
                                 for s in summaries]      
            
            decoded_summaries = [d.replace("\n", " ") for d in decoded_summaries]
            
            metric.add_batch(predictions=decoded_summaries, references=target_batch)
            
        score = metric.compute()
        return score

    def evaluate(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Initializing evaluation on device: {device}")
        
        tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(device)
       
        # CRITICAL: Load RAW dataset for evaluation, NOT the transformed one
        logger.info(f"Loading RAW dataset for evaluation from: {self.config.data_path}")
        dataset_pt = load_from_disk(self.config.data_path)

        rouge_metric = evaluate.load('rouge')

        text_col = self.config.text_column
        summary_col = self.config.summary_column

        # Handle split logic robustly
        eval_split = 'test' if 'test' in dataset_pt else 'validation'
        logger.info(f"Evaluating on '{eval_split}' split | Input: '{text_col}' -> Target: '{summary_col}'")

        score = self.calculate_metric_on_test_ds(
            dataset=dataset_pt[eval_split], 
            metric=rouge_metric, 
            model=model, 
            tokenizer=tokenizer, 
            batch_size=2,
            device=device,
            column_text=text_col, 
            column_summary=summary_col
        )

        # Structure scores
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, score[rn]) for rn in rouge_names)

        # Save to CSV
        model_name = str(self.config.model_path).split('/')[-1]
        df = pd.DataFrame(rouge_dict, index=[model_name])
        
        os.makedirs(os.path.dirname(self.config.metric_file_name), exist_ok=True)
        df.to_csv(self.config.metric_file_name, index=False)
        logger.info(f"Metrics saved to {self.config.metric_file_name}")