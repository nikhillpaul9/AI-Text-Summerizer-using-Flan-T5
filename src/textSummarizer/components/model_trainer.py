import os
from textSummarizer.logging import logger
from transformers import DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, GenerationConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_from_disk
from textSummarizer.entity import ModelTrainerConfig
import torch
import evaluate
import numpy as np

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def compute_metrics(self, eval_pred):
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        rouge_metric = evaluate.load("rouge")
        
        predictions, labels = eval_pred
        
        # 1. Handle Tuple or Logits
        if isinstance(predictions, tuple):
            predictions = predictions[0]
            
        # If the trainer returned logits (3D array), convert to token IDs
        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)
            
        # 2. Prepare Labels (Mask -100)
        predictions = np.where(predictions != -100, predictions, tokenizer.pad_token_id)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        
        # 3. Decode
        # Ensure inputs are NumPy arrays/lists as expected by batch_decode
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # ... (rest of your ROUGE calculation) ...
        result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        return {k: round(v, 4) for k, v in result.items()}

    def train(self):

        # 1. Native Tokenizer & Model Initialization
        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt)
        
        # 2. Dynamic Data Collator mapping
        seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)
        
        # 3. Loading Data-Agnostic Processed Splitting Artifacts
        logger.info(f"Loading tokenized dataset features from: {self.config.data_path}")
        dataset_pt = load_from_disk(self.config.data_path)

        # Load the default configuration from the checkpoint to preserve architectural tokens
        gen_config = GenerationConfig.from_pretrained(self.config.model_ckpt)
        
        # Apply your custom beam search tuning parameters on top of the inherited defaults
        gen_config.max_length = 256
        gen_config.num_beams = 4
        gen_config.length_penalty = 1.2
        gen_config.early_stopping = True
        gen_config.no_repeat_ngram_size = 3

        # 4. Constructing Core TrainingArguments Configuration
        trainer_args = Seq2SeqTrainingArguments(
            output_dir=self.config.root_dir, 
            num_train_epochs=self.config.num_train_epochs, 
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size, 
            
            # Keep eval batch size small to prevent evaluation OOM
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            
            weight_decay=self.config.weight_decay, 
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy, 
            eval_steps=self.config.eval_steps, 
            save_steps=int(self.config.save_steps) if hasattr(self.config, 'save_steps') else 500,
            save_strategy=self.config.save_strategy,
            
            # Safely fetch save_total_limit from config, default to 2 if missing
            save_total_limit=getattr(self.config, 'save_total_limit', 2),
            
            gradient_accumulation_steps=self.config.gradient_accumulation_steps, 
            gradient_checkpointing=self.config.gradient_checkpointing,
            
            # CRITICAL CPU OPTIMIZATION: Flush evaluation tensors from RAM to disk iteratively
            eval_accumulation_steps=1, 
            
            fp16=self.config.fp16, 
            bf16=self.config.bf16, 
            max_grad_norm=self.config.max_grad_norm, 
            use_cpu=self.config.use_cpu,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            load_best_model_at_end=self.config.load_best_model_at_end,
            predict_with_generate=True,
            generation_max_length=256,
            generation_config=gen_config
        ) 

        # 5. Instantiating the Global Pipeline Trainer Engine
        logger.info("Initializing HuggingFace Seq2SeqTrainer framework components...")
        
        # Swapped to Seq2SeqTrainer
        trainer = Seq2SeqTrainer(
            model=model, 
            args=trainer_args,
            processing_class=tokenizer, 
            data_collator=seq2seq_data_collator,
            train_dataset=dataset_pt["train"], 
            eval_dataset=dataset_pt["validation"],
            compute_metrics=self.compute_metrics
        )
        
        logger.info("Commencing model compilation and training sequence...")
        trainer.train()

        # 6. Serializing Final Output Artifacts to Disk
        model_output_dir = os.path.join(self.config.root_dir, "fine_tuned_model")
        tokenizer_output_dir = os.path.join(self.config.root_dir, "tokenizer")
        
        logger.info(f"Saving compiled model checkpoints to: {model_output_dir}")
        model.save_pretrained(model_output_dir)
        
        logger.info(f"Saving processing vocabulary assets to: {tokenizer_output_dir}")
        tokenizer.save_pretrained(tokenizer_output_dir)
        
        logger.info("Model training pipeline lifecycle executed successfully.")