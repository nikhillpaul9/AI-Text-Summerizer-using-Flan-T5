import torch
from textSummarizer.config.configuration import ConfigurationManager
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.tokenizer_path)
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_path).to(self.device)
    
    def predict(self, text, **gen_kwargs):
        """
        Accepts raw input text along with dynamic generation overrides 
        passed from the Streamlit front-end.
        """
        print("Input Text:")
        print(text)

        formatted_text = "summarize: " + text

        inputs = self.tokenizer(
            formatted_text, 
            max_length=1024, 
            truncation=True, 
            return_tensors="pt"
        ).to(self.device)

        # Set fallback production defaults if parameters aren't supplied by the caller
        generation_config = {
            "max_length": gen_kwargs.get("max_length", 256),
            "min_length": gen_kwargs.get("min_length", 30),          
            "length_penalty": gen_kwargs.get("length_penalty", 1.2),      
            "num_beams": gen_kwargs.get("num_beams", 4),             
            "no_repeat_ngram_size": gen_kwargs.get("no_repeat_ngram_size", 3),   
            "early_stopping": gen_kwargs.get("early_stopping", True)
        }

        with torch.no_grad():
            summary_ids = self.model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **generation_config
            )
        
        output = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        print("\nModel Summary:")
        print(output)

        return output