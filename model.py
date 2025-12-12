import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from logics import load_mkb_codes


class MedicModel:
    def __init__(self, model_dir: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir, trust_remote_code=True)
        self.id2label = load_mkb_codes(f"{model_dir}/labels.json")
    
    def predict(self, query: str) -> str:
        """
        Returns the MKB-code of the disease
        """
        inputs = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding="max_length",
            max_length=256
        )
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
        pred_id = torch.argmax(logits, dim=1).item()
        pred_code = self.id2label[pred_id]
        return pred_code