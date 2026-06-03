import os
os.environ["TRANSFORMERS_NO_TF"] = "1"

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class SentimentAnalysis:

    def __init__(self):
        self.device = torch.device("cpu")

        BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(BASE_DIR, "best2") 

        print(f"Loading Tokenizer from Hugging Face & Model from local path: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            "finiteautomata/bertweet-base-sentiment-analysis", 
            normalization=True
        )
        
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_path,
            local_files_only=True
        )

        self.model.to(self.device)
        self.model.eval()
        print("System is ready with your retrained model! 🚀")

    def analyze_mentee_review(self, review_text):
    
        
        if not review_text.strip():
            return "0"  

        inputs = self.tokenizer(
            review_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            prediction = torch.argmax(logits, dim=-1).item() # استخدم dim=-1 أضمن دايماً للـ 1D/2D arrays
  
        print("--- New Request ---")
        print("Cleaned Text:", review_text)
        print("Logits      :", logits)
        print("Prediction  :", prediction)

        return str(prediction)

sentiment_service = SentimentAnalysis()
