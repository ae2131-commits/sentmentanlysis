import os
import emoji
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer, pipeline

class SentimentAnalysis:

    def __init__(self):
        # مسار الفولدر اللي هيكون جواه ملف الـ model.onnx والـ tokenizer
        self.model_path = 'best2'
        
        print("🔄 Loading Optimized ONNX Model...")
        # تحميل الموديل عن طريق ONNX Runtime
        self.model = ORTModelForSequenceClassification.from_pretrained(
            self.model_path, 
            file_name="model.onnx"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # عمل الـ Pipeline كالعادة بس شغال بـ ONNX في الخلفية
        self.nlp = pipeline(
            "sentiment-analysis", 
            model=self.model, 
            tokenizer=self.tokenizer
        )
        print("🚀 ONNX Model is ready and super fast!")
              
    def analyze_mentee_review(self, review_text):
        if not review_text:
            return "0"
            
        results = self.nlp(review_text)
        # ارجاع الـ Label بناءً على شكل الـ output بتاعك
        return results[0]['label']
sentiment_service = SentimentAnalysis()
