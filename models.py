# from interfaces import ToxicityClassifier
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForSequenceClassification
import torch


class ToxicityClassifier():

    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("models/cga_deberta_onnx_int8")

        self.ort_model = ORTModelForSequenceClassification.from_pretrained(
            "models/cga_deberta_onnx_int8", 
            file_name="model_quantized.onnx"
        )
    def predict(self, text: str) -> float:

        #print(self.ort_model.config)

        inputs = self.tokenizer(text, return_tensors="pt", padding=True, truncation=True)

        with torch.no_grad():
            outputs = self.ort_model(**inputs)
        
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        toxicity_score = probs[0][1].item()

        return toxicity_score

    
# if __name__ == "__main__":
#     classifier = ToxicityClassifier()
#     text = "Hello!!"
#     result = classifier.predict(text)
#     print(result)