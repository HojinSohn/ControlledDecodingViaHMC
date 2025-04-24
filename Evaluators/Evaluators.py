
from transformers import GPT2LMHeadModel, GPT2Tokenizer, AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class PerplexityEvaluator:
    def __init__(self, device, debug=False):
        self.device = device
        self.debug = debug
        self.model = model_name = GPT2LMHeadModel.from_pretrained("gpt2-xl")
        self.model.to(device)
        self.model.eval()
        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2-xl")
        self.tokenizer.pad_token = self.tokenizer.eos_token


    # Define a function to compute perplexity for a given text
    def compute_perplexity(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs, labels=inputs["input_ids"])
        
        loss = outputs.loss
        perplexity = torch.exp(loss).item()
        
        if self.debug:
            print(f"Text: {text}")
            print(f"Loss: {loss.item():.4f}, Perplexity: {perplexity:.2f}")
        
        return perplexity

class SentimentEvaluator:
    def __init__(self, model_name, device="cuda", debug=False):
        self.device = device
        self.debug = debug
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    def predict_sentiment(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()

        if self.debug:
            print(f"Text: {text}")
            print(f"Probabilities: {probs}")
            print(f"Predicted class: {pred}")

        return pred, probs.squeeze().tolist()

if __name__ == "__main__":
    # Example usage
    evaluator = PerplexityEvaluator(device="cuda", debug=True)
    text = "I love you so much"
    perplexity = evaluator.compute_perplexity(text)
    print(f"Perplexity: {perplexity:.2f}")

    c1_evaluator = SentimentEvaluator("textattack/roberta-base-SST-2", device="cuda", debug=True)
    sentiment, probs = c1_evaluator.predict_sentiment(text)
    print(f"Sentiment: {probs[1]:.2f}")
