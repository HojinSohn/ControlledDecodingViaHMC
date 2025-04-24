import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

# Load pre-trained model and tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")

# Example sentence
sentence = "Hello, my dog is cute"

# Tokenize the sentence and get input IDs (tokens)
inputs = tokenizer(sentence, return_tensors="pt", padding=True, truncation=True)

# Get token embeddings (you would typically get this from a custom source)
with torch.no_grad():  # No need to compute gradients for the tokenizer part
    embeddings = model.distilbert.embeddings(input_ids=inputs['input_ids'])

# Enable gradient calculation for the embeddings
embeddings.requires_grad = True

# Pass the embeddings through the model to compute logits (sentiment score)
outputs = model.transformer(dropout=0)(embeddings)
logits = model.classifier(outputs[0])

# Get the sentiment score (logits)
sentiment_score = logits[0, 1]  # The second value corresponds to the 'positive' sentiment (class 1)

# Compute gradients of the sentiment score with respect to the embeddings
sentiment_score.backward()

# Now `embeddings.grad` contains the gradients of the sentiment score with respect to the token embeddings
gradients = embeddings.grad

# Print the sentiment score and the gradients
print("Sentiment score (positive class):", sentiment_score.item())
print("Gradients with respect to embeddings:", gradients)
