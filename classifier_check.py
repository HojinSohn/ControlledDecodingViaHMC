import torch
from torch.nn import functional as F
from finetune import GPT2SentimentClassifier  # Import the SentimentClassifier class
from transformers import GPT2Tokenizer, GPT2LMHeadModel


def get_projected_tokens(embeddings, embed_lut):
    """
    Project embeddings to token IDs.
    
    Returns:
        token_ids: Tensor [batch_size, sent_length]
    """
    embeddings = embeddings.unsqueeze(0)
    print(embeddings.shape)
    print(embed_lut.weight.shape)
    scores = torch.cdist(embeddings.view(-1, embeddings.size(-1)), embed_lut.weight)
    return scores.argmin(dim=-1).view(embeddings.size(0), -1)

def get_projected_tokens_cosine(embeddings, embed_lut):
    """
    Project embeddings to token IDs.
    
    Returns:
        token_ids: Tensor [batch_size, sent_length]
    """
    embeddings = embeddings.unsqueeze(0)
    normalized_embs = F.normalize(embeddings.view(-1, embeddings.size(-1)), dim=-1)
    normalized_lut = F.normalize(embed_lut.weight, dim=-1)
    similarities = torch.matmul(normalized_embs, normalized_lut.T)  # shape: [seq_len, vocab_size]

    token_ids = torch.argmax(similarities, dim=-1)
    return token_ids


# Define the model architecture
classifier_model = GPT2SentimentClassifier()

# Load the state dictionary
classifier_model.load_state_dict(torch.load("best_sentiment_classifier.pt", map_location=torch.device("cuda")))

# Move the model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
classifier_model = classifier_model.to(device)

# Initialize tokenizer and GPT-2 model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # Set padding token

gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
embed_lut = gpt2_model.get_input_embeddings()  # Embedding layer of GPT-2

gpt2_model.eval()  # Set GPT-2 model to evaluation mode

classifier_model.eval() # Ensure the model is in evaluation mode

# sample_embeddings = torch.randn(5, 768, requires_grad=True, device=device)  # Example embeddings for testing

''' what if sample embeddigns starting from sensical negative sentence? '''
text = "I love this movie so much"
token_ids = tokenizer.encode(text, return_tensors="pt").to(device)
# Get the embeddings for the input text
sample_embeddings = embed_lut(token_ids).detach().clone().to(device).requires_grad_(True)

attention_mask = torch.ones_like(token_ids).to(device)
labels = torch.tensor([1], dtype=torch.long, device=device)  # Batch size 1
logits, loss = classifier_model(sample_embeddings, attention_mask=attention_mask, labels=labels)
print("Logits:", logits)
print("Loss:", loss.item())