import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np

import sys
import os

from sampler.SequenceEnergy import SequenceEnergy
from sampler.HMCSampler import HMCSampler


# Load GPT-2 model and tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda")
embed_lut = model.get_input_embeddings()

def get_word_embedding(word):
    token_ids = tokenizer.encode(word, add_special_tokens=False)  # Get all subword tokens
    token_embeddings = embed_lut(torch.tensor(token_ids).to("cuda"))  # Get embeddings
    return token_embeddings.mean(dim=0)  # Average across subword tokens

# Expanded word list for sentiment analysis
words = ["curious", "cautious", "happy", "joyful", "sad", "angry", "excited", "bored"]

# Compute word embeddings properly
word_embeddings = torch.stack([get_word_embedding(word) for word in words])  # Convert list to tensor

# Compute cosine similarity matrix
cosine_sim_matrix = torch.nn.functional.cosine_similarity(
    word_embeddings.unsqueeze(1),  # [num_words, 1, embedding_dim]
    word_embeddings.unsqueeze(0),  # [1, num_words, embedding_dim]
    dim=-1
).cpu().detach().numpy()  # Convert to NumPy

# Print results
print("Cosine Similarity Matrix:\n")
print(f"{'':<10} " + "  ".join([f"{w:<10}" for w in words]))
for i, word in enumerate(words):
    print(f"{word:<10} " + "  ".join([f"{cosine_sim_matrix[i, j]:.2f}" for j in range(len(words))]))

# Check specific pair similarity
curious_emb = get_word_embedding("curious")
cautious_emb = get_word_embedding("cautious")

cos_sim = torch.nn.functional.cosine_similarity(
    curious_emb.unsqueeze(0), cautious_emb.unsqueeze(0), dim=-1
)

print(f"\nCosine Similarity (curious vs cautious): {cos_sim.item():.4f}")
