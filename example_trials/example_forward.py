import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Hello, my dog is cute."
input_ids = tokenizer.encode(text, return_tensors="pt")  
print(input_ids)  
# Output: [15496, 11, 616, 3290, 318, 11850, 13]

# Get the token embeddings
with torch.no_grad():  # Disable gradient computation
    outputs = model(input_ids, output_hidden_states=True)  # Forward pass

    final_token_embeddings = outputs.hidden_states[-1]   # Shape: (batch_size, seq_len, hidden_dim)
    logits = outputs.logits  # Logits for next token prediction

print(logits.shape)  # Shape: (batch_size, seq_length, vocab_size)
print(final_token_embeddings.shape)  # Example: torch.Size([1, 7, 768])

next_token_id = torch.argmax(logits[:, -1, :], dim=-1)
print("Predicted next word:", tokenizer.decode(next_token_id))

'''
OUTPUT:

tensor([[15496,    11,   616,  3290,   318, 13779,    13]])
torch.Size([1, 7, 50257])
torch.Size([1, 7, 768])
Predicted next word:  I
'''