import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

text = "Hello, my dog is cute."
input_ids = tokenizer.encode(text, return_tensors="pt")  
print(input_ids)  
# Output: [15496, 11, 616, 3290, 318, 11850, 13]

# Generate 20 words (max_new_tokens = 20 ensures we get 20 additional words)
output = model.generate(input_ids, max_new_tokens=20, do_sample=True, temperature=0.7)

# output always contains a lsit of token ids "for each batch"!!! 
#That is why output[0] is used to get first batch sentence

# get the token ids of output
print(output[0])


decoded_text = tokenizer.decode(output[0], skip_special_tokens=True)
print(decoded_text)  

'''
OUTPUT:

tensor([[15496,    11,   616,  3290,   318, 13779,    13]])
tensor([15496,    11,   616,  3290,   318, 13779,    13,   921,   760,    11,
          339,   460,   307,   257,  1310,  1643,   286,   257, 29757,    11,
          475,   314,  1101,  1016,   284,  2298,   683])
Hello, my dog is cute. You know, he can be a little bit of a jerk, but I'm going to pick him

'''