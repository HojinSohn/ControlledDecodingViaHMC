

class Decode:
    def __init__(self, device, ):
   
# Setup
device = "cuda" if torch.cuda.is_available() else "cpu"
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
embed_lut = model.get_input_embeddings()

# Sentiment classifier for evaluation
sentiment_model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english").to(device)
sentiment_tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased-finetuned-sst-2-english")
sentiment_model.eval()

# Test parameters
prompt = "The movie was"
prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
output_ids = model.generate(prompt_ids, max_length=23, do_sample=False)[0, len(prompt_ids[0]):]
Y = embed_lut(output_ids).unsqueeze(0)  # [1, 20, 768]

decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
print(f"Initial sentence: {decoded_text}")

e_positive = embed_lut(tokenizer.encode("great happy good", return_tensors="pt")[0].to(device)).mean(dim=0)

# Initialize
seq_energy = SequenceEnergy(model, prompt_ids, Y, e_positive, device=device, lambda_energy=1.0, epsilon=0.7)
sampler = HMCSampler(seq_energy, rng=np.random.RandomState(42), device=device)

samples = sampler.sample(10, 0.1, 1, 10)
print(samples[0])