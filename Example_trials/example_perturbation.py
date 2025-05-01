# import torch
# from transformers import GPT2Tokenizer, GPT2LMHeadModel

# tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
# model = GPT2LMHeadModel.from_pretrained('gpt2')

# text = "Hello, my dog is cute."
# input_ids = tokenizer.encode(text, return_tensors="pt")  
# print(input_ids)  
# # Output: [15496, 11, 616, 3290, 318, 11850, 13]

# # Get the token embeddings
# with torch.no_grad():  # Disable gradient computation
#     outputs = model(input_ids, output_hidden_states=True)  # Forward pass

#     final_token_embeddings = outputs.hidden_states[-1]   # Shape: (batch_size, seq_len, hidden_dim)

# print(final_token_embeddings.shape)  # Example: torch.Size([1, 7, 768])

# modified_embeddings = final_token_embeddings + 0.01 * torch.randn_like(final_token_embeddings)

# # Feed modified embeddings back to generate text
# # We need to use the model's decoder part directly
# with torch.no_grad():
#     # Project embeddings to logits
#     logits = model.lm_head(final_token_embeddings)
#     # Get the most likely next tokens
#     next_token_ids = torch.argmax(logits, dim=-1)
#     # Decode tokens back to text
#     new_text = tokenizer.decode(next_token_ids[0])

#     print("Original text:", new_text)

#     # Project embeddings to logits
#     logits = model.lm_head(modified_embeddings)
#     # Get the most likely next tokens
#     next_token_ids = torch.argmax(logits, dim=-1)
#     # Decode tokens back to text
#     new_text = tokenizer.decode(next_token_ids[0])

#     print("Modified text:", new_text)
# '''

# '''

import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import torch.nn.functional as F

# Load tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# Original sentence
original_text = "Hello, my dog is cute."
input_ids = tokenizer.encode(original_text, return_tensors="pt")

print("Original tokens:", tokenizer.convert_ids_to_tokens(input_ids[0]))


# Get token embeddings directly from the embedding layer
token_embeddings = model.transformer.wte(input_ids)

vocab_embeddings = model.transformer.wte.weight
vocab_embeddings = vocab_embeddings.unsqueeze(0)

# Function to clean tokens by removing the special "Ġ" character
def clean_tokens(tokens):
    return [token.replace('Ġ', ' ') if token.startswith('Ġ') else token for token in tokens]

def compute_negative_log_likelihood(token_ids):
    with torch.no_grad():
        # Run the model
        outputs = model(token_ids)
        logits = outputs.logits  # Shape: [batch_size, seq_len, vocab_size]

    # Compute log probabilities
    log_probs = torch.log_softmax(logits, dim=-1)  # Shape: [batch_size, seq_len, vocab_size]
    # print("Log probs shape:", log_probs.shape)

    # Reshape the token_ids to have shape [batch_size, seq_len, 1]
    token_ids = token_ids.unsqueeze(-1)  # Shape: [batch_size, seq_len, 1]

    # Gather the log probabilities for the correct tokens
    nll_per_token = -log_probs.gather(2, token_ids)  # Shape: [batch_size, seq_len, 1]
    nll_per_token = nll_per_token.squeeze(-1)  # Shape: [batch_size, seq_len]

    # print("Negative Log-Likelihood per token shape:", nll_per_token.shape)

    # Sum the negative log-likelihoods over all tokens (seq_len dimension)
    nll_sum = nll_per_token.sum()  # Sum over seq_len

    return nll_sum

sample_token_embeddings = token_embeddings
for _ in range(10):
    sample_token_embeddings = token_embeddings + torch.randn_like(sample_token_embeddings) * 0.55

    sample_token_embeddings = sample_token_embeddings.squeeze(0)
    sample_token_embeddings = sample_token_embeddings.unsqueeze(1)

    # Compute cosine similarities between the input embeddings and all vocab embeddings
    cosine_similarities = F.cosine_similarity(sample_token_embeddings.unsqueeze(1), vocab_embeddings.unsqueeze(0), dim=-1)

    # Get the token ids by finding the max similarity for each input token
    projected_token_ids = torch.argmax(cosine_similarities, dim=-1)

    nll = compute_negative_log_likelihood(projected_token_ids)

    # Convert token IDs back to words using the tokenizer
    projected_tokens = tokenizer.convert_ids_to_tokens(projected_token_ids)

    # Clean the projected tokens to remove the "Ġ" character
    cleaned_projected_tokens = clean_tokens(projected_tokens)

    # Print the cleaned tokens as a sentence
    print("Cleaned Projected Tokens:", ''.join(cleaned_projected_tokens), " with ", nll.item())



'''
OUTPUT:


Original tokens: ['Hello', ',', 'Ġmy', 'Ġdog', 'Ġis', 'Ġcute', '.']
Cleaned Projected Tokens: Hello, mydog f cute.  with  51.87257766723633
Cleaned Projected Tokens: Hello, Campus dog is cute.  with  50.46645736694336
Cleaned Projected Tokens: Hello, Jacket dog isfoundland g  with  57.978485107421875
Cleaned Projected Tokens: Hello, my dog is cute disclaimer  with  54.59376525878906
Cleaned Projected Tokens: Hello, my dogIs Drag's  with  54.232139587402344
Cleaned Projected Tokens: Hello sway Fitz dog dru cute.  with  61.762062072753906
Cleaned Projected Tokens: Hello, my dog does cute.  with  50.086585998535156
Cleaned Projected Tokens: Hello on my deduction is adorable.  with  50.21208190917969
Cleaned Projected Tokens: Hello,boys retribution is cute.  with  56.42522048950195
Cleaned Projected Tokens: Hello, my Presents was cute.  with  50.158660888671875
Cleaned Projected Tokens: Learn, my dog is cute reference  with  52.86248016357422
Cleaned Projected Tokens: Regarding, My dog Â© playful.  with  54.86393737792969
Cleaned Projected Tokens: Hello, accept dog Winning cute.  with  55.90278244018555
Cleaned Projected Tokens: Maps, blasts dog is Mini.  with  52.608154296875
Cleaned Projected Tokens: Hello, my dog ultra flaws.  with  54.03535461425781
Cleaned Projected Tokens: HelloâĢĶomething dog eye cute.  with  58.9241943359375
Cleaned Projected Tokens: Hello, Lens dog does cute.  with  53.904659271240234
Cleaned Projected Tokens: Hello, my dogconst cute.  with  51.918174743652344
Cleaned Projected Tokens: Hello- my veterinary is cute.  with  48.80023193359375
Cleaned Projected Tokens: Hello883 my dog has cute P  with  56.792816162109375
Cleaned Projected Tokens: Hello, one dog is cute.  with  47.13533401489258
Cleaned Projected Tokens: Hello, my dog is Sexy.  with  48.109962463378906
Cleaned Projected Tokens: Hello, my sausages cute.  with  51.14419937133789
Cleaned Projected Tokens: Hellomanshipales dog sub fish.  with  60.01034164428711
Cleaned Projected Tokens: Hello, gallery dogs is commem.  with  53.08464431762695
Cleaned Projected Tokens: Hello, my dog comes cute.  with  50.516624450683594
Cleaned Projected Tokens: Hello, my geek is cute.  with  50.19437026977539
Cleaned Projected Tokens: Hello, myhelps Attack descendant observation  with  61.132667541503906
Cleaned Projected Tokens: Hello, ourETF isnel.  with  49.137290954589844
Cleaned Projected Tokens: Hello, my stirous cute.  with  52.82746124267578
Cleaned Projected Tokens: Hello by my dog first cute.  with  49.849159240722656
Cleaned Projected Tokens: Hello, my dog wouldnocent.  with  50.764862060546875
Cleaned Projected Tokens: Hello, my dog have cute nuts  with  53.212608337402344
Cleaned Projected Tokens: Hello. my notices wasacies.  with  52.180999755859375
Cleaned Projected Tokens: Hello into my dog is cute one  with  50.83032989501953
Cleaned Projected Tokens: Helloab Higher dog is cute.  with  52.04257583618164
Cleaned Projected Tokens: Hello, my dog (= cute's  with  54.75408172607422
Cleaned Projected Tokens: Hello, my dog Qu cute.  with  52.0526123046875
Cleaned Projected Tokens: Hello, my Ferrari is cute.  with  50.04774475097656
Cleaned Projected Tokens: Hello, what dog is cute.  with  47.55024719238281
Cleaned Projected Tokens: Follow, my dog becomes cute.  with  50.835113525390625
Cleaned Projected Tokens: Hello detail Cait dog is cute.  with  55.19873809814453
Cleaned Projected Tokens: Hello, my dogop cute.  with  49.22510528564453
Cleaned Projected Tokens: Hello. my dog is Vor.  with  48.343902587890625
Cleaned Projected Tokens: Hello, my antis Gl Bride.  with  54.61764907836914
Cleaned Projected Tokens: Hello - ni dog is node.  with  51.2931022644043
Cleaned Projected Tokens: Hello, Sisters dog is cute.  with  51.39607238769531
Cleaned Projected Tokens: Hello, my Exploration analyses glorious.  with  56.090885162353516
Cleaned Projected Tokens: Hello, actually stories is cute.  with  48.23164367675781
Cleaned Projected Tokens: reetings, my dog electric sophisticated.  with  53.048118591308594
Cleaned Projected Tokens: Hello, my dog is cutesburgh  with  53.69908905029297
Cleaned Projected Tokens: Hello,igg dog program substantial.  with  51.83456039428711
Cleaned Projected Tokens: Hello, My dog stages cute.  with  53.59718704223633
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello, my CrossRef is cute for  with  49.973304748535156
Cleaned Projected Tokens: Hello, my dog possibility cute.  with  52.43196105957031
Cleaned Projected Tokens: Hello, revelation accidental is cute.  with  54.48446273803711
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello at my dog is cute.  with  48.7083740234375
Cleaned Projected Tokens: Hello, my dogck cute.  with  50.72217559814453
Cleaned Projected Tokens:  dear, mygenre turned dece.  with  55.02532196044922
Cleaned Projected Tokens: Hello for disqualxon is cute.  with  54.44773483276367
Cleaned Projected Tokens: Hello, ang dog is cute.  with  51.37443542480469
Cleaned Projected Tokens: Hello. my mammal is Hawaiian,  with  49.581565856933594
Cleaned Projected Tokens: Hello, my dog could cute.  with  49.66326904296875
Cleaned Projected Tokens: Hello, my dog is Eag.  with  50.00501251220703
Cleaned Projected Tokens:  solder,bas dog is cute!  with  51.71731948852539
Cleaned Projected Tokens: Hello, my dog Plan adorable.  with  51.66063690185547
Cleaned Projected Tokens: Hello, my dog says cute be  with  50.89398193359375
Cleaned Projected Tokens: Hello presence my dog is cute.)  with  56.79261779785156
Cleaned Projected Tokens: HelloTime my dog is cute.  with  51.93507385253906
Cleaned Projected Tokens: Hello, my dog is colony.  with  48.546226501464844
Cleaned Projected Tokens: Hello, Porter dog is cute.  with  51.244956970214844
Cleaned Projected Tokens: Hello, my dog is cute :  with  49.30381774902344
Cleaned Projected Tokens: Hello, myequality has cute,  with  52.495094299316406
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello, my dog President cute.  with  48.64891815185547
Cleaned Projected Tokens: Hello, myoola is cute.  with  51.210262298583984
Cleaned Projected Tokens: Hello, adventure dog visibility cute.  with  56.30854797363281
Cleaned Projected Tokens: Hello.ged dog socks cute.  with  56.655242919921875
Cleaned Projected Tokens: Hello, my canine finally cute.  with  50.4804801940918
Cleaned Projected Tokens: Hello, my dog is adorable.  with  48.27338409423828
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello,"); dog H cute.  with  53.11176300048828
Cleaned Projected Tokens: Ah, dozens string describes cute.  with  54.75208282470703
Cleaned Projected Tokens: Helloprlying dog is cute5  with  58.51027297973633
Cleaned Projected Tokens: Hello, my surveillance lasts cute.  with  52.4044189453125
Cleaned Projected Tokens: Hello,yp miles is cute.  with  50.37682342529297
Cleaned Projected Tokens: Hello,milo dog is pussy.  with  55.07115936279297
Cleaned Projected Tokens: Hello & my dogs physical cute.  with  52.726890563964844
Cleaned Projected Tokens:  Hello, my dog is cute in  with  47.673011779785156
Cleaned Projected Tokens: Hello, my dog is sacr.  with  48.53643798828125
Cleaned Projected Tokens: Hello, my dog J cute.  with  49.54962921142578
Cleaned Projected Tokens: Hello, my dog citizen cute.  with  52.001060485839844
Cleaned Projected Tokens: Hello H Terry aur is cute.  with  54.679203033447266
Cleaned Projected Tokens: Hello, my dog is Huh.  with  48.93547058105469
Cleaned Projected Tokens: Hello; my dog is cute.  with  49.450660705566406
Cleaned Projected Tokens: Hello, decency dog is cute,  with  53.8095588684082
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello,+ dog by reckless.  with  49.14192581176758
Cleaned Projected Tokens: Hello, Grateful dog is cute:  with  51.71494674682617
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello,The dog Pipe beautiful.  with  52.58978271484375
Cleaned Projected Tokens: Hello, my dog is cute complain  with  54.12433624267578
Cleaned Projected Tokens: Hello, my dog is cute,  with  47.9481201171875
Cleaned Projected Tokens: Hello, my dog is10,  with  44.94781494140625
Cleaned Projected Tokens: Hello, but dog water cute.  with  49.12776565551758
Cleaned Projected Tokens: Hi's my dog @ cutedue  with  56.449951171875
Cleaned Projected Tokens: Hello, accomp Cantor Mass cute.  with  60.022735595703125
Cleaned Projected Tokens: Hello, my dog Pharaoh cute.  with  52.35009002685547
Cleaned Projected Tokens: Hello, my dog or cute,  with  48.12675476074219
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello, my dog is suspect.  with  47.29480743408203
Cleaned Projected Tokens: Hello: Shining dog is cutely  with  53.795814514160156
Cleaned Projected Tokens: Hello that my dog does cute neighb  with  58.909454345703125
Cleaned Projected Tokens: Hello all my portraits is cute.  with  51.483978271484375
Cleaned Projected Tokens: Hello thatrespect dog News cute.  with  55.414283752441406
Cleaned Projected Tokens: Hello, Longh dog is country.  with  50.54071044921875
Cleaned Projected Tokens: Hello, Fior dog The cutepoke  with  59.21677780151367
Cleaned Projected Tokens: Hello, my dog is cute retail  with  51.85203552246094
Cleaned Projected Tokens:  DID, my war is cute burd  with  52.48468780517578
Cleaned Projected Tokens: atial, mySite behaves cute.  with  56.57844543457031
Cleaned Projected Tokens: Hello, my paw is cute.  with  49.55287551879883
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello, my dog also visas.  with  48.19242858886719
Cleaned Projected Tokens: Hello, my dogess cute.  with  50.33097839355469
Cleaned Projected Tokens: Hello,ci dog from phrases.  with  51.712642669677734
Cleaned Projected Tokens:  Before, my dog is cute,  with  47.22154235839844
Cleaned Projected Tokens: Hello also my Colorado is anecdotes.  with  51.61273193359375
Cleaned Projected Tokens: Hello. my Fighters holding cute.  with  52.71297073364258
Cleaned Projected Tokens: Hello, my dogin cute.  with  49.09654998779297
Cleaned Projected Tokens: Hello, my company control cute &  with  50.58941650390625
Cleaned Projected Tokens: Helloult my canine is kissingato  with  57.846134185791016
Cleaned Projected Tokens: Hello,Ù dog decisive cute.  with  53.49517822265625
Cleaned Projected Tokens: Hello. West dog event stylish.  with  51.42692947387695
Cleaned Projected Tokens:  Match, my dog is kitten.  with  47.63353729248047
Cleaned Projected Tokens:  doct, the dog games cute.  with  48.80499267578125
Cleaned Projected Tokens: Hello, my dog plates cute.  with  52.26252746582031
Cleaned Projected Tokens: Hello,specialPhoto traditional cute Elder  with  58.984596252441406
Cleaned Projected Tokens: Hello: my dog calfiddling.  with  57.283851623535156
Cleaned Projected Tokens: Hello, Det dog walks Empress dis  with  61.841209411621094
Cleaned Projected Tokens: Hello such mydog sheer cute.  with  54.5186653137207
Cleaned Projected Tokens: Hello, my footsteps is creepy.  with  51.908111572265625
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: Hello, my discipl is cute.  with  51.345176696777344
Cleaned Projected Tokens: Hello, my dog is cute.  with  47.918487548828125
Cleaned Projected Tokens: IELD, my Penny will cute.  with  49.692867279052734
Cleaned Projected Tokens: Hello, mylu social cute.  with  48.68157196044922
Cleaned Projected Tokens: Hello, my dog is cheek;  with  49.76654815673828
'''