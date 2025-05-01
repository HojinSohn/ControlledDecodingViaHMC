'''
Get the sampling works for one prompt: "The movie was"

The sample sequence should looks different that the initial sequence



Issue:

1. The initial sequence is sampled by greedy, repeating sentence issue arises. Need to sample initial sequence through top-p, top-k, or beam search
2. Currently, assumes the batch size is 1. In the case when there are multiple batches, need to change the code accordingly
3. Maybe read prompts from input file?
4. Currently rejecting all samples. Issue with energy function or leap frong steps
'''

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np

from util import *
from sampler.SequenceEnergy import SequenceEnergy
from sampler.HMCSampler import HMCSampler
import matplotlib.pyplot as plt

def options():
    # Initialize the ArgumentParser
    parser = argparse.ArgumentParser(description="Example script with various arguments including boolean flags and default values.")
    
    # Add argument for device (e.g., 'cpu', 'cuda')
    parser.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'], help='Device to run the model on (cpu or cuda) [Default: cpu]')
 
    # Add argument for lambda_energy (float) with default
    parser.add_argument('--lambda_energy', type=float, default=1.0, help='Lambda energy value [Default: 1.0]')

    # Add argument for epsilon (float) with default
    parser.add_argument('--epsilon', type=float, default=0.1, help='Epsilon value [Default: 0.1]')

    # Add argument for debug toggle (boolean flag) with default
    parser.add_argument('--debug', action='store_true', help='Enable debug mode [Default: False]')

    # Add argument for n_steps (integer) with default
    parser.add_argument('--n_steps', type=int, default=100, help='Number of steps [Default: 100]')

    # Add argument for std_dev (float) with default
    parser.add_argument('--std_dev', type=float, default=0.2, help='Standard deviation value [Default: 0.2]')

    # Add argument for delta (float) with default
    parser.add_argument('--delta', type=float, default=0.5, help='Delta value [Default: 0.5]')

    # Add argument for num_leapfrog (integer) with default
    parser.add_argument('--num_leapfrog', type=int, default=10, help='Number of leapfrog steps [Default: 10]')

    # Add argument for num_leapfrog (integer) with default
    parser.add_argument('--alpha', type=float, default=0.1, help='Step size of lambda update [Default: 0.1]')

    # Parse the arguments
    args = parser.parse_args()

    # Print out the values of the arguments for confirmation (or debug purposes)
    print("="*50)
    print(f"Running with the following parameters:")
    print("="*50)
    print(f"Device               : {args.device}")
    print(f"Lambda Energy        : {args.lambda_energy}")
    print(f"Epsilon              : {args.epsilon}")
    print(f"Debug Mode           : {'Enabled' if args.debug else 'Disabled'}")
    print(f"Number of Steps      : {args.n_steps}")
    print(f"Standard Deviation   : {args.std_dev}")
    print(f"Delta                : {args.delta}")
    print(f"Leapfrog Steps       : {args.num_leapfrog}")
    print(f"Lambda Update Alpha  : {args.alpha}")
    print("="*50)

    return args


def create_plot(lamba_records, nll_records, sentiment_records, filename):
    # Convert tensors to NumPy arrays if they are on a GPU
    # def convert_to_numpy(tensor_list):
    #     return [t.numpy() for t in tensor_list] 

    # lamba_records = convert_to_numpy(lamba_records)
    # nll_records = convert_to_numpy(nll_records)
    # sentiment_records = convert_to_numpy(sentiment_records)

    # Create a figure and a set of subplots
    fig, ax1 = plt.subplots()

    # Create a color for each axis
    color_nll = 'tab:blue'
    color_sentiment = 'tab:orange'

    # Plot NLL
    ax1.set_xlabel('Lambda')
    ax1.set_ylabel('NLL', color=color_nll)
    ax1.plot(lamba_records, nll_records, color=color_nll, marker='o', label='NLL')
    ax1.tick_params(axis='y', labelcolor=color_nll)

    # Create a second y-axis for the sentiment scores
    ax2 = ax1.twinx()  
    ax2.set_ylabel('Sentiment Score', color=color_sentiment)  
    ax2.plot(lamba_records, sentiment_records, color=color_sentiment, marker='s', label='Sentiment Score')
    ax2.tick_params(axis='y', labelcolor=color_sentiment)

    # Title and legend
    plt.title('Lambda vs NLL and Sentiment Score')
    fig.tight_layout()  
    plt.grid()

    plt.savefig(filename, format='png')  # Change format if needed
    plt.close()  # Close the figure to free up memory

def generate_filename(base_name, model_name, prompt, lambda_energy, epsilon, std_dev, n_steps, delta, num_leapfrog, alpha):
    """
    Generate a filename based on a base name and parameters.

    Parameters:
    - base_name (str): The base name for the file.
    - model_name (str): The name of the model used.
    - prompt (str): The prompt text.
    - lambda_energy (float): The lambda energy parameter.
    - epsilon (float): The epsilon parameter.
    - std_dev (float): The standard deviation parameter.
    - n_steps (int): The number of steps.
    - delta (float): The delta parameter.
    - num_leapfrog (int): The number of leapfrog steps.
    - alpha (float): The alpha parameter.

    Returns:
    - str: Generated filename.
    """
    # Create a sanitized prompt for the filename
    sanitized_prompt = prompt.replace(" ", "_").replace(",", "").replace(".", "")[:10]  # Limit length for filename
    
    # Construct the filename components
    param_strings = [
        f"model={model_name}",
        f"prompt={sanitized_prompt}",
        f"lambda={lambda_energy}",
        f"epsilon={epsilon}",
        f"stddev={std_dev}",
        f"nsteps={n_steps}",
        f"delta={delta}",
        f"leapfrog={num_leapfrog}",
        f"alpha={alpha}"
    ]
    
    # Join parameters with underscores
    param_part = "_".join(param_strings)
    
    # Create the full filename
    filename = f"{base_name}/{param_part}.png"  # Change extension if needed
    
    return filename

def experiment(model, tokenizer, prompt_ids, e_positive, Y, embed_lut, device, lambda_energy, epsilon, alpha, rng, n_steps, std_dev, delta, num_leapfrog, debug):
    
    # Initialize sequence energy based on prompt and initial output token embeddings
    seq_energy = SequenceEnergy(model, prompt_ids, Y, e_positive, device=device, lambda_energy=lambda_energy, epsilon=epsilon, debug=debug)
    
    # set file name based on parameters
    file_name = get_file_name(lambda_energy, epsilon, alpha, n_steps, std_dev, delta, num_leapfrog)

    # Initialize hmc sampler
    sampler = HMCSampler(seq_energy, rng=np.random.RandomState(42), device=device, alpha=alpha, debug=debug, log_file_name=file_name)
    
    # sample from HMC sampler
    samples, lamba_records, nll_records, sentiment_records = sampler.sample(n_steps, std_dev, delta, num_leapfrog)

    # plot energy graph
    plot_energy_movement(file_name)

    min_nll = float('inf')
    max_nll = float('-inf')

    min_sample = None
    max_sample = None
    min_sentiment = None
    max_sentiment = None

    for embeddings in samples:
        print(embeddings.shape)
        scores = torch.cdist(embeddings.view(-1, embeddings.size(-1)), embed_lut.weight)
        token_ids = scores.argmin(dim=-1).view(embeddings.size(0), -1)
        token_ids_list = token_ids.squeeze().tolist() # Convert tensor to list for tokenizer.decode
        decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
        print(f"sample sentence: {decoded_text}")

        with torch.no_grad():
            outputs = model(token_ids)
            logits = outputs.logits

        log_probs = torch.log_softmax(logits, dim=-1)

        nll = -log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)  # [batch_size]

        print(f"NLL: {nll.item():.4f}")

        normalized_embeddings = torch.nn.functional.normalize(embeddings.view(-1, embeddings.size(-1)), dim=-1)
        normalized_target = torch.nn.functional.normalize(e_positive, dim=-1)
        sentiment = torch.sum(normalized_embeddings * normalized_target, dim=-1).mean(-1)
        print(f"Sentiment: {sentiment:.4f}\n")

        # Update min and max NLL and corresponding samples
        if nll.item() < min_nll:
            min_nll = nll.item()
            min_sample = decoded_text
            min_sentiment = sentiment.item()  # Save the sentiment score

        if nll.item() > max_nll:
            max_nll = nll.item()
            max_sample = decoded_text
            max_sentiment = sentiment.item()  # Save the sentiment score
    

def main():
    args = options()
    device = args.device

    # GPT2 Model and Tokenizer
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Lookup table: token IDs into token embeddings
    embed_lut = model.get_input_embeddings()
    
    # Prompt 
    prompt = "Once upon a time, "

    # Token ids for Prompt
    prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

    # Token ids for tokens generated by model (Excluding prompt)
    output_ids = model.generate(prompt_ids, max_length=25, do_sample=True, temperature=0.8, top_k=50)[0, len(prompt_ids[0]):]
    
    # Token embeddings of output sequence (Excluding prompt)
    Y = embed_lut(output_ids).unsqueeze(0)  # [1, 20, 768]

    # Initial sentence generated by model
    decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
    print(f"Initial sentence: {decoded_text}")

    # initialize the target sentiment embedding
    e_positive = embed_lut(tokenizer.encode("good love amaze joy fantastic happy hope success kind brave care compassion inspire confident enthusiastic grate generous friend bright peace support motivate honest trust respect talent excite strong achieve fulfill gentle patient courageous content positive", return_tensors="pt")[0].to(device)).mean(dim=0)

    # e_positive = embed_lut(tokenizer.encode("good love joy happy", return_tensors="pt")[0].to(device)).mean(dim=0)

    # # Prompt 
    # e_positive_sequence = embed_lut(tokenizer.encode("The book is so good"))

    experiment(model, tokenizer, prompt_ids, e_positive, Y, embed_lut, device, args.lambda_energy, args.epsilon, args.alpha, np.random.RandomState(42), args.n_steps, args.std_dev, args.delta, args.num_leapfrog, args.debug)
    
        
    

if __name__ == "__main__":
    main()
