'''
This is main script for running the HMC sampler on a language model.
Intializes the model, tokenizer, and other parameters.
Samples initial sentences from the model and then runs the HMC sampler starting from the generated sentences.

Saves the generated samples to a CSV file.
For debugging purpose, plot the energy movement during the sampling process.
'''

import argparse
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
import numpy as np
from util import *
from Sampler.HMCSampler_2 import HMCSampler2
import matplotlib.pyplot as plt
from Sampler.Embeddings import Embeddings
from Evaluators.Evaluators import PerplexityEvaluator, SentimentEvaluator

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


   # Add argument for prompt string
   parser.add_argument('--prompt', type=str, default="Once upon a time, ", help='Prompt text [Default: "Once upon a time, "]')


   # Add argument for sequence length
   parser.add_argument('--seq_length', type=int, default=25, help='Sequence length [Default: 25]')


   # Add argument for sequence length
   parser.add_argument('--plot_energy', action='store_true', help='Enable plot energy [Default: False]')




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
   print(f"Prompt               : {args.prompt}")
   print(f"Sequence Length      : {args.seq_length}")
   print("="*50)


   return args




def create_plot(lamba_records, nll_records, sentiment_records, filename):
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


def save_samples(samples, filename):
   with open(filename, mode="w", newline="") as file:
       writer = csv.writer(file)
       # Write header
       writer.writerow([
           "Text", "Fluency Loss", "Sentiment Loss",
           "Potential Energy", "Kinetic Energy", "Lambda", "Accepted"
       ])
      
       for sample in samples:
           writer.writerow([
               sample["decoded_text"],
               sample["fluency_loss"],
               sample["sentiment_loss"],
               sample["potential_energy"],
               sample["kinetic_energy"],
               sample["lambda"],
               int(sample["Accepted"])  # store shape as tuple
           ])


def experiment(model, tokenizer, prompt_ids, Y, embed_lut, device, lambda_energy, epsilon, alpha, rng, n_steps, std_dev, delta, num_leapfrog, debug, plot_energy):
   # set file name based on parameters
   file_name = get_file_name(lambda_energy, epsilon, alpha, n_steps, std_dev, delta, num_leapfrog)
      
   embeddings = Embeddings(model.config.n_embd , embed_lut, Y.size(1), Y.size(0), device, Y, metric="dot")

   # Initialize hmc sampler
   sampler = HMCSampler2(model, embeddings, prompt_ids, lambda_energy, rng=np.random.RandomState(42), device=device, alpha=alpha, epsilon=epsilon, debug=debug, log_file_name=file_name)
  
   # sample from HMC sampler
   samples = sampler.sample(n_steps, std_dev, delta, num_leapfrog)
   # plot energy graph
   if plot_energy:
       plot_energy_movement(file_name)
   save_samples(samples, file_name)

def main():
   args = options()
   device = args.device


   # GPT2 Model and Tokenizer
   model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
   tokenizer = GPT2Tokenizer.from_pretrained("gpt2")


   # Lookup table: token IDs into token embeddings
   embed_lut = model.get_input_embeddings()
  
   # Prompt
   prompt = args.prompt


   # Token ids for Prompt
   prompt_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)


   # Token ids for tokens generated by model (Excluding prompt)
   output_ids = model.generate(prompt_ids, max_length=args.seq_length , do_sample=True, temperature=0.8, top_k=50)[0, len(prompt_ids[0]):]
  
   # Token embeddings of output sequence (Excluding prompt)
   Y = embed_lut(output_ids).unsqueeze(0)  # [1, 20, 768]


   # Initial sentence generated by model
   decoded_text = tokenizer.decode(output_ids, skip_special_tokens=True)
   print(f"Initial sentence: {decoded_text}")
  
   # # Prompt
   # e_positive_sequence = embed_lut(tokenizer.encode("The book is so good"))


   experiment(model, tokenizer, prompt_ids, Y, embed_lut, device, args.lambda_energy, args.epsilon, args.alpha, np.random.RandomState(42), args.n_steps, args.std_dev, args.delta, args.num_leapfrog, args.debug, args.plot_energy)


if __name__ == "__main__":
   main()



