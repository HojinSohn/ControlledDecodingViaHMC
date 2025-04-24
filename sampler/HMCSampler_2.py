''' 

'''
import torch 
import gc
import numpy as np
from util import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer
from sampler.losses.gpt2_loss import GPT2Loss as FluencyLoss
from sampler.losses.sentiment_loss import SentimentLoss as SentimentLoss
import torch.optim as optim

DEBUG_ACCEPT = False


class HMCSampler2:
    
    # sequence_embeddings = embeddings of the output tokens
    def __init__(self, model, embeddings, prompt_ids, init_lambda, rng, device, alpha=0.1, epsilon=0.8, debug=False, log_file_name='log_data/sample_log'):
        self.embeddings = embeddings

        self.prompt_ids = prompt_ids

        self.rng = rng

        self.alpha = alpha
        
        self.device = device

        self.debug = debug

        self.lambda_grad = 0.0

        self.current_lambda = torch.nn.Parameter(torch.tensor(init_lambda, dtype=torch.float32, device=device))

        self.file_name = log_file_name

        self.model = model

        self.sentiment_loss = SentimentLoss(model, device, prompt_ids, epsilon=epsilon, debug=debug)

        self.gpt2_loss = FluencyLoss(model, device, prompt_ids, debug=debug)


        print(debug)

    def get_sampled_velocities(self, stddv):
        """
        Sample random velocities from zero-mean Gaussian for all parameters.
        
        Parameters
        ----------
        stddv : float32
            standard deviation for all parameters sampling.

        Returns
        -------
        velocities : list of tensors with the same shape as each shape in self.shape sampled velocities for all parameters.

        """
        return torch.normal(mean=0, std=stddv, size=self.embeddings.pred_embeds.size(), device=self.device)
        
    
    def leapfrog(self, velocities, current_potential_energy, delta, std_dev):
        """
        In-place leapfrog iteration.
        It should update `list(self.model.parameters())` as position $x$ in
        HMC.
        It should update `velocities` as momentum $p$ in HMC.
        

        Parameters
        ----------
        velocities : list of length(self.shapes), float32
            sampled velocities for all parameters.
        delta : float32
            delta in HMC algorithm.
        *ARGS : (X, y, y_1hot) as described in utils.py and NeuralNetwork model learning
            
        Returns
        -------
        velocities : list of length(self.shapes), float32
            leapfrog updated velocities for all parameters.

        """

        # Inverse mass matrix scaling: R^-1 = (1/stddv^2) * I
        # inv_mass_scale = 1.0 / (std_dev ** 2) 
        inv_mass_scale = 1.0

        self.embeddings.pred_embeds.grad = None
        self.current_lambda.grad = None

        current_potential_energy.backward(retain_graph=True)

        if self.current_lambda.grad is not None:
            self.lambda_grad = self.current_lambda.grad

        velocities_half = velocities - 0.5 * delta * self.embeddings.pred_embeds.grad

        self.embeddings.pred_embeds.data = self.embeddings.pred_embeds.data + delta * velocities_half * inv_mass_scale

        pred_embeds, _, probs = self.embeddings.forward()

        new_potential_energy = self.compute_potential_energy(pred_embeds, probs)

        # self.embeddings.pred_embeds.grad = None  # Clear before new backward
        new_potential_energy.backward(retain_graph=True)

        velocities = velocities_half - 0.5 * delta * self.embeddings.pred_embeds.grad

        # Clear intermediate tensors and cache
        del pred_embeds, probs, velocities_half
        torch.cuda.empty_cache()

        return velocities, new_potential_energy
        
        

    def accept_or_reject(self, potential_energy_previous, potential_energy_current, 
                         kinetic_energy_previous, kinetic_energy_current):
        """
        Given the potential and kinetic energies  of the last sample and new sample, 
        check if we should accept new sample.
        If True, we will accept new sample.
        If False, we will reject new sample, and repeat the last sample.
        

        Parameters
        ----------
        potential_energy_previous : float32
            potential energy of last sample.
        potential_energy_current : float32
            potential energy of new sample.
        kinetic_energy_previous : float32
            kinetic energy of last sample.
        kinetic_energy_current : float32
            kinetic energy of new sample.

        Returns
        -------
        boolean
            True if to accept, False if to reject.

        """
        
        alpha = min(1, torch.exp(potential_energy_previous + kinetic_energy_previous - potential_energy_current - kinetic_energy_current))
        if np.random.uniform(low=0, high=1) <= alpha:
            return True, alpha
        else:
            return False, alpha
        
        # raise NotImplementedError("Complete Implementation")

    def compute_potential_energy(self, predictions, probs):
        # Compute the potential energy
        fluency_loss = self.gpt2_loss.compute_loss((predictions, probs))
        sentiment_loss = self.sentiment_loss.compute_loss((predictions, probs))
        # Compute the total potential energy
        potential_energy = fluency_loss + self.current_lambda * sentiment_loss
        return potential_energy

    '''
        For debugging purpose
    '''
    def compute_loss(self, predictions, probs):
        with torch.no_grad():
            sentiment_loss = self.sentiment_loss.compute_loss((predictions, probs))
            fluency_loss = self.gpt2_loss.compute_loss((predictions, probs))
            return fluency_loss, sentiment_loss
        
    
    def sample(self, n, std_dev, delta_start, num_leapfrogs, /, *ARGS):
        """
        Sample from given parameters using Hamiltonian Monte Carlo.
        

        Parameters
        ----------
        n : int
            number of samples to generate.
        std_dev : float32
            standard deviation for sampling velocities.
        delta : float32
            delta in sampling velocities as in ALgorithm.
        num_leapfrogs : int
            number of leapfrog steps to do.
        *ARGS : (X, y, y_1hot) as described in utils.py and NeuralNetwork model learning
            
        Returns
        -------
        samples : list of length (1 + n), comprising of list of samples (model parameters) of length (self.model.shapes)
            initial and generated samples of model parameters.

        """
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2") # debugging TODO

        n = int(n)

        # Initialize buffer.
        samples = []
        potentials = []

        # samples.append(self.embeddings.pred_embeds.clone().detach())
    
        num_accepts = 0

        prev_token_ids_list = None

        delta = delta_start
        delta_max = 0.5     #TODO
        s = 15               #TODO
        static_count = 0

        # std_dev = max(1.0, (self.embeddings.pred_embeds.numel() / 200.0) ** 0.5) # testing TODO
        if self.debug:
            print(f"std_dev: {std_dev}")

        ke_scale = 0.1 / self.embeddings.pred_embeds.size()[1] # scale kinetic energy TODO
        # ke_scale = 0.5 / self.embeddings.seq_length # scale kinetic energy TODO
        # ke_scale = 1.0 / (std_dev ** 2 * self.embeddings.seq_length)
        NLL_scale = 1.0  # scale nll part of energy

        # Open log file
        with open(self.file_name, "a") as log_file:
            for i in range(n):
                curr_velocities = self.get_sampled_velocities(std_dev)
                
                saved_embeddings = self.embeddings.pred_embeds.clone().detach()
                
                pred_embeds, _, probs = self.embeddings.forward()

                # Compute the potential energy
                potential_energy_previous = self.compute_potential_energy(pred_embeds, probs)

                # scale kinetic energy with respect to the length of sequence and by ke_scale
                kinetic_energy_previous = ke_scale * 0.5 * torch.sum(curr_velocities ** 2).item()
                
                current_potential_energy = potential_energy_previous

                if i == 0:
                    with torch.no_grad():
                        # save to samples
                        fluency_loss, sentiment_loss = self.compute_loss(pred_embeds, probs)
                        # Save all relevant data in the sample
                        accepted_token_ids = self.embeddings.get_projected_tokens()
                        token_ids_list = accepted_token_ids.squeeze().tolist() # Convert tensor to list for tokenizer.decode
                        decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
                        samples.append({
                            "pred_embeds": self.embeddings.pred_embeds.clone().detach(),
                            "decoded_text": decoded_text,
                            "fluency_loss": fluency_loss.item(),
                            "sentiment_loss": sentiment_loss.item(),
                            "potential_energy": potential_energy_previous.item(),
                            "kinetic_energy": kinetic_energy_previous,
                            "lambda": self.current_lambda.item(),
                            # "alpha": 1,
                            "Accepted": True
                        })


                # Update by multiple leapfrog steps to get a new sample.
                for step in range(num_leapfrogs):
                    #
                    curr_velocities, current_potential_energy = self.leapfrog(curr_velocities, current_potential_energy, delta, std_dev)

                potential_energy_current = current_potential_energy
                # scale kinetic energy with respect to the length of sequence and by ke_scale
                kinetic_energy_current = ke_scale * 0.5 * torch.sum(curr_velocities ** 2).item()
                

                # Metropolis-Hasting rejection sampling.
                accept_new, alpha_val = self.accept_or_reject(potential_energy_previous, potential_energy_current,
                                                kinetic_energy_previous, kinetic_energy_current)
            
                fluency_loss, sentiment_loss = self.compute_loss(pred_embeds, probs)
                if self.debug:
                    print(f"Sample {i}: Fluency Loss: {fluency_loss.item()}, "
                        f"Sentiment Loss: {sentiment_loss.item()}, "
                        f"Potential Energy: {potential_energy_current.item()}, "
                        f"Kinetic Energy: {kinetic_energy_current}, "
                        f"Total Energy: {potential_energy_current.item() + kinetic_energy_current}, "
                        f"Lambda: {self.current_lambda.item()}, "
                        f"Alpha: {alpha_val.item()}, ")

                if accept_new:
                    # Accept new samples.
                    if self.debug:
                        print("Accepting new sample")

                    accepted_token_ids = self.embeddings.get_projected_tokens()
                    token_ids_list = accepted_token_ids.squeeze().tolist() # Convert tensor to list for tokenizer.decode
                    decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
                        # print(f"Accepted Text: {decoded_text}")
                        # write_accepted_text(log_file, decoded_text, fluency_loss.item(), sentiment_loss.item(), potential_energy_current.item(), kinetic_energy_current, self.current_lambda.item(), alpha_val)

                    sampled_token_ids = self.embeddings.get_projected_tokens()
                    if prev_token_ids_list is not None and self.embeddings is not None:
                        if torch.equal(prev_token_ids_list, sampled_token_ids):
                            static_count += 1
                            if static_count <= s:
                                delta += (delta_max - delta_start) / s
                                delta = min(delta, delta_max)
                            print(f"INCREASE DELTA: {delta}")
                        else:
                            static_count = 0
                            delta = delta_start

                    prev_token_ids_list = sampled_token_ids

                    # samples.append(self.embeddings.pred_embeds.clone().detach())

                    # Save all relevant data in the sample
                    samples.append({
                        "pred_embeds": self.embeddings.pred_embeds.clone().detach(),
                        "decoded_text": decoded_text,
                        "fluency_loss": fluency_loss.item(),
                        "sentiment_loss": sentiment_loss.item(),
                        "potential_energy": potential_energy_current.item(),
                        "kinetic_energy": kinetic_energy_current,
                        "lambda": self.current_lambda.item(),
                        # "alpha": alpha_val.item(),
                        "Accepted": True
                    })
                    # potentials.append(potential_energy_current.item())
                else:
                    # Reject new samples.
                    # Need to recover model parameters back to the last sample.
                    # potentials.append(potential_energy_previous.item())
                    if self.debug:
                        print("Rejecting new sample")

                    rejected_token_ids = self.embeddings.get_projected_tokens()
                    token_ids_list = rejected_token_ids.squeeze().tolist() # Convert tensor to list for tokenizer.decode
                    decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
                        # print(f"Rejected Text: {decoded_text}")
                        # write_rejected_text(log_file, decoded_text, fluency_loss.item(), sentiment_loss.item(), potential_energy_current.item(), kinetic_energy_current, self.current_lambda.item(), alpha_val)

                    samples.append({
                        "pred_embeds": self.embeddings.pred_embeds.clone().detach(),
                        "decoded_text": decoded_text,
                        "fluency_loss": fluency_loss.item(),
                        "sentiment_loss": sentiment_loss.item(),
                        "potential_energy": potential_energy_current.item(),
                        "kinetic_energy": kinetic_energy_current,
                        "lambda": self.current_lambda.item(),
                        # "alpha": alpha_val.item(),
                        "Accepted": False
                    })

                    self.embeddings.pred_embeds.data = saved_embeddings

                if i % 10 == 0:
                    # Adaptive lambda update (gradient-based) TODO debug
                    lambda_update = self.alpha * self.lambda_grad
                    self.current_lambda = self.current_lambda + lambda_update

                    self.current_lambda.data = torch.clamp(self.current_lambda.data, min=0.0, max=300.0)

                    num_accepts = num_accepts + int(accept_new)

                if i % 100 == 0:
                    print(torch.cuda.memory_summary(device=torch.device('cuda')))
                    print(f"[INFO] Clearing cache at iteration {i}")
                    gc.collect()
                    torch.cuda.empty_cache()
                
        return samples

    