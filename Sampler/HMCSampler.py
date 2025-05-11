''' 
OLD VERSION
'''
import torch 
import numpy as np
from util import *
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DistilBertForSequenceClassification, DistilBertTokenizer


DEBUG_ACCEPT = False


class HMCSampler:
    
    # sequence_embeddings = embeddings of the output tokens
    def __init__(self, sequence_energy, rng, device, alpha=0.1, debug=False, log_file_name='log_data/sample_log'):
        self.sequence_energy = sequence_energy
        
        self.rng = rng

        self.alpha = alpha
        
        self.device = device

        self.debug = debug

        self.lambda_grad = 0.0

        self.file_name = log_file_name

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
        return torch.normal(mean=0, std=stddv, size=self.sequence_energy.embeddings.size(), device=self.device)
        
    
    def leapfrog(self, velocities, delta, /, *ARGS):
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

        # Half-step update for velocities
        emb_grad, lambda_grad = self.sequence_energy.compute_gradients()

        '''
        emb_grad:           torch.Size([1, 20, 768])
        lambda_grad:        scalar value
        velocities:         torch.Size [1, 20, 768]
        embeddings:         torch.Size([1, 20, 768])
        '''

        # DEBUG
        # if self.debug:
        #     print(f"emb_grad shape: {emb_grad.shape}")
        #     print(f"lambda_grad shape: {lambda_grad.shape}")
        #     print(f"lambda_grad: {lambda_grad}")
        #     print(f"velocities len: {velocities.shape}")
        #     print(f"self.sequence_energy.embeddings shape : {self.sequence_energy.embeddings.shape}")

        # Remove batch dimension
        emb_grad = emb_grad.squeeze(0)

        # Half step update for velocity / lambda update
        velocities_half = velocities - 0.5 * delta * emb_grad
        # new_lambda = self.sequence_energy.lambda_energy + 0.5 * self.alpha * lambda_grad # right?

        # Full-step update for embeddings
        self.sequence_energy.embeddings = (
            self.sequence_energy.embeddings + delta * velocities_half
        ).detach().clone().requires_grad_() 
        # reset lambda to remove from graph
        self.sequence_energy.lambda_energy = self.sequence_energy.lambda_energy.detach().clone().requires_grad_()

        # Another half-step update for velocities
        emb_grad, lambda_grad = self.sequence_energy.compute_gradients()

        # save it for adaptive update
        self.lambda_grad = lambda_grad

        emb_grad = emb_grad.squeeze(0)

        # Half step update for velocity / lambda update
        velocities = velocities_half - 0.5 * delta * emb_grad
        # new_lambda = new_lambda + 0.5 * self.alpha * lambda_grad # right?
        # self.sequence_energy.lambda_energy.data = torch.tensor(max(new_lambda, 0), dtype=torch.float32, device=self.sequence_energy.device)

        return velocities
        
        

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

        samples.append(self.sequence_energy.embeddings)

        with torch.no_grad():  # No need for gradients here
            # initial_nll = self.sequence_energy.compute_negative_log_likelihood().item()
            initial_nll = self.sequence_energy.compute_negative_log_likelihood()
        
        if self.debug:
            print("\n=== Initial State ===")
            print(f"NLL: {initial_nll:.4f}")
            print("=====================\n")

        Y = self.sequence_energy.embeddings.clone()  # Current state

        num_accepts = 0

        if self.debug:
            print(f"self.sequence_energy.embeddings.size(): {self.sequence_energy.embeddings.size()}")

        lamba_records = []
        nll_records = []
        sentiment_records =[]
        record_nll = 0.0
        record_se = 0.0
        record_ss = 0.0
        record_ld = 0.0
        record_e = 0.0
        prev_token_ids_list = None

        delta = delta_start
        delta_max = 0.5     #TODO
        s = 15               #TODO
        static_count = 0

        ke_scale = 0.1 / self.sequence_energy.embeddings.size()[1] # scale kinetic energy
        # NLL_scale = 1 / self.sequence_energy.embeddings.size()[1]  # scale nll part of energy
        NLL_scale = 1.0  # scale nll part of energy

        # Open log file
        with open(self.file_name, "a") as log_file:
            write_file_header(log_file)
            for i in range(n):
                curr_velocities = self.get_sampled_velocities(std_dev)
                if self.debug:
                    print(f"Iter {i}: Velocity Norm: {torch.norm(curr_velocities).item():.4f}")  # Check velocity magnitude

                Y_old = Y.clone()

                potential_energy_previous = self.sequence_energy.compute_energy()

                # scale kinetic energy with respect to the length of sequence and by ke_scale
                kinetic_energy_previous = ke_scale * 0.5 * torch.sum(curr_velocities ** 2).item()
                

                if self.debug:
                    print(f"Iter {i}: Pre-Leapfrog - Potential: {potential_energy_previous.item():.4f}, Kinetic: {kinetic_energy_previous:.4f}")
                # Update by multiple leapfrog steps to get a new sample.
                for step in range(num_leapfrogs):
                    #
                    curr_velocities = self.leapfrog(curr_velocities, delta, *ARGS)
                    if self.debug and step == num_leapfrogs - 1:
                        with torch.no_grad():
                            record_nll = self.sequence_energy.compute_negative_log_likelihood()
                            record_se = self.sequence_energy.compute_sentiment_energy()
                            record_ss = self.sequence_energy.compute_sentiment_score()
                            record_ld = self.sequence_energy.lambda_energy
                            record_e = self.sequence_energy.compute_energy()
                            print(f"{step}th LEAPFROG: {record_nll}\t\t\t{record_se}\t\t\t{record_ld}\t\t\t{record_ss}")

                    if self.debug and step == num_leapfrogs - 1:  # Print last step
                        distance = torch.norm(self.sequence_energy.embeddings - Y_old).item()
                        print(f"Iter {i}: Post-Leapfrog Distance: {distance:.6f}")
                
                potential_energy_current = self.sequence_energy.compute_energy()
                
                # scale kinetic energy with respect to the length of sequence and by ke_scale
                kinetic_energy_current = ke_scale * 0.5 * torch.sum(curr_velocities ** 2).item()

                # Metropolis-Hasting rejection sampling.
                accept_new, alpha_val = self.accept_or_reject(potential_energy_previous, potential_energy_current,
                                                kinetic_energy_previous, kinetic_energy_current)
                
                with torch.no_grad():
                    nll_curr = self.sequence_energy.compute_negative_log_likelihood()
                    sentiment_curr = self.sequence_energy.compute_sentiment_score()

                if accept_new:
                    # Accept new samples.
                    samples.append(self.sequence_energy.embeddings.clone().detach())
                    potentials.append(potential_energy_current)
                    if self.debug:
                        print(f"Accept Current energy => Potential: {potential_energy_current.item()}, "
                            f"Kinetic: {kinetic_energy_current}, "
                            f"Total: {potential_energy_current.item() + kinetic_energy_current}")

                        print(f"Previous energy => Potential: {potential_energy_previous.item()}, "
                            f"Kinetic: {kinetic_energy_previous}, "
                            f"Total: {potential_energy_previous.item() + kinetic_energy_previous}")

                        print(f"negative log likelihood: {nll_curr}")
                        print(f"sentiment energy: {potential_energy_current.item() - nll_curr}")
                        
                        accepted_token_ids = self.sequence_energy.get_projected_tokens()
                        token_ids_list = accepted_token_ids.squeeze().tolist() # Convert tensor to list for tokenizer.decode
                        decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
                        # print(f"Accepted Text: {decoded_text}")
                        write_accepted_text(log_file, decoded_text, nll_curr.item(), sentiment_curr, potential_energy_current.item(), kinetic_energy_current, self.sequence_energy.lambda_energy.item(), alpha_val)
                    Y = self.sequence_energy.embeddings

                    sampled_token_ids = self.sequence_energy.get_projected_tokens()
                    if prev_token_ids_list is not None and Y is not None:
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

                    lamba_records.append(record_ld)
                    nll_records.append(record_nll)
                    sentiment_records.append(record_ss)
                else:
                    # Reject new samples.
                    # Need to recover model parameters back to the last sample.
                    potentials.append(potential_energy_previous)
                    if self.debug:
                        print(f"Reject Current energy => Potential: {potential_energy_current.item()}, "
                            f"Kinetic: {kinetic_energy_current}, "
                            f"Total: {potential_energy_current.item() + kinetic_energy_current}")

                        print(f"Previous energy => Potential: {potential_energy_previous.item()}, "
                            f"Kinetic: {kinetic_energy_previous}, "
                            f"Total: {potential_energy_previous.item() + kinetic_energy_previous}")

                        rejected_token_ids = self.sequence_energy.get_projected_tokens()
                        token_ids_list = rejected_token_ids.squeeze().tolist() # Convert tensor to list for tokenizer.decode
                        decoded_text = tokenizer.decode(token_ids_list, skip_special_tokens=True)
                        # print(f"Rejected Text: {decoded_text}")
                        write_rejected_text(log_file, decoded_text, nll_curr.item(), sentiment_curr, potential_energy_current.item(), kinetic_energy_current, self.sequence_energy.lambda_energy.item(), alpha_val)

                    Y = Y_old
                self.sequence_energy.embeddings = Y.detach().requires_grad_()  # Update for next iteration

                if self.debug:
                    print(f"Iter {i}: Final Y Norm: {torch.norm(Y).item():.4f}")

                # Adaptive lambda update (gradient-based) TODO debug
                # lambda_update = self.alpha * self.lambda_grad
                # new_lambda = torch.max(
                #     self.sequence_energy.lambda_energy + lambda_update,
                #     torch.tensor(0.0, device=self.device)
                # )

                # # Correct reassignment that breaks from old computation graph
                # self.sequence_energy.lambda_energy = new_lambda.detach().clone().requires_grad_()
                
                # if self.debug:
                #     print(f"Iter {i}: Lambda energy: {new_lambda.detach()}")
                #     print(f"Iter {i}: Lambda energy update: {lambda_update}")
                #     print()

                num_accepts = num_accepts + int(accept_new)
                
        return samples, lamba_records, nll_records, sentiment_records

    