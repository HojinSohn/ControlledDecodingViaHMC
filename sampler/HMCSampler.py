''' 

'''
import torch 
import numpy as np

DEBUG_ACCEPT = False


class HMCSampler:
    
    # sequence_embeddings = embeddings of the output tokens
    def __init__(self, sequence_energy, rng, device, alpha=0.1):
        self.sequence_energy = sequence_energy
        
        self.rng = rng

        self.alpha = alpha
        
        self.device = device

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
        velocities = [torch.normal(mean=0, std=stddv, size=self.embedding_space_size, device=self.device) for _ in self.seq_length]
        return velocities
        
        # raise NotImplementedError("Complete Implementation")
        
    
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
        emb_grad, lambda_grad = self.sequence_energy.compute_gradient()
        velocities_half = velocities - 0.5 * delta * emb_grad
        new_lambda = self.sequence_energy.lambda_energy + 0.5 * self.alpha * lambda_grad # right?

        # Full-step update for embeddings
        self.sequence_energy.embeddings = self.sequence_energy.embeddings + delta * velocities_half

        # Another half-step update for velocities
        emb_grad, lambda_grad = self.sequence_energy.compute_gradient()
        velocities = velocities_half - 0.5 * delta * emb_grad
        new_lambda = new_lambda + 0.5 * self.alpha * lambda_grad # right?
        self.sequence_energy.lambda_energy = max(new_lambda, 0)

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
        
        alpha = min(1, np.exp(potential_energy_previous + kinetic_energy_previous - potential_energy_current - kinetic_energy_current))
        if np.random.uniform(low=0, high=1) <= alpha:
            return True
        else:
            return False
        
        # raise NotImplementedError("Complete Implementation")
    
    def sample(self, n, std_dev, delta, num_leapfrogs, /, *ARGS):
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
        n = int(n)

        # Initialize buffer.
        samples = []
        potentials = []
        
        Y = self.sequence_energy.embeddings.clone()  # Current state

        for i in range(n):
            curr_velocities = self.get_sampled_velocities(std_dev)

            Y_old = Y.clone()

            potential_energy_previous = self.sequence_energy.compute_energy()
            kinetic_energy_previous = sum(0.5 * torch.sum(velocity ** 2).item() for velocity in curr_velocities)

            # Update by multiple leapfrog steps to get a new sample.
            for _ in range(num_leapfrogs):
                #
                curr_velocities = self.leapfrog(curr_velocities, delta, *ARGS)
            
            potential_energy_current = self.sequence_energy.compute_energy()
            kinetic_energy_current = sum(0.5 * torch.sum(new_velocity ** 2).item() for new_velocity in curr_velocities)
            
            # Metropolis-Hasting rejection sampling.
            accept_new = self.accept_or_reject(potential_energy_previous, potential_energy_current,
                                               kinetic_energy_previous, kinetic_energy_current)
            
            if accept_new:
                # Accept new samples.
                samples.append(self.sequence_energy.embeddings)
                potentials.append(potential_energy_current)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Accept", "{:.6f}".format(potential_energy_current)),
                )
                Y = self.sequence_energy.embeddings
            else:
                # Reject new samples.
                # Need to recover model parameters back to the last sample.
                samples.append(samples[-1])
                potentials.append(potential_energy_previous)
                print(
                    "{:>3d} {:>6s} {:>8s}"
                    .format(i, "Reject", "{:.6f}".format(potential_energy_previous)),
                )
                Y = Y_old
            self.sequence_energy.embeddings = Y  # Update for next iteration
            samples.append(Y.clone().detach())
            num_accepts = num_accepts + int(accept_new)

    