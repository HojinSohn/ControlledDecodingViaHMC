# -*- coding: utf-8 -*-
"""
SequenceEnergy: Wrapper for sequence embeddings and energy computation
"""

import torch
import torch.nn as nn

from util import compute_negative_log_likelihood

class SequenceEnergy:
    def __init__(self, model, prompt_ids, initial_embeddings, e_target, device, lambda_energy=1.0, epsilon=0.7):
        """
        Initialize the sequence energy wrapper.
        
        Args:
            model: Pretrained GPT-2 model
            prompt_ids: Tensor of prompt token IDs [prompt_len]
            initial_embeddings: Initial embeddings [batch_size, sent_length, embed_dim]
            e_target: Target sentiment embedding [embed_dim]
            device: Device (e.g., 'cuda')
            lambda_energy: Energy scaling parameter
            epsilon: Sentiment threshold
        """
        self.model = model
        self.prompt_ids = prompt_ids
        self.embeddings = initial_embeddings.clone().to(device).requires_grad_(True)  # Y
        self.e_target = e_target.to(device)
        self.device = device
        self.lambda_energy = torch.tensor(lambda_energy, device=device, requires_grad=True)
        self.epsilon = epsilon
        self.embed_lut = model.get_input_embeddings()  # For projection

    def compute_negative_log_likelihood(self, model, prompt_ids, output_embeddings):
        # Fluency term: -log P_LM(project(Y) | x)
        token_ids = self.get_projected_tokens()
        input_ids = torch.cat([prompt_ids.unsqueeze(0), token_ids], dim=1)
        with torch.no_grad():
            logits = model(input_ids).logits[:, len(prompt_ids)-1:-1, :]
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1)  # [batch_size]
        return nll

    def compute_energy(self):
        """
        Compute E(Y) = -log P_LM(project(Y) | x) - λ (ε - f(Y)).
        
        Returns:
            energy: Tensor [batch_size]
        """
        nll = self.compute_negative_log_likelihood(self.model, self.prompt_ids, self.embeddings)
        sentiment = torch.nn.functional.normalize(self.embeddings.view(-1, self.embeddings.size(-1)), dim=-1).matmul(
            torch.nn.functional.normalize(self.e_target, dim=-1)).view(self.embeddings.size(0), -1).mean(dim=1)
        return nll - self.lambda_energy * (self.epsilon - sentiment)

    def compute_gradients(self):
        """Compute gradients of energy w.r.t. embeddings and lambda."""
        energy = self.compute_energy().sum()  # Sum over batch for scalar
        gradients = torch.autograd.grad(energy, [self.embeddings, self.lambda_energy])
        return gradients  # (emb_grad, lambda_grad)

    def get_projected_tokens(self):
        """
        Project embeddings to token IDs.
        
        Returns:
            token_ids: Tensor [batch_size, sent_length]
        """
        scores = torch.cdist(self.embeddings.view(-1, self.embeddings.size(-1)), self.embed_lut.weight)
        return scores.argmin(dim=-1).view(self.embeddings.size(0), -1)