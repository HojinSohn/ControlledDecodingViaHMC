# -*- coding: utf-8 -*-
"""
OLD VERSION
SequenceEnergy: Wrapper for sequence embeddings and energy computation
"""

import torch
import torch.nn as nn

class SequenceEnergy:
    def __init__(self, model, prompt_ids, initial_embeddings, e_target, device, lambda_energy=1.0, epsilon=0.7, debug=False):
        """
        Initialize the sequence energy wrapper.
        
        Args:
            model: Pretrained GPT-2 model
            prompt_ids: Tensor of prompt token IDs [batch_size, rompt_len]
            initial_embeddings: Initial embeddings [batch_size, sent_length, embed_dim]
            e_target: Target sentiment embedding [embed_dim]
            device: Device (e.g., 'cuda')
            lambda_energy: Energy scaling parameter
            epsilon: Sentiment threshold
        """
        self.model = model
        self.prompt_ids = prompt_ids
        self.embeddings = initial_embeddings.clone().detach().to(device).requires_grad_()
        self.seq_length = initial_embeddings.shape[1]
        self.embedding_space_size = initial_embeddings.shape[2]
        self.e_target = e_target.to(device)
        self.device = device
        self.lambda_energy = torch.tensor(lambda_energy, device=device, requires_grad=True)
        self.epsilon = epsilon
        self.embed_lut = model.get_input_embeddings()  # For projection
        self.debug = debug
        self.prompt_embeddings = self.embed_lut(self.prompt_ids)

    def compute_discrete_sentiment_score(self):
        token_ids = self.get_projected_tokens()
        projected_embeddings = self.embed_lut(token_ids)
        normalized_embeddings = torch.nn.functional.normalize(projected_embeddings.view(-1, projected_embeddings.size(-1)), dim=-1)
        normalized_target = torch.nn.functional.normalize(self.e_target, dim=-1)
        sentiment = torch.sum(normalized_embeddings * normalized_target, dim=-1).mean(-1).item()
        return sentiment

    def compute_discrete_sentiment_energy(self):
        sentiment = self.compute_discrete_sentiment_score()
        return self.lambda_energy.item() * (self.epsilon - sentiment)

    def compute_discrete_nll(self):
        # Fluency term: -log P_LM(project(Y) | x)
        # shape of token_ids: (1, token length)
        token_ids = self.get_projected_tokens()

        prompt_len = self.prompt_ids.shape[1]

        # shape of input_ids: (1, total sequence length)
        input_ids = torch.cat([self.prompt_ids, token_ids], dim=1)

        with torch.no_grad():
            outputs = self.model(input_ids)
            logits = outputs.logits[:, prompt_len-1:-1, :]
            log_probs = torch.log_softmax(logits, dim=-1)
            nll = -log_probs.gather(2, token_ids.unsqueeze(-1)).squeeze(-1).sum(dim=1) # [batch_size]

        return nll

    def compute_discrete_energy(self):
        # For acceptance
        nll = self.compute_discrete_nll()
        sentiment_energy = self.compute_discrete_sentiment_energy()
        energy = nll + sentiment_energy
        # if self.debug:
        #     print(f"Discrete Energy: {energy.item():.4f}, Sentiment: {self.compute_discrete_sentiment_score():.4f}")
        return energy

    # have to check the validity
    def compute_negative_log_likelihood(self):
        # Fluency term: -log P_LM(project(Y) | x)
        # shape of token_ids: (1, token length)

        input_embeddings = torch.cat([self.prompt_embeddings, self.embeddings], dim=1)

        prompt_len = self.prompt_ids.shape[1]

        outputs = self.model(inputs_embeds=input_embeddings)
        logits = outputs.logits[:, prompt_len-1:-1, :]

        # Soft target: project embeddings to nearest token for loss
        scores = torch.cdist(self.embeddings.view(-1, self.embeddings.size(-1)), self.embed_lut.weight)
        soft_targets = torch.softmax(-scores / 0.1, dim=-1)
        log_probs = torch.log_softmax(logits, dim=-1)
        nll = -torch.sum(soft_targets * log_probs.view(-1, log_probs.size(-1)), dim=-1).sum()

        return nll

    def compute_sentiment_score(self):
        normalized_embeddings = torch.nn.functional.normalize(self.embeddings.view(-1, self.embeddings.size(-1)), dim=-1)
        normalized_target = torch.nn.functional.normalize(self.e_target, dim=-1)
        # sentiment = torch.sum(normalized_embeddings * normalized_target, dim=-1).mean(-1)
        similarities = torch.sum(normalized_embeddings * normalized_target, dim=-1)
        k = min(2, similarities.size(0))
        top_k_similarities = torch.topk(similarities, k=k, largest=True)[0]
        sentiment = top_k_similarities.mean()
        return sentiment

    def compute_sentiment_energy(self):
        sentiment = self.compute_sentiment_score()
        sentiment_energy = self.lambda_energy * (self.epsilon - sentiment)
        return sentiment_energy

    def compute_energy(self):
        """
        Compute E(Y) = -log P_LM(project(Y) | x) + λ (ε - f(Y)).
        
        Returns:
            energy: Tensor [batch_size]
        """
        nll = self.compute_negative_log_likelihood()
        sentiment_energy = self.compute_sentiment_energy()

        return nll + sentiment_energy

    

    def compute_gradients(self):
        """Compute gradients of energy w.r.t. embeddings and lambda."""
        self.embeddings.grad = None
        self.lambda_energy.grad = None
        energy = self.compute_energy()
        # if self.debug:
        #     print(f"Pre-Backward - Energy: {energy:.4f}")
        energy.backward(retain_graph=True) # not sure
        emb_grad = self.embeddings.grad.clone() if self.embeddings.grad is not None else torch.zeros_like(self.embeddings)
        lambda_grad = self.lambda_energy.grad.clone() if self.lambda_energy.grad is not None else torch.tensor(0.0, device=self.device)
        self.embeddings.grad.zero_()
        self.lambda_energy.grad.zero_()
        return emb_grad, lambda_grad

    def get_projected_tokens(self):
        """
        Project embeddings to token IDs.
        
        Returns:
            token_ids: Tensor [batch_size, sent_length]
        """
        scores = torch.cdist(self.embeddings.view(-1, self.embeddings.size(-1)), self.embed_lut.weight)
        return scores.argmin(dim=-1).view(self.embeddings.size(0), -1)