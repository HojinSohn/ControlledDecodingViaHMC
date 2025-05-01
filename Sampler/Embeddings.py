'''
This class is wrapper for the token embbedings being optimized wrt potential energy.
'''

import torch.nn as nn
import torch
import torch.nn.functional as F

class Embeddings(nn.Module): 
    def __init__(
        self,
        embed_dim,
        embed_lut,
        seq_length,
        batch_size,
        device,
        init_value=None,
        metric="dot",
    ):
        super(Embeddings, self).__init__()
        # [batch_size, seq_length, embed_space]
        # batch size should be 1 // for simplicity now
        # sequence of token embeddings are learnable parameters, optimized wrt potential energy.
        self.pred_embeds = nn.Parameter(torch.Tensor(batch_size, seq_length, embed_dim).to(device))
        self.seq_length = seq_length
        self.device = device
        self.metric = metric   
        self.temperature = 0.1
        self.embed_lut = embed_lut
        
        if self.metric == "cosine":
            self.target_embeds = torch.nn.functional.normalize(embed_lut.weight.data, p=2, dim=-1)
        else:
            self.target_embeds = embed_lut.weight.data
        
        self.initialize(init_value=init_value)

    def forward(self):
        # [batch_size, seq_length, embed_space]
        pred_embeds = self.pred_embeds    

        # [batch_size, seq_length, vocab_size]
        pred_logits = get_emb_score(self.metric, pred_embeds, self.target_embeds)
        
        # [batch_size, seq_length, vocab_size]
        pred_probs = F.softmax(pred_logits / self.temperature, dim=-1)
        
        # [batch_size, seq_length, 1]
        _, index = pred_probs.max(-1, keepdim=True)
        
        # [batch_size, seq_length, 1]
        # sequence of token ids (predictions) the most probable token ids
        predictions = index.squeeze(-1)
        
        # [batch_size, seq_length, vocab_size]
        softmax_pred_probs = pred_probs
        
        # [batch_size, seq_length, vocab_size] one hot encodding
        y_hard = torch.zeros_like(pred_probs).scatter_(-1, index, 1.0)
        
        # [batch_size, seq_length, vocab_size] one hot encodding, but differentiable
        pred_probs = y_hard - pred_probs.detach() + pred_probs

        # [batch_size, seq_length, embed_space]
        # sequence of token embeddings (pred_embeds) the most probable token embeddings
        n_pred_embs = self.embed_lut(predictions)

        pred_embeds = pred_embeds + (n_pred_embs - pred_embeds).detach()
    
        return (pred_embeds, self.pred_embeds), predictions, (pred_probs, softmax_pred_probs)


    def initialize(self, init_value=None):
        # [batch_size, seq_length, embed_space]
        if init_value is not None:
            self.pred_embeds.data.copy_(init_value.data)
        else: 
            torch.nn.init.zeros_(self.pred_embeds)
            
        
    def printparams(self):
        print(self.pred_embeds)

    def get_projected_tokens(self):
        # [batch_size, seq_length, vocab_size]
        pred_logits = get_emb_score(self.metric, self.pred_embeds, self.target_embeds)
        
        # [batch_size, seq_length, vocab_size]
        pred_probs = F.softmax(pred_logits / self.temperature, dim=-1)
        
        # [batch_size, seq_length, 1]
        _, index = pred_probs.max(-1, keepdim=True)
        
        # [batch_size, seq_length, 1]
        # sequence of token ids (predictions) the most probable token ids
        predictions = index.squeeze(-1)
        
        return predictions

def get_emb_score(metric, pred_emb, tgt_out_emb):
    if metric == "l2": 
        scores = (pred_emb.unsqueeze(2) - tgt_out_emb.unsqueeze(0))
        scores = -(scores*scores).sum(dim=-1)

    elif metric == "cosine": 
        pred_emb_unitnorm = torch.nn.functional.normalize(pred_emb, p=2, dim=-1)
        scores = pred_emb_unitnorm.matmul(tgt_out_emb.t())
    
    else: 
        return pred_emb.matmul(tgt_out_emb.t())
    
    return scores