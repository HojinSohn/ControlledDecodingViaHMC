import torch

class GPT2Loss:
    def __init__(self, model, device, prompt_ids, debug=False):
        """
        Initialize the GPT-2 loss computation class.
        
        Args:
            model: Pretrained GPT-2 model
            prompt_ids: Tensor of prompt token IDs [batch_size, prompt_len]
            initial_embeddings: Initial embeddings [batch_size, sent_length, embed_dim]
            device: Device (e.g., 'cuda')
            debug: Boolean flag for debug printing
        """
        self.model = model
        self.model.eval()  # Set to evaluation mode
        for param in self.model.parameters():
            param.requires_grad = False
        self.device = device
        self.debug = debug
        self.prompt_ids = prompt_ids
        self.embed_lut = model.get_input_embeddings()  # For projection

        self.prompt_embeddings = self.embed_lut(prompt_ids)  # [batch_size, prompt_len, embed_dim]
        self.prompt_len = prompt_ids.shape[1]

    # 
    def compute_loss(self, preds):
        # Fluency term: -log P_LM(project(Y) | x)
        # shape of token_ids: (1, token length)

        # (pred_embeds, self.pred_embeds), (pred_probs, softmax_pred_probs)
        pred_embeds, pred_probs = preds

        pred_embeds = pred_embeds[0]
        pred_probs = pred_probs[0]
        
        input_embeddings = torch.cat([self.prompt_embeddings, pred_embeds], dim=1)

        outputs = self.model(inputs_embeds=input_embeddings)
        logits = outputs.logits[:, self.prompt_len-1:-1, :]
        logprobs = torch.log_softmax(logits, dim=-1)

        # Soft target: project embeddings to nearest token for loss
        loss = (-logprobs * pred_probs).sum(dim=-1).sum(dim=-1)

        return loss
