import torch
import torch.nn as nn
from finetune import GPT2SentimentClassifier
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class SentimentLoss:
    def __init__(self, model, device, prompt_ids, epsilon=0.8, debug=False):
        """
        Initialize the GPT-2 loss computation class.
        
        Args:
            model: Pretrained GPT-2 model
            device: Device (e.g., 'cuda')
            debug: Boolean flag for debug printing
        """
        self.model = model
        
        # self.classifier_model = classifier sentiment
        self.classifier_model = GPT2SentimentClassifier().to(device)

        self.classifier_model.load_state_dict(torch.load("best_sentiment_classifier.pt", map_location=torch.device(device)))

        self.classifier_model.eval()  # Set to evaluation mode
        
        self.model.eval()
        self.classifier_model.eval()

        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.classifier_model.parameters():
            param.requires_grad = False

        self.tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

        self.device = device
        self.debug = debug
        self.prompt_ids = prompt_ids
        self.embed_lut = model.get_input_embeddings()  # For projection
        self.epsilon = epsilon

        self.prompt_len = len(prompt_ids)
        self.prompt_embeddings = self.embed_lut(prompt_ids)  # [batch_size, prompt_len, embed_dim]

    # 
    def compute_loss(self, preds):
        # (pred_embeds, self.pred_embeds), _, (pred_probs, softmax_pred_probs)
        pred_embeds, pred_probs = preds

        pred_embeds = pred_embeds[0]
        pred_probs = pred_probs[0]

        input_embeddings = torch.cat([self.prompt_embeddings, pred_embeds], dim=1)

        attention_mask = torch.ones(input_embeddings.shape[0], input_embeddings.shape[1], device=self.device)

        logits, _ = self.classifier_model(input_embeddings, attention_mask=attention_mask, labels=None)

        probs = torch.softmax(logits, dim=-1)

        loss = self.epsilon - probs[:, 1]  # Assuming binary classification, we want the probability of the positive class

        return loss