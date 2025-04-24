from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch
import torch.nn as nn
from datasets import load_dataset
from tqdm import tqdm

class GPT2SentimentClassifier(nn.Module):
    def __init__(self, model_name="gpt2", num_labels=2):
        super().__init__()
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token

        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        self.gpt2.config.output_hidden_states = True
        self.gpt2.config.pad_token_id = tokenizer.pad_token_id

        for param in self.gpt2.parameters():
            param.requires_grad = False  # freeze GPT-2

        self.num_labels = num_labels
        self.classifier = nn.Linear(self.gpt2.config.n_embd, num_labels)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_embeds, attention_mask=None, labels=None):
        outputs = self.gpt2(inputs_embeds=input_embeds, attention_mask=attention_mask, output_hidden_states=True)
        # Use last hidden state of last token for classification
        last_hidden_state = outputs.hidden_states[-1]
        last_token = input_embeds.shape[1] - 1  # last token index
        last_token_indices = attention_mask.sum(1) - 1  # shape: (batch_size,)
        last_token_indices = last_token_indices.long()
        cls_rep = last_hidden_state[range(last_hidden_state.size(0)), last_token_indices, :]

        logits = self.classifier(self.dropout(cls_rep))

        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss()(logits, labels)
        return logits, loss
    

# Tokenize the training data
def tokenize_function(examples):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer(examples['sentence'], padding="max_length", truncation=True)

def load_data():
    dataset = load_dataset("stanfordnlp/imdb")
    train_data = dataset["train"]
    # validation_data = dataset["validation"]
    test_data = dataset["test"]

    # Apply tokenization to the entire dataset
    train_data = train_data.map(tokenize_function, batched=True)
    validation_data = validation_data.map(tokenize_function, batched=True)
    test_data = test_data.map(tokenize_function, batched=True)

    # Set the format for PyTorch
    train_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    validation_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_data.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Convert to DataLoader
    train_batch = torch.utils.data.DataLoader(train_data, batch_size=8, shuffle=True)
    validation_batch = torch.utils.data.DataLoader(validation_data, batch_size=8)
    test_batch = torch.utils.data.DataLoader(test_data, batch_size=8)

    return train_batch, validation_batch, test_batch

def train(model, batch, optimizer, device):
    train_batch, validation_batch, test_batch = batch
    model.train()
    total_loss = 0
    for batch in train_batch:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # convert input_ids to embeddings
        input_embeds = model.gpt2.get_input_embeddings()(input_ids)

        optimizer.zero_grad()

        # Forward pass
        _, loss = model(input_embeds, attention_mask=attention_mask, labels=labels)
        total_loss += loss.item()

        # Backward pass
        loss.backward()
        optimizer.step()
    return total_loss / len(train_batch)

def evaluate(model, validation_batch, device):
    model.eval()
    total_correct = 0
    total_samples = 0
    with torch.no_grad():
        for batch in validation_batch:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            input_embeds = model.gpt2.get_input_embeddings()(input_ids)
            logits, _ = model(input_embeds, attention_mask=attention_mask)
            predictions = torch.argmax(logits, dim=-1)
            total_correct += (predictions == labels).sum().item()
            total_samples += labels.size(0)
    return total_correct / total_samples


if __name__ == "__main__":
    print("Loading model...")
    classifier = GPT2SentimentClassifier()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier.to(device)
    
    # Load data
    train_batch, validation_batch, test_batch = load_data()

    # Define optimizer
    optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5)

    min_loss = float('inf')
    # Train the model
    for epoch in range(5):  # Example: 3 epochs
        print(f"Epoch {epoch + 1}/{5}")
        avg_loss = train(classifier, (train_batch, validation_batch, test_batch), optimizer, device)
        print(f"Epoch {epoch + 1}, Loss: {avg_loss}")
        val_acc = evaluate(classifier, validation_batch, device)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        if avg_loss < min_loss:
            min_loss = avg_loss
            torch.save(classifier.state_dict(), "best_movie_sentiment_classifie2.pt")
            print("Model saved.")
        else:
            print("No improvement, not saving the model.")
        

