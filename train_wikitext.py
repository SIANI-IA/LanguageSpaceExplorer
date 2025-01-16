import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, AdamW
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import random
from sklearn.metrics import accuracy_score, f1_score

print("Initializing data preparation...")

# Load the datasets
print("Loading WikiText-2 dataset...")
wikitext = load_dataset("wikitext", "wikitext-2-raw-v1")
print("WikiText-2 loaded.")

# Initialize the GPT-2 tokenizer
print("Initializing GPT-2 tokenizer...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
print("Tokenizer initialized with pad_token set to eos_token.")

# Tokenize WikiText-2
print("Tokenizing WikiText-2 dataset...")
wikitext_tokenized = wikitext.map(
    lambda batch: tokenizer(batch['text'], truncation=True, padding='max_length', max_length=128),
    batched=True
)
print("WikiText-2 tokenization completed.")


# Define batch size
batch_size = 16
print(f"Batch size set to {batch_size}.")

# Create DataLoaders for WikiText-2
print("Creating DataLoader for WikiText-2 training set...")
wikitext_train_loader = DataLoader(wikitext_tokenized['train'], batch_size=batch_size, shuffle=True)
print("Creating DataLoader for WikiText-2 validation set...")
wikitext_val_loader = DataLoader(wikitext_tokenized['validation'], batch_size=batch_size)
print("WikiText-2 DataLoaders created.")

# Fix random seeds for reproducibility
print("Fixing random seeds for reproducibility...")
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
print("Random seeds fixed.")

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Initialize GPT-2 models for both experiments
print("Initializing GPT-2 models for both experiments...")
model_cf = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
model_dr = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=2).to(device)
print("GPT-2 models initialized.")

# Common hyperparameters
learning_rate = 3e-5
epochs = 3
optimizer_type = "SGD"
momentum = 0.9
print(f"Hyperparameters set: Learning Rate={learning_rate}, Epochs={epochs}, Optimizer=SGD with Momentum={momentum}")

# Define Optimizers
optimizer_cf = optim.SGD(model_cf.parameters(), lr=learning_rate, momentum=momentum)
optimizer_dr = optim.SGD(model_dr.parameters(), lr=learning_rate, momentum=momentum)
print("Optimizers defined for both experiments.")

# Define loss function
criterion = nn.CrossEntropyLoss()
print("Loss function defined: CrossEntropyLoss")

# Step 4: Baseline Training on Task 1 (WikiText-2)
print("\n--- Training on Task 1: WikiText-2 ---")
print("Training GPT-2 on WikiText-2 to establish baseline performance.")

model_cf.train()
for epoch in range(epochs):
    total_loss = 0
    for batch in wikitext_train_loader:
        inputs = torch.tensor(batch['input_ids']).to(device)
        attention = torch.tensor(batch['attention_mask']).to(device)
        labels = inputs  # Language modeling labels
        outputs = model_cf(input_ids=inputs, attention_mask=attention, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer_cf.step()
        optimizer_cf.zero_grad()
        total_loss += loss.item()
    avg_loss = total_loss / len(wikitext_train_loader)
    print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")

print("Baseline training on WikiText-2 completed.")