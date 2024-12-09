import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch
import torch.nn.functional as F
import transformer_lens
from datasets import load_dataset, concatenate_datasets 
from tqdm import tqdm
import re
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import pickle
import transformer_lens.utils as utils
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from typing import List, Tuple
from transformers import PreTrainedModel
from transformer_lens import HookedTransformer
from collections import defaultdict

print("Device:", device)


def load_model_and_tokenizer(model_name: str) -> Tuple:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return tokenizer, model

def transform_to_lens_input(tokenizer, model_base: str, model_ft: PreTrainedModel = None) -> Tuple:
    return HookedTransformer.from_pretrained(
        model_base,
        hf_model=model_ft, 
        tokenizer=tokenizer,
        device=device,
        move_to_device=True
    )

def process_activations(samples: List[str], model: HookedTransformer, tokenizer, num_layers: int, db_activations: defaultdict, task_name: str = None):
    """
    Process activations for a list of samples using a model and store the results in the provided activations database.

    Args:
    - samples (list): List of text samples to process.
    - model (object): Model with a `run_with_cache` method to obtain activations.
    - num_layers (int): Number of layers of the model to extract activations from.
    - db_activations (dict): Dictionary to store activations and task labels.
    - task_name (str): Name of the task to assign to the activations. Default is 'tel'.

    Returns:
    - db_activations (dict): Updated dictionary with activations and task labels.
    """
    
    for text in tqdm(samples, desc="Processing DB"):
        
        db_activations["text"].append(text)
        db_activations["language"].append(task_name)
        ids = tokenizer(text, return_tensors="pt")["input_ids"][0].cpu().numpy()
        db_activations["tokens"].append(ids)
        
        _, activations = model.run_with_cache(text)
        db_activations["embeddings"].append(activations[f'hook_embed'].cpu().numpy())
    
        
        for layer in range(num_layers):
            block_act_fn = np.squeeze(activations[f'blocks.{layer}.mlp.hook_post'].cpu().numpy(), axis=0)
            db_activations[f"mlp_act_{layer}"].append(block_act_fn)
            block_state_fn = np.squeeze(activations[f'blocks.{layer}.hook_resid_post'].cpu().numpy(), axis=0)
            db_activations[f"states_{layer}"].append(block_state_fn)

    return db_activations 

MODEL_NAME = "meta-llama/Llama-3.2-1B"
FOLDER = "data/01-raw/phrases_with_languages.csv"
#main
if __name__ == "__main__":
    # Load the dataset
    dataset = pd.read_csv(FOLDER)
    activation_tracker_dataset = defaultdict()
    activation_tracker_dataset["text"] = []
    activation_tracker_dataset["language"] = []
    activation_tracker_dataset["tokens"] = []
    activation_tracker_dataset["embeddings"] = []
    # Load the model and tokenizer
    tokenizer, model = load_model_and_tokenizer(MODEL_NAME)
    num_layers = model.config.num_hidden_layers
    print(f"Model and tokenizer loaded with {num_layers} layers")
    for layer in range(num_layers):
        activation_tracker_dataset[f"mlp_act_{layer}"] = []
        activation_tracker_dataset[f"states_{layer}"]  = []

    # Transform the model and tokenizer to the lens input
    model_hooked = transform_to_lens_input(tokenizer, MODEL_NAME)
    print("Model transformed to lens input")

    language = dataset["Language"].unique()
    for lang in language:
        dataset_lang = dataset[dataset["Language"] == lang]
        print("Language:", lang)
        db_activations = process_activations(
            dataset_lang["Phrase"].tolist(), 
            model_hooked,
            tokenizer,
            num_layers, 
            activation_tracker_dataset, 
            task_name=lang,
        )

    # to pandas
    activation_tracker_dataset_df = pd.DataFrame(activation_tracker_dataset)
    print(activation_tracker_dataset_df.head())


    # Save the activations database in pkl
    with open("data/02-processed/activation_tracker.pkl", "wb") as f:
        pickle.dump(activation_tracker_dataset_df, f)