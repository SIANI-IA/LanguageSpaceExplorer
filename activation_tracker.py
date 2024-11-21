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

print("Device:", device)