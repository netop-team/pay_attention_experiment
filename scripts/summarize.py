import argparse
parser = argparse.ArgumentParser(description='Generates samples of a given instruction on a given dataset.')
parser.add_argument(
    "--delta", 
    help = "The value of delta attention you wish to add",
    required=True
)
parser.add_argument(
    "--layers",
    help="The layers you wish to augment attention",
    required=True,
)
parser.add_argument(
    "--num_generations",
    help="Number of generations for each text.",
    required=True
)

parser.add_argument(
    "--max_new_tokens",
    help="Maximum new tokens generated. Refers to https://huggingface.co/docs/transformers/en/main_classes/text_generation",
    required=True
)

parser.add_argument(
    "--model_name",
    help="Model name",
    required=True
)
args = parser.parse_args()

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, \
  BitsAndBytesConfig, GPTQConfig
import os

while "scripts" in os.getcwd():
    os.chdir("..")
import ast

from time import time
from tqdm import tqdm
import torch
from langdetect import detect
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from huggingface_hub import notebook_login
from datasets import load_dataset
import math
from typing import List, Optional, Tuple, Union
from torch import nn
from tqdm import tqdm
from IPython.display import clear_output
from copy import deepcopy

import sys
sys.path.insert(0, 'src')
sys.path.insert(0, './')
from src.utils import get_context_length, get_hf_model_name
from src.attention_saver import Mistral7BAttentionSaver

import logging
from src.logger import setup_logger

# argument parsing

DELTA_ATTENTION = float(args.delta)

# parsing layers
layers = args.layers
if layers.startswith('['):
    layers = ast.literal_eval(layers)


n_times_generation = int(args.num_generations)
max_new_tokens = int(args.max_new_tokens)
model_name = args.model_name

device = "cuda" if torch.cuda.is_available() else "cpu"


# Configure the logging
setup_logger("logger", f'logs/generated_text_delta={DELTA_ATTENTION}_{layers}_layers.log')
logger = logging.getLogger("logger")

logger.info(f"Using device {device}")
logger.info("Fetching model...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

hf_model_name = get_hf_model_name(model_name)

tokenizer = AutoTokenizer.from_pretrained(
    hf_model_name,
    cache_dir = "/Data"    
)

tunned_model = AutoModelForCausalLM.from_pretrained(
    hf_model_name,
    quantization_config = quantization_config,
    device_map="auto",
    attn_implementation="eager",
    cache_dir = "/Data" 
)

model_name = tunned_model.config._name_or_path.split("/")[1]

dir = f"data/{model_name}"

if not os.path.exists(dir):
    os.mkdir(dir)

# testing dir
pd.DataFrame([]).to_pickle(f"{dir}/test.pkl")
logger.info("Directory tested")

logger.info(
    f'''
    Running code with arguments
    DELTA_ATTENTION = {DELTA_ATTENTION},
    layers = {layers}
    num_generations = {n_times_generation},
    dir = {dir}
    '''
)

model = Mistral7BAttentionSaver(
    tunned_model,
    tokenizer,
    DELTA_ATTENTION,
    augmented_layers= layers,
    should_save_params= False
)

logger.info("Done.")


logger.info("Fetching and preprocessing dataset...")

df = load_dataset("stas/openwebtext-10k", cache_dir="/Data")['train'].to_pandas()
df["text_len"] = df["text"].apply(lambda x: len(x.split(" ")))

instruction = "Summarize in french"
df["context_length"] = (instruction + '\n' +df["text"])\
    .apply(lambda x: get_context_length(x, tokenizer))

chunks = []
for n in range (0, 10):
    samples = df.query(f"context_length > {500*n} & context_length < {500*(n+1)}")\
        .sample(20, random_state = 42)
    
    chunks.append(samples)

study_df = pd.concat(chunks)\
    .sort_values("context_length", ascending = False)

instructions = [
    "Summarize in french: ".upper(),
    # "Important: Summarize in french: ",
    # "You must summarize the following text in french: "
]

new_samples = []
for instruction in instructions:
    new_samples_df = study_df.copy()

    new_samples_df["text"] = instruction + " \n " + new_samples_df["text"]
    new_samples_df["instruction"] = instruction
    new_samples.append(new_samples_df)

new_samples_df = pd.concat(new_samples)

start_of_generation_tag = "[/INST]" if "mistral" in model_name.lower() else "<|start_header_id|>assistant<|end_header_id|>" 

logger.info("Done.")

instruction = None
tokens = None

decoded = None

results = dict()
for idx, row in study_df.iterrows():
    results[idx] = {}
    results[idx]["base_text"] = row["text"]

for generation_epoch in range(n_times_generation):
    count = 0

    t = tqdm(enumerate(new_samples_df.iterrows()), total = len(new_samples_df))
    for i, (idx, row) in t:

        prompt = row["text"]
        instruction = row['instruction']
        message = [ {"role": "user", "content": prompt}]

        template = tokenizer.apply_chat_template(
            message,
            tokenize= False
        )

        splits = template.split(instruction)
        initial_prompt = splits[0]
        context = instruction.join(splits[1:])

        assert (hash(initial_prompt+instruction+context) == hash(template)), "Error in spliting strings. Initial and final string does not match"

        initial_tokens = tokenizer.encode(initial_prompt, return_tensors='pt')
        instruction_tokens = tokenizer.encode(
            instruction, 
            return_tensors='pt', 
            add_special_tokens=False
        )
        context_tokens = tokenizer.encode(
            context, 
            return_tensors='pt',
            add_special_tokens=False
        )

        start_idx = initial_tokens.size(1)
        end_idx = start_idx + instruction_tokens.size(1)

        model.set_reference_tokens(start_idx, end_idx)
        
        tokens = torch.concat([
            initial_tokens.squeeze(), 
            instruction_tokens.squeeze(),
            context_tokens.squeeze()
        ]).unsqueeze(0)

        q = tokenizer.decode(tokens.squeeze()[start_idx: end_idx])

        assert instruction in q, "Error in tokenization. Not giving attention to correct tokens"

        tokens2 = tokenizer(template, return_tensors='pt')

        assert (abs(tokens.shape[1] - tokens2['input_ids'].shape[1]) <=5 ), "Error in tokenization. Tokens do not match"

        

        clear_output()

        if decoded is not None:

            logger.info(
                f'''
                generating text...
                sample idx = {i}
                context_length = {tokens.shape}
                instruction = {instruction}
                generation_epoch = {generation_epoch}
                delta = {DELTA_ATTENTION}
                last generated text = {decoded[0].split(start_of_generation_tag) [1]}
                '''
            )

        else:
            logger.info(f'''
                generating text...
                sample idx = {i}
                context_length = {tokens.shape}
                instruction = {instruction}
                generation_epoch = {generation_epoch}
                '''
            )

        generated_ids = tunned_model.generate(
            tokens,
            # attention_mask = tokens['attention_mask'].to("cuda"),
            max_new_tokens = max_new_tokens,
            do_sample = True,
            # temperature = 1.
        )

        decoded = tokenizer.batch_decode(generated_ids)

        print(f'''
            generated text : {decoded[0].split(start_of_generation_tag) [1]}
            '''
        )

        if not f"epoch {generation_epoch}" in results[idx]:
            results[idx][f"epoch {generation_epoch}"] = {}

        results[idx][f"epoch {generation_epoch}"][instruction] = decoded

    if not os.path.exists(f"{dir}/checkpoints"):
        os.mkdir(f"{dir}/checkpoints")
    pd.DataFrame(results).to_pickle(f"{dir}/checkpoints/{layers}_layers_generated_delta={DELTA_ATTENTION}_upper.pkl")

pd.DataFrame(results).to_pickle(f"{dir}/{layers}_layers_generated_delta={DELTA_ATTENTION}_upper.pkl")