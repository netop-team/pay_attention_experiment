import argparse
parser = argparse.ArgumentParser(description='Generates samples of a given instruction on a given dataset.')
parser.add_argument(
    "--delta", 
    help = "The value of delta attention you wish to add",
    required=True
)
args = parser.parse_args()
DELTA_ATTENTION = float(args.delta)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, \
  BitsAndBytesConfig, GPTQConfig
import os

while "notebooks" in os.getcwd():
    os.chdir("..")


import sys
sys.path.insert(0, './')

from time import time
from pathlib import Path
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
import warnings
from bert_score import BERTScorer
warnings.filterwarnings("ignore")
from copy import deepcopy
from openai import OpenAI

from src.utils import rotate_half, apply_rotary_pos_emb, repeat_kv, \
    get_context_length, insert_needle, read_txt_files, split_text_into_chunks

import logging
from src.logger import setup_logger
# Configure the logging
setup_logger("logger", f'logs/a_needle_in_a_haystack_delta=0_layers.log')
logger = logging.getLogger("logger")

from src.attention_saver import Mistral7BAttentionSaver


needle = "\n\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n\n"
question = "Your objective is to answer the following question based on the context: \nWhat is the best thing to do in San Francisco? \nDon't give information outside the document or repeat our findings"

logger.info("Loading model...")

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    cache_dir = "/Data"    
)


base_model = AutoModelForCausalLM.from_pretrained(
    "mistralai/Mistral-7B-Instruct-v0.1",
    quantization_config = quantization_config,
    device_map="auto",
    attn_implementation="eager",
    cache_dir = "/Data"
)

logging.info("Done")

model_name = base_model.config._name_or_path.split("/")[1]

model = Mistral7BAttentionSaver(
    base_model,
    tokenizer,
    delta_attention=DELTA_ATTENTION,
    should_save_params= False
)

device = "cuda" if torch.cuda.is_available() else "cpu"

df = load_dataset("stas/openwebtext-10k", cache_dir="/Data")['train'].to_pandas()
df["text_len"] = df["text"].apply(lambda x: len(x.split(" ")))

# df = read_txt_files("data/PaulGrahamEssays")

df['context_length'] = df['text'].apply(get_context_length, tokenizer = tokenizer)

# chunks = []
# for n in range (9):
#     samples = df.query(f"context_length > {500*n} & context_length < {500*(n+1)}")\
#         .sample(25, random_state = 42)
    
#     chunks.append(samples)

# study_df = pd.concat(chunks)\
#     .sort_values("context_length", ascending = False)

# study_df = df.query("context_length < 5_000")\
#     .sort_values("context_length" ,ascending = False)

large_text_df = df.query("context_length > 6000")\
    .sample(50, random_state = 33)

samples = []
for idx, row in tqdm(large_text_df.iterrows(), total = len(large_text_df)):
    chunks = split_text_into_chunks(row.text, range(500, 5_000, 500), tokenizer)
    samples.append(pd.Series(chunks))


study_df = pd.DataFrame(
    pd.concat(samples).reset_index(drop = True)
).rename(columns = {0: 'text'})

study_df['context_length'] = study_df['text']\
    .apply(get_context_length, tokenizer = tokenizer)    

study_df['context_length_bins'] = pd.cut(
    study_df['context_length'],
    range(0, 5000, 500)
)



all_df = []

for depth_percent in tqdm(range(0, 110, 10)):

    percent_df = study_df.apply(
        insert_needle, 
        depth_percent = depth_percent, 
        question = question,
        needle = needle, 
        insert_in = 'both',
        axis = 1
    )

    all_df.append(percent_df)

needle_in_a_haystack_df = pd.concat(all_df)\
    .reset_index(drop = True)

print(f"Dataframe size {needle_in_a_haystack_df.shape}" )

generate_kwargs = {
    'max_new_tokens': 30,
    'max_length': None,
    'num_beams': 1,
    'do_sample': False,
    'temperature': None,
    'top_p': None,
    'top_k': None,
}


generated_texts = []
base_filepath = "data/study-02-needle_in_a_haystack/{model_name}/generated_text_delta={delta_attention}.parquet"
checkpoint_filepath = "data/study-02-needle_in_a_haystack/{model_name}//generated_text_delta={delta_attention}.parquet"
decoded = None

for i, (idx, row) in enumerate(tqdm(needle_in_a_haystack_df.iterrows())):
    # load dataset

    # Prepare files with predictions, prompt, and generation configurations
    checkpoint_file = checkpoint_filepath.format(
        model_name = model_name,
        delta_attention = DELTA_ATTENTION
    )
    checkpoint_file = Path(checkpoint_file)
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)

    _needle = row['needle']
    _question = row['question']
    input_text = row['new_text']
    context_length = row['context_length']
    depth = row['depth']
    
    clear_output()

    message = [
        {"role": "user", "content": input_text},
        # {"role": "assistant", "content": "here is the most relevant sentence in the context:"}    
    ]
    template = tokenizer.apply_chat_template(message, tokenize = False)

    initial_prompt = template.split(_needle)[0]
    context = template.split(_needle)[1]

    assert (hash(initial_prompt+_needle+context) == hash(template)), "Error in spliting strings. Initial and final string does not match"

    initial_tokens = tokenizer.encode(initial_prompt, return_tensors='pt')
    needle_tokens = tokenizer.encode(_needle, return_tensors='pt')
    context_tokens = tokenizer.encode(context, return_tensors='pt')

    start_idx = initial_tokens.size(1)
    end_idx = start_idx + needle_tokens.size(1) - 1

    model.set_reference_tokens(start_idx, end_idx)
    
    tokens = torch.concat([
        initial_tokens.squeeze(), 
        needle_tokens.squeeze()[1:],
        context_tokens.squeeze()[1:]
    ]).unsqueeze(0)

    q = tokenizer.decode(tokens.squeeze()[start_idx: end_idx])

    assert _needle in q, "Error in tokenization. Not giving attention to correct tokens"

    tokens2 = tokenizer(template, return_tensors='pt')

    assert (abs(tokens.shape[1] - tokens2['input_ids'].shape[1]) <=5 ), "Error in tokenization. Tokens do not match"

    print(f'''
        generating text...
        question = {_question}
        sample idx = {i}
        delta = {DELTA_ATTENTION}
        context_length = {tokens.shape}
        depth = {row['depth']}
        start_idx = {start_idx}
        end_idx = {end_idx}
        last generated text = {decoded[0].split("[/INST]") [1] if decoded is not None else 'None'}
        '''
    )

    with torch.no_grad():
        output = model.generate(tokens, **generate_kwargs)

    output = tokenizer.batch_decode(output)
    decoded = output

    generated_texts.append({
        "generated_text": decoded, 
        "target" : _needle, 
        "question": _question, 
        "context_length": context_length, 
        'depth' : depth,
        "text_index": idx    
    })

    if i % 100 ==0:
        pd.DataFrame(generated_texts).to_parquet(checkpoint_file)

outfile = base_filepath.format(
    model_name = model_name,
    delta_attention = DELTA_ATTENTION
)
outfile = Path(outfile)
outfile.parent.mkdir(parents=True, exist_ok=True)

pd.DataFrame(generated_texts).to_parquet(outfile)