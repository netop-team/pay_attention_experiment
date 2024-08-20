import torch
from transformers import AutoTokenizer
import os
from langdetect import detect
import numpy as np
from typing import List, Dict
import glob
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from copy import deepcopy
import json
import re

def get_hf_model_name(casual_name : str):
    if "mistral" in casual_name :
        return "mistralai/Mistral-7B-Instruct-v0.1"
    
    if "llama" in casual_name:
        return "meta-llama/Meta-Llama-3-8B-Instruct"

# pattern = r'\{(?:[^{}]+|(?R))+\}'
pattern = r"\{(.*)\}"

def get_text_whithin_braces(x):
    match = re.search(pattern, x, re.DOTALL)
    if match:
        return match.group(0)

    return None

def convert_to_json(x):
        if x is None:
            return None
        try:
            return json.loads(x)

        except:
            return None


def score_json(x : Dict, fields : List[str] = ['title', 'genre', 'characters', 'author','summary', "scenery", "date"]):
    if x is None:
        return 0
    
    
    score = 0
    for key in fields:
        if key == 'characters':
            if not key in x:
                continue

            if type(x['characters']) != list:
                continue

            for character in x['characters']:
                if not "name" in character:
                    break
                if not "description" in character:
                    break

            score += 1

        else:
            if not key in x:
                continue

            if type(x[key]) != str:
                continue

            score += 1

    return score/(max(len(fields), len(x))) 

def generate_augmented_json_df(
        tokenizer : AutoTokenizer, 
        template : str = None,
        instruction : str = None,
        upper : bool = False,
        is_important : bool = False
    ):
    df = load_dataset("NousResearch/json-mode-eval", cache_dir = "/Data")['train']\
    .to_pandas()
    df['instruction'] = df['prompt'].apply(lambda x: x[0]['content'])
    df['text'] = df['prompt'].apply(lambda x : x[1]['content'])

    tqdm.pandas()

    df['json_context_length'] = df['completion'].progress_apply(get_context_length, tokenizer = tokenizer)

    if instruction is None:
        instruction = "You are a helpful assistant that answers in JSON. Here's the json schema you must adhere to:\n\n<schema>\n{schema}\n</schema>"
    if is_important:
        instruction = "IMPORTANT: \n"+ instruction

    if template is None:
        TEMPLATE = "{instruction}\n\n{noise}\n\n{content}\n\nDo not output anything else than the JSON."

    else: 
        TEMPLATE = template


    noise_df = load_dataset(
        "stas/openwebtext-10k", cache_dir="/Data"
    )['train'].to_pandas()

    noise_df['context_length'] = noise_df['text']\
        .progress_apply(get_context_length, tokenizer = tokenizer)

    noises = ['']

    for n in range(7):
        noise = noise_df.query(f"context_length >= 500*{n} & context_length < 500*{n+1}")\
            .sample(1, random_state = 42)\
            .text\
            .item()
        
        noises.append(noise)
    all_df = []

    for noise in noises:
        this_df = deepcopy(df)
        this_df['instruction'] =  this_df.apply(
            lambda row: instruction.upper().format(SCHEMA = row['schema']) if upper else instruction.format(schema = row['schema']),
            axis = 1
        )

        this_df['prompt'] = this_df.apply(
            lambda row: TEMPLATE.format(
                noise = noise,
                content = row['text'],
                instruction = row['instruction']
            ),
            axis = 1
        )


        all_df.append(this_df)

    augmented_df = pd.concat(all_df)\
        .reset_index(drop = True)

    augmented_df['context_length'] = augmented_df['prompt']\
        .progress_apply(get_context_length, tokenizer = tokenizer)
    
    return augmented_df

def split_text_into_chunks(text, token_sizes : List[int], tokenizer):
    # Split the text into sentences
    sentences = text.split('.')
    
    # Remove empty sentences and strip whitespace
    sentences = [sentence.strip() for sentence in sentences if sentence.strip()]

    # Helper function to count tokens
    def count_tokens(text):
        
        return len(tokenizer.encode(text))

    # List to store the chunks
    chunks = {size: '' for size in token_sizes}
    
    for size in (token_sizes):
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = count_tokens(sentence)
            
            # If adding the sentence exceeds the token size, start a new chunk
            if current_length + sentence_length + 10> size:
                chunks[size] = '. '.join(current_chunk) + '.'
                
                break
            else:
                current_chunk.append(sentence)
                current_length += sentence_length

        # Add the last chunk if it's not empty
        # if current_chunk:
        #     chunks[size] = '. '.join(current_chunk) + '.'
    
    return chunks

def read_txt_files(path: str):

    all_text = []
    txt_files = glob.glob(os.path.join(path, '*.txt'))

    # Loop through each file and read its content
    for file_path in txt_files:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            all_text.append(content)

    return pd.Series(all_text)\
        .reset_index()\
        .rename(columns={0:"text"})

def insert_needle(
    df : str, 
    depth_percent : int, 
    question : str,
    needle : str,
    insert_in : str = 'both'
):
    assert insert_in in ['both', 'start', 'end'], "insert_in must be one of ['both', 'start', 'end']"

    text = df['text']

    insertion_point = int(len(text) * depth_percent/100)

    if depth_percent == 100:
        new_text = text + needle
        new_depth_percent = 100
    else:

        while text[insertion_point] != '.' and insertion_point > 0:
            insertion_point -= 1

        new_depth_percent = insertion_point/len(text) * 100
        new_text = text[:insertion_point] + needle + text[(insertion_point+1):]

    question_with_tag = f"\n<question>\n{question}\n</question>\n\n"

    if insert_in == 'both':
        new_text = question_with_tag + new_text + question_with_tag

    elif insert_in == 'start':
        new_text = question_with_tag + new_text

    elif insert_in == 'end':
        new_text = new_text + question_with_tag 

    df['new_text'] = new_text
    df['depth'] = new_depth_percent
    df['question'] = question
    df['needle'] = needle

    return df

def get_context_length(text : str, tokenizer : AutoTokenizer):
    messages = [
        {"role": "user", "content": text}
    ]
    input_ids = tokenizer\
        .apply_chat_template(messages, return_tensors="pt")

    return input_ids.shape[1]


def rolling_mean(tensor, window_size):
    return tensor.unfold(dimension=0, size=window_size, step=window_size).mean(dim=1)

class FileReader:
    @staticmethod
    def get_study_filename(
        study_name : str,
        delta_attention : float,
        use_checkpoint : bool = True,
        all_layers : str = None
    ):
        
        base_path = f"data/{study_name}/"
        path = ""

        if not os.path.exists(base_path):
            dir_files = os.listdir("data")
            raise Exception(f"study_name must be one of {dir_files}")
        
        if use_checkpoint:
            path+= "checkpoints/"

        # if all_layers:
        if all_layers == True:
            path += f"all_layers_"

        if type(all_layers) == str:
            path += all_layers + "_" 

        path += f"generated_delta={delta_attention}.pkl"

        return base_path + path

def is_text_in_language(x, language = "fr"):
    try: 
        return detect(x) == language
    
    except:
        print(f"Exception raised while analysing the text {x}")
        return np.nan

def get_context_length(text : str, tokenizer):
    messages = [
        {"role": "user", "content": text}
    ]
    input_ids = tokenizer\
        .apply_chat_template(messages, return_tensors="pt")

    return input_ids.shape[1]

def get_generated_text(x, start_of_generation_token : str = "[/INST]"):
    return x[0].split(start_of_generation_token)[1]

def rotate_half( x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb( q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        position_ids (`torch.Tensor`, *optional*):
            Deprecated and unused.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

def repeat_kv( hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)