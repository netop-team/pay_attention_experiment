import argparse
parser = argparse.ArgumentParser(description='Generates samples of a given instruction on a given dataset.')
parser.add_argument(
    "--delta", 
    help = "The value of delta attention you wish to add",
    required=True
)
parser.add_argument(
    "--num_generations",
    help="Number of generations for each text.",
    required=True
)
parser.add_argument(
    "--model_name",
    help="Model name",
    required=True
)
parser.add_argument(
    "--upper",
    help="int indicating if the instruction should be capitalized",
    required=True
)

# add do-sample argument

args = parser.parse_args()
is_upper = int(args.upper)

import os 
while "notebooks" in os.getcwd():
    os.chdir("..")

import sys
sys.path.insert(0, './')

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, \
  BitsAndBytesConfig, GPTQConfig
import torch
from openai import OpenAI
import json
import pandas as pd
import numpy as np
from pathlib import Path
from src.utils import get_context_length
from transformers import AutoTokenizer
from tqdm import tqdm
from datasets import load_dataset
from huggingface_hub import notebook_login
import seaborn as sns
from IPython.display import clear_output

from src.attention_saver import GUIDEModel
from src.utils import get_hf_model_name, get_start_of_generation_token
tqdm.pandas()

DELTA_ATTENTION = float(args.delta)
n_times_generation = int(args.num_generations)

quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)
model_name = args.model_name
start_of_generation_token = get_start_of_generation_token(model_name)
model_name = get_hf_model_name(model_name)

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir = "/Data"    
)


base_model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config = quantization_config,
    device_map="auto",
    attn_implementation="eager",
    cache_dir = "/Data"
)


model = GUIDEModel(
    base_model,
    tokenizer,
    delta_attention=DELTA_ATTENTION,
    should_save_params= False
)


dataset = load_dataset("TheBritishLibrary/blbooks", "1510_1699", cache_dir = "/Data")['train']\
    .to_pandas()\
    [["record_id", "title", "text", "pg", "all_names", "Language_1"]]

dataset['context_length'] = dataset['text'].progress_apply(get_context_length, tokenizer=tokenizer)
dataset.record_id.nunique()
all_df = []

for book_id, df in dataset.groupby('record_id'):
    df['text'] = df['text'].cumsum()
    df['context_length'] = df['context_length'].cumsum()

    all_df.append(df)
books = pd.concat(all_df)

np.random.seed(42)
book_ids = np.random.choice(books.record_id.unique(), replace= False, size = 300)

mask = books.record_id.isin(book_ids)
selected_books = books[mask].query("context_length <6000 & context_length > 500")


SCHEMA = '''{
    "title": "title of the story (string)", 
    "genre": string, 
    "characters": [{"name": string, "description": string. If not available set it to none} (one dict per character)], 
    "author": "the author of the story. If not available, set it to None", 
    "summary": "a brief summary of the story. Do not write more than 50 words",
    "date": "when the story was released (string)",
    "scenery": "where the story takes place (string)",
}
'''

INSTRUCTION = "Your response should follow exactly this template:"

if is_upper:
    INSTRUCTION = INSTRUCTION.upper()

TEMPLATE = '''
You are an assistant designed to provide information in JSON format. 
I will give you a story, and you need to extract and return specific details from the story. 
Do not output anything else than the JSON.

{instruction}

<schema>
{schema}
</schema>

{content}

'''

if is_upper :
    checkpoint_path = "data/study-04-json/{model_name}/05/checkpoints/delta={delta}_large_upper.parquet"
    base_path = "data/study-04-json/{model_name}/05/delta={delta}_large_upper.parquet"

else:
    checkpoint_path = "data/study-04-json/{model_name}/05/checkpoints/delta={delta}_large.parquet"
    base_path = "data/study-04-json/{model_name}/05/delta={delta}_large.parquet"


generated_text = {}

for epoch in range(n_times_generation):
    decoded = None
    for i, (idx, row) in enumerate(selected_books.sort_values("context_length", ascending=False).iterrows()):
        text = row['text']

        prompt = TEMPLATE.format(content = text, schema = SCHEMA, instruction = INSTRUCTION)
        schema= SCHEMA
        
        message = [{"role": "user", "content": prompt}]
        template = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)

        instruction = INSTRUCTION

        splits = template.split(instruction)
        initial_prompt = splits[0]
        context = splits[1]

        assert (hash(initial_prompt+instruction+context) == hash(template)), "Error in spliting strings. Initial and final string does not match"

        initial_tokens = tokenizer.encode(initial_prompt, return_tensors='pt')
        # open_tag_tokens = tokenizer.encode("<schema>\n", return_tensors='pt')
        # schema_tokens = tokenizer.encode(schema, return_tensors='pt')
        instruction_tokens=  tokenizer.encode(instruction, return_tensors='pt')
        # close_tag_tokens = tokenizer.encode("\n</schema>", return_tensors='pt')
        context_tokens = tokenizer.encode(context, return_tensors='pt')

        start_idx = initial_tokens.size(1)
        end_idx = start_idx + instruction_tokens.size(1) - 1

        model.set_reference_tokens(start_idx, end_idx)
        
        tokens = torch.concat([
            initial_tokens.squeeze(), 
            instruction_tokens.squeeze()[1:],
            # open_tag_tokens.squeeze()[1:],
            # schema_tokens.squeeze()[1:],
            # close_tag_tokens.squeeze()[1:],
            context_tokens.squeeze()[1:]
        ]).unsqueeze(0)

        q = tokenizer.decode(tokens.squeeze()[start_idx: end_idx])

        assert instruction in q, "Error in tokenization. Not giving attention to correct tokens"

        tokens2 = tokenizer(template, return_tensors='pt')

        assert (abs(tokens.shape[1] - tokens2['input_ids'].shape[1]) <=5 ), "Error in tokenization. Tokens do not match"

        clear_output()
        
        if decoded is not None:
            is_json = True    
            json_obj = decoded.split(start_of_generation_token)[1]\
                .replace("\n", '')\
                .replace('</s>', '')\
                .replace('\r', '')\
                .replace('          ', '')\
                .replace("\\", '')
            
            try:
                json_obj   = json.loads(json_obj)

            except:
                is_json = False

            print(
                f'''
                last generated text = 
    {decoded.split(start_of_generation_token)[1]}
                text index = {i}/{len(selected_books)},
                context length = {row['context_length']},
                start index = {start_idx}
                end index = {end_idx}
                is json = {is_json}    
                delta = {DELTA_ATTENTION}
                model_name = {model_name}
                keys = {None if not is_json else json_obj.keys()}
                '''
            )

        with torch.no_grad():
            generated_ids = model.generate(
                tokens, 
                max_new_tokens = 1_000, 
                # suppress_tokens = [endline_token, tab_token, double_backspace_token, backspace_token, triple_backspace_token],
                # forced_decoder_ids = [[0, left_brace_token]],
                max_time = 20,
                do_sample = False,
            )

        decoded = (tokenizer.batch_decode(generated_ids)[0])

        generated_text[idx] = {
            "generated_text": decoded,
            "original_text": text,
            "schema" : schema,
            "context_length" : row['context_length'],
            "book_id": row['record_id']
        }

        if i % 100 == 0 :


            checkpoint_file = checkpoint_path.format(
                delta = DELTA_ATTENTION,
                model_name = model_name
            )
            checkpoint_file = Path(checkpoint_file)
            checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(generated_text).T.to_parquet(checkpoint_file)

outfile = base_path.format(
    delta = DELTA_ATTENTION,
    model_name = model_name
)
outfile = Path(outfile)

outfile.parent.mkdir(parents = True, exist_ok = True)
pd.DataFrame(generated_text).T.to_parquet(outfile)
