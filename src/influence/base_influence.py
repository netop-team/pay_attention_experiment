from abc import ABC, abstractmethod
import torch
from src.attention_saver import Mistral7BAttentionSaver
import pandas as pd
from transformers import AutoModel, AutoTokenizer
from IPython.display import clear_output
from typing import Dict, Union
from tqdm import tqdm
from time import time
import gc

class BaseMetric(ABC):
    def __init__(
        self,
        base_model : AutoModel,
        tokenizer : AutoTokenizer,
        num_layers : int,
    ) -> None:

        self.attn_saver_model = Mistral7BAttentionSaver(
            base_model, 
            tokenizer
        )

        self.tokenizer = tokenizer

        self.num_layers : int = num_layers

        self.tokens = None

        self.reset()

    def reset(self):
        self.dp = {
            "influences":  {  layer : [] for layer in range(-1,self.num_layers)},
            "influences_heads":   {  head : [] for head in range(-1,self.num_layers)},
            "embeddings": {layer : [] for layer in range(-1,self.num_layers)},
            "outputs" :  {layer : [] for layer in range(-1,self.num_layers)}
        }

    def influence_of_sums(
        self,
        v1 : torch.Tensor,
        I1: float,
        v2 : torch.Tensor,
        I2 : float,
        p_norm : int = 1,

    ):
        
        n1 = torch.norm(v1, dim = 1, p = p_norm)\
            .pow(p_norm)
        n2 = torch.norm(v2, dim = 1, p = p_norm)\
            .pow(p_norm)

        return (n1*I1 + n2*I2)/(n1 + n2)

        

    @abstractmethod
    def compute_influence(
        self,
        layer : int,
        use_values : bool = False,
        p_norm : int = 1,
        **kwargs
    ):
        ...

    def __call__(
        self,
        data : Union[pd.DataFrame, pd.Series], 
        delta_attention : float,
        use_values : bool = False,
        instruction_in_text : bool = True,
        text_col : str = "text",
        instruction_col : str = "instruction",
        *args: torch.Any, 
        **kwds: torch.Any
    ):
        self.attn_saver_model.remove_hooks()
        results = dict()
        self.attn_saver_model.set_delta_attention(delta_attention)

        def _compute_influence_series(
            row : pd.Series,
            i : int = 0
        ):
            instruction = row[instruction_col]
            if instruction_in_text:
                text = row[text_col]

            else:
                text = instruction + "\n" + row[text_col]

            text_index = row['index']

            messages = [
                {"role": "user", "content": text},
            ]

            template = self.tokenizer\
                .apply_chat_template(messages, tokenize = False)
            
            # return template
            
            splits = template.split(instruction)
            initial_prompt = splits[0]
            context = instruction.join(splits[1:])

            assert (hash(initial_prompt+instruction+context) == hash(template)), "Error in spliting strings. Initial and final string does not match"

            initial_tokens = self.tokenizer.encode(initial_prompt, return_tensors='pt', add_special_tokens = False)
            instruction_tokens = self.tokenizer.encode(instruction, return_tensors='pt', add_special_tokens = False)
            context_tokens = self.tokenizer.encode(context, return_tensors='pt', add_special_tokens = False)

            start_idx = initial_tokens.size(1)
            end_idx = start_idx + instruction_tokens.size(1)

            
            tokens = torch.concat([
                initial_tokens.squeeze(), 
                instruction_tokens.squeeze(),
                context_tokens.squeeze()
            ]).unsqueeze(0)

            self.tokens = tokens

            q = self.tokenizer.decode(tokens.squeeze()[start_idx: end_idx])

            assert instruction in q, "Error in tokenization. Not giving attention to correct tokens"


            self.attn_saver_model.set_reference_tokens(start_idx, end_idx)
            self.attn_saver_model.insert_hook()

            clear_output()

            print(start_idx, end_idx)
            print(f"Forward propagation on instruction = {instruction}. Index = {i}")
            print(f"Influence tokens : {start_idx} to {end_idx}")
            print(f"Studying influence to '{q}'")

            t0 = time()
            with torch.no_grad():
                self.attn_saver_model(tokens, output_attentions = True)
            t1 = time()

            print(f"Finished forward step in {(t1 - t0)} s")

            token_index_in_text = torch.arange(start_idx, end_idx, step=1)

            # layer -1 is the initial input
            # computing influence before layer 0
            embedding : torch.Tensor = self.attn_saver_model\
                .internal_parameters[0]\
                ['raw_embedding']\
                .squeeze()\
                .to("cuda")

            influence_0 = torch.zeros(len(embedding))
            influence_0[token_index_in_text] = 1

            self.dp["influences"][-1] = torch.tensor(
                influence_0 ,
                dtype = embedding.dtype
            ).to("cpu")

            self.dp['embeddings'][-1] = embedding

            for layer in tqdm(range(0, self.num_layers, 1)):

                self.compute_influence(
                    layer,
                    use_values,
                    p_norm =1,
                    **kwds
                )

                
            self.dp.pop('embeddings')
            self.dp.pop("influences_heads")
            print("Passing tensors to CPU...")

            for layer in range(self.num_layers):
                self.dp['influences'][layer] = self.dp['influences'][layer].to("cpu")

            self.attn_saver_model.remove_hooks()
            self.attn_saver_model.reset_internal_parameters()

            if isinstance(data, pd.DataFrame) and 'depth' in data.columns:
                key = (text_index, row['depth'])

            else:
                key = (text_index, instruction)
                
            results[key] = self.dp
            self.reset()

            gc.collect()
            torch.cuda.empty_cache() 

        if isinstance(data, pd.Series):
            _compute_influence_series(data, data['index'])
        
        elif isinstance(data, pd.DataFrame):

            for i, (idx, row) in enumerate(data.iterrows()):
                _compute_influence_series(row, i )

        return results