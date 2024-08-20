from torch import Tensor
import torch
from src.influence.base_influence import BaseMetric
from typing_extensions import override

class Influence(BaseMetric):
    def compute_influence(
        self, 
        layer : int,
        use_values : bool = False, 
        p_norm: int = 1,
        **kwargs
    ):
        values = self.attn_saver_model\
            .internal_parameters[layer]\
            ['value']\
            .squeeze()\
            .to("cuda")
                
        if not use_values:
            values = None
        
        attn_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]['avg_attention_heads']\
            .squeeze()\
            .to("cuda")

        embedding : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['raw_embedding']\
            .squeeze()\
            .to("cuda")

        output_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['modified_embedding']\
            .squeeze()\
            .to("cuda")
        

        if layer - 2 in self.dp['influences']:
            if self.dp['influences'][layer-2].device != "cpu":
                self.dp['influences'][layer-2].to("cpu")

        if layer-2 in self.dp['embeddings']:
            self.dp['embeddings'].pop(layer-2)

        last_influence = self.dp["influences"][layer-1].to("cuda")
        last_embedding = self.dp['embeddings'][layer -1 ].to("cuda")

        if values is not None:
            v_norm = values.norm(dim =1, p =1)
            device = v_norm.device
            attn_matrix = attn_matrix.to(device)
            influence_out = (v_norm* attn_matrix) @ (self.dp["influences"][layer-1].to("cuda"))

            influence_out = influence_out/(attn_matrix @ (v_norm))

        else:
            influence_out = attn_matrix @ last_influence

        influence = self.influence_of_sums(
            last_embedding,
            last_influence,
            output_matrix,
            influence_out,
            p_norm,
            **kwargs
        )

        self.dp['influences'][layer]= influence
        self.dp['embeddings'][layer] = embedding

class InfluenceHeads(BaseMetric):
    def influence_heads(
        self,
        layer : int,
        attn_matrix : torch.Tensor,
        values : torch.Tensor = None,
        p_norm : int = 1,
        **kwargs
    ):
        '''
        embedding : n dimensional tensor
        embedding_idx : int
        layer : int
        out : n dimensional tensor
        attn_vector : n dimensional tensor
        instruction_tokens_id : k dimensional tensor
        values : n x 4096 matrix
        '''


        if values is not None:

            v_norm = values.norm(dim =1, p =1)
            device = v_norm.device
            attn_matrix = attn_matrix.to(device)
            influence_heads = (v_norm* attn_matrix) @ (self.dp["influences"][layer-1].to("cuda"))

            influence_heads = influence_heads/(attn_matrix @ (v_norm))

        else:
            influence_heads = attn_matrix @ (self.dp["influences"][layer-1].to("cuda"))

        self.dp['influences_heads'] = influence_heads

    def influence_of_concat(
        self,
        attn_output_per_head : torch.Tensor,
    ):
        """_summary_

        Args:
            attn_output_per_head (torch.Tensor): size (32 x s x 128)

        Returns:
            _type_: _description_
        """        
        influence_heads = self.dp['influences_heads']

        dtype = attn_output_per_head.dtype

        norms = attn_output_per_head.norm(dim = -1)

        influence_heads = influence_heads.to("cuda").to(dtype)
        influence_concat = (norms * influence_heads).sum(dim = 0)/norms.sum(dim = 0)

        return influence_concat
    
    def influence_layer(
        self,
        influence_concat : torch.tensor, 
        concatenated_output : torch.Tensor,
        embedding : torch.Tensor,
        layer : int
    ):  
        if layer - 2 in self.dp['influences']:
            if self.dp['influences'][layer-2].device != "cpu":
                self.dp['influences'][layer-2].to("cpu")

        if layer-2 in self.dp['embeddings']:
            self.dp['embeddings'].pop(layer-2)


        self.dp['embeddings'][layer]= embedding

        last_influence = self.dp['influences'][layer-1].to("cuda")
        last_embedding = self.dp['embeddings'][layer -1 ].to("cuda")
        
        influence = self.influence_of_sums(
            last_embedding,
            last_influence,
            concatenated_output,
            influence_concat,
            1,
        )

        self.dp['influences'][layer] = influence

    def compute_influence(
        self, 
        layer : int, 
        use_values : bool = False,
        p_norm: int = 1, 
        **kwargs
    ):
        values = self.attn_saver_model\
            .internal_parameters[layer]\
            ['value']\
            .squeeze()\
            .to("cuda")
                
        if not use_values:
            values = None
        
        attn_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]['attention']\
            .squeeze()\
            .to("cuda")

        embedding : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['raw_embedding']\
            .squeeze()\
            .to("cuda")

        output_matrix : torch.Tensor = self.attn_saver_model\
            .internal_parameters[layer]\
            ['modified_embedding']\
            .squeeze()\
            .to("cuda")
        
        output_per_head = self.attn_saver_model\
            .internal_parameters\
            [layer]\
            ['output_before_mlp']\
            .squeeze()\
            .to("cuda")
        
        self.influence_heads(
            layer,
            attn_matrix,
            values,
            p_norm =1,
            **kwargs
        )

        influence_concat = self.influence_of_concat(
            output_per_head
        )

        self.influence_layer(
            influence_concat, 
            output_matrix,
            embedding,
            layer
        )

class AttentionRollout(Influence):
    @override
    def influence_of_sums(self, v1: Tensor, I1: float, v2: Tensor, I2: float, p_norm: int = 1):
        return (I1+ I2)/2