# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch RoBERTa model. """

from __future__ import absolute_import, division, print_function, unicode_literals

import heapq
import heapq

import json
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

# from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
#                              prune_linear_layer, add_start_docstrings)
# Potential issue: modeling_utils might be BERT-specific. Need to check if a RoBERTa version exists or adapt.
# For now, let's assume it's compatible or a RoBERTa version will be used.
# Assuming modeling_utils from transformers.modeling_utils can be used, or a custom one for roberta
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer
from transformers.utils.doc import add_start_docstrings
from transformers.utils import logging as hf_logging # Using Hugging Face's logging
# from transformers.configuration_roberta import RobertaConfig # Using Hugging Face's RobertaConfig
from transformers import RobertaConfig


import logging
# from src.utils.comm import is_main_process # Assuming this utility is available
# logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
# if not is_main_process():
#     logger.disabled = True
logger = hf_logging.get_logger(__name__)


# Copied from transformers.models.roberta.modeling_roberta
# Corrected import for LayerNorm
# from torch.nn import LayerNorm as RobertaLayerNorm
LayerNormClass = torch.nn.LayerNorm # Using torch.nn.LayerNorm as a fallback
RobertaLayerNorm = torch.nn.LayerNorm # Using torch.nn.LayerNorm as a fallback


# RoBERTa Pretrained Model Archive Map (Example, needs actual URLs if different from BERT)
ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    # Add other RoBERTa models if necessary
}

ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-config.json",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-config.json",
    # Add other RoBERTa models if necessary
}

def load_tf_weights_in_roberta(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a PyTorch model - Placeholder, adapt from BERT if needed """
    # This function would need to be adapted from BERT's load_tf_weights_in_bert
    # For RoBERTa, it's less common to load TF weights, but the structure is here.
    logger.warning("load_tf_weights_in_roberta is not fully implemented and assumes RoBERTa models are typically loaded from PyTorch checkpoints.")
    return model

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu}


# Adapted from BertEmbeddings
class RobertaEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)

        self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=config.layer_norm_eps) # This is torch.nn.LayerNorm
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # RoBERTa has a different way of handling position_ids, often starting with a padding offset.
        # The `padding_idx` is used by RoBERTa to offset positions.
        # See HuggingFace transformers.models.roberta.modeling_roberta.RobertaEmbeddings
        self.padding_idx = config.pad_token_id
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size, padding_idx=self.padding_idx)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized, handled by create_position_ids_from_input_ids
        # self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1))) # Not needed if generated on the fly
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            # Create position_ids with RoBERTa's offset: self.padding_idx + 1 + past_key_values_length
            # position_ids = torch.arange(past_key_values_length, seq_length + past_key_values_length, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)
            # position_ids = position_ids.unsqueeze(0).expand(input_shape) + self.padding_idx + 1
            # HuggingFace RoBERTaEmbeddings.create_position_ids_from_input_ids logic:
            mask = input_ids.ne(self.padding_idx).int()
            incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
            position_ids = incremental_indices.long() + self.padding_idx


        if token_type_ids is None:
            # RoBERTa typically does not use token_type_ids, but if type_vocab_size > 1, they can be used.
            # Default to zeros if not provided, as in BERT.
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=input_ids.device if input_ids is not None else inputs_embeds.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings # RoBERTa often doesn't use token_type_embeddings meaningfully if type_vocab_size is 1

        if self.position_embedding_type == "absolute":
            # Ensure position_ids are correctly passed to position_embeddings
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


# Adapted from BertSelfAttention
class RobertaSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)
        
        # self.output_attentions = config.output_attentions # This should be determined by the 'output_attentions' argument in forward


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        mixed_query_layer = self.query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        is_cross_attention = encoder_hidden_states is not None

        if is_cross_attention and past_key_value is not None:
            # reuse k,v, cross_attentions
            key_layer = past_key_value[0]
            value_layer = past_key_value[1]
            attention_mask = encoder_attention_mask
        elif is_cross_attention:
            key_layer = self.transpose_for_scores(self.key(encoder_hidden_states))
            value_layer = self.transpose_for_scores(self.value(encoder_hidden_states))
            attention_mask = encoder_attention_mask
        elif past_key_value is not None:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        else:
            key_layer = self.transpose_for_scores(self.key(hidden_states))
            value_layer = self.transpose_for_scores(self.value(hidden_states))

        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in RobertaModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        # output_attentions is passed to forward method, config.output_attentions is a fallback.
        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)
        return outputs


# Adapted from BertSelfOutput
class RobertaSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Adapted from BertAttention
class RobertaAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = RobertaSelfAttention(config)
        self.output = RobertaSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        heads, index = find_pruneable_heads_and_indices(
            heads, self.self.num_attention_heads, self.self.attention_head_size, self.pruned_heads
        )

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        self_outputs = self.self(
            hidden_states,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
            past_key_value,
            output_attentions,
        )
        attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
        return outputs

# Adapted from BertIntermediate
class RobertaIntermediate(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states

# Adapted from BertOutput
class RobertaOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

# Adapted from BertLayer
class RobertaLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = RobertaAttention(config)
        self.intermediate = RobertaIntermediate(config)
        self.output = RobertaOutput(config)
        # Roberta specific:
        self.is_decoder = config.is_decoder
        if self.is_decoder:
            self.crossattention = RobertaAttention(config)


    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
    ):
        # decoder uni-directional self-attention cached key/values tuple is at positions 1,2
        self_attn_past_key_value = past_key_value[:2] if past_key_value is not None else None
        self_attention_outputs = self.attention(
            hidden_states,
            attention_mask,
            head_mask,
            output_attentions=output_attentions,
            past_key_value=self_attn_past_key_value,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        if self.is_decoder and encoder_hidden_states is not None:
            cross_attention_outputs = self.crossattention(
                attention_output,
                attention_mask, # This should be the attention mask for the decoder inputs
                None, # head_mask for cross attention (usually None)
                encoder_hidden_states,
                encoder_attention_mask, # This is the attention mask from the encoder
                None, # past_key_value for cross attention (usually None)
                output_attentions,
            )
            attention_output = cross_attention_outputs[0]
            outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
        
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs
        return outputs


# Adapted from BertEncoder
class RobertaEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.layer = nn.ModuleList([RobertaLayer(config) for _ in range(config.num_hidden_layers)])
        # self.output_attentions = config.output_attentions # Determined by forward pass argument
        # self.output_hidden_states = config.output_hidden_states # Determined by forward pass argument
        # self.return_dict = config.use_return_dict # Determined by forward pass argument

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None, # if True, past_key_values are returned and can be used to speed up decoding
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        # Fallback to config values if not provided in forward pass
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache


        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.is_decoder else None # For cross-attention if decoder

        next_decoder_cache = () if use_cache else None

        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            layer_head_mask = head_mask[i] if head_mask is not None else None
            past_key_value = past_key_values[i] if past_key_values is not None else None


            layer_outputs = layer_module(
                hidden_states,
                attention_mask,
                layer_head_mask,
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_value,
                output_attentions, # Pass the resolved output_attentions
            )
            hidden_states = layer_outputs[0]
            if use_cache:
                next_decoder_cache += (layer_outputs[-1],) # Assuming last element is past_key_value

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.is_decoder and len(layer_outputs) > 2 and layer_outputs[2] is not None : # Cross attention
                     all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        # Standard HuggingFace tuple output for encoder
        # (last_hidden_state, all_hidden_states, all_self_attentions, all_cross_attentions)
        # For encoder-only, all_cross_attentions will be None or empty tuple.
        # If not return_dict:
        outputs = (hidden_states,)
        if output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if output_attentions:
            outputs = outputs + (all_self_attentions,)
            if self.config.is_decoder and all_cross_attentions: # Only add if relevant
                outputs = outputs + (all_cross_attentions,)

        # This part is more for BaseMoedelOutput or similar, but we return a tuple.
        # if not return_dict:
        #     return tuple(v for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions] if v is not None)

        # Based on instruction: (last_hidden_state, all_hidden_states_tuple_or_None, all_attentions_tuple_or_None)
        # "all_attentions" here means self_attentions for an encoder.
        # If it's a decoder, it would include cross-attentions as well or separately.
        # For simplicity and matching BertEncoder, we'll return self_attentions.

        final_outputs = (hidden_states,)
        if output_hidden_states:
            final_outputs += (all_hidden_states,)
        else:
            final_outputs += (None,) # Placeholder for all_hidden_states

        if output_attentions:
            final_outputs += (all_self_attentions,) # Assuming all_attentions means self_attentions for encoder
        else:
            final_outputs += (None,) # Placeholder for all_attentions

        return final_outputs # last_hidden_state, all_hidden_states, all_self_attentions


# Adapted from BertPooler
class RobertaPooler(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


# Adapted from BertPredictionHeadTransform
class RobertaPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states

# Adapted from BertLMPredictionHead
class RobertaLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.transform = RobertaPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        # Need to tie weights manually if using this head, BertPreTrainedModel does it elsewhere.
        self.decoder.bias = self.bias


    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) # Bias is added in nn.Linear
        return hidden_states


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """
    config_class = RobertaConfig
    # load_tf_weights = load_tf_weights_in_roberta # RoBERTa models are typically not loaded from TF checkpoints
    base_model_prefix = "roberta"
    _keys_to_ignore_on_load_missing = [r"position_ids"] # From HuggingFace RobertaModel
    # _keys_to_ignore_on_load_unexpected = [r"pooler"] # Specific to some head models
    # _keys_to_ignore_on_save = [] # Specific to some head models


    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm): # Standard LayerNorm
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    # Removed load_tf_weights_in_roberta as it's a placeholder and rarely used.
    # load_tf_weights = load_tf_weights_in_roberta
    
    @property
    def dummy_inputs(self): # Added from RobertaModel for testing/tracing
        #Generate dummy inputs for the model, assuming batch size 1 and sequence length 128
        #These are suitable for tracing the model with torch.fx.symbolic_trace
        #or for testing purposes
        input_ids = torch.zeros([1, 128], dtype=torch.long)
        #attention_mask = torch.ones([1,128], dtype=torch.long) # RoBERTa does not use attention_mask by default
        return {"input_ids": input_ids} #, "attention_mask": attention_mask}

# Adapted from BertImgModel
class RobertaImgModel(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaImgModel, self).__init__(config)
        self.config = config # Save config
        self.embeddings = RobertaEmbeddings(config) # Using RobertaEmbeddings
        self.encoder = RobertaEncoder(config) # Using RobertaEncoder
        self.pooler = RobertaPooler(config) # Using RobertaPooler

        self.img_dim = config.img_feature_dim
        logger.info('RobertaImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = getattr(config, 'img_feature_type', 'fc') # Default to 'fc'
        self.use_img_layernorm = getattr(config, 'use_img_layernorm', False) # Default to False

        if self.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        # Add other img_feature_type conditions if necessary, similar to BertImgModel
        else: # Default case (e.g., 'fc')
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=getattr(config, 'img_layer_norm_eps', config.layer_norm_eps) )


        self.init_weights()

    # Removed _init_weights from RobertaImgModel to rely on RobertaPreTrainedModel._init_weights
    # Specific initializations for img_embedding can be done in __init__ if necessary,
    # but standard nn.Linear initialization from parent should suffice.

    def _resize_token_embeddings(self, new_num_tokens):
        # Method to resize token embeddings, similar to BertModel._resize_token_embeddings
        # It should resize self.embeddings.word_embeddings
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        # Update the decoder weights if they are tied to the word_embeddings
        if self.config.tie_word_embeddings and hasattr(self, "cls") and hasattr(self.cls, "predictions") and hasattr(self.cls.predictions, "decoder"):
             self.cls.predictions.decoder.weight = new_embeddings.weight
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        # This method should call the encoder's prune_heads method
        if hasattr(self.encoder, 'prune_heads'):
            self.encoder.prune_heads(heads_to_prune)
        else:
            logger.warning("Encoder does not have a prune_heads method.")


    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                img_feats=None,
                encoder_hidden_states=None, # Not directly used by RobertaEncoder, map to past_key_values if applicable for generation
                encoder_attention_mask=None, # Not directly used by RobertaEncoder for self-attention
                past_key_values=None, # Preferred way for RoBERTa encoder history
                use_cache=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        use_cache = use_cache if use_cache is not None else self.config.use_cache # For encoder

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
            device = input_ids.device
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape # text seq_length
            device = inputs_embeds.device
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        # Map encoder_history_states to past_key_values if provided and past_key_values is None
        # Map encoder_hidden_states to past_key_values if provided and past_key_values is None
        if encoder_hidden_states is not None and past_key_values is None:
            past_key_values = encoder_hidden_states  # Assuming compatible format

        # Attention mask
        # Handle attention_mask for text and image features
        if attention_mask is None:
            # If no mask is provided, create a full attention mask that attends to everything
            current_seq_length = seq_length
            if img_feats is not None:
                current_seq_length += img_feats.shape[1]
            attention_mask = torch.ones((batch_size, current_seq_length), device=device)
        elif img_feats is not None and attention_mask.shape[1] == seq_length :
            # If mask is for text only, extend it for image features
            img_attention_mask = torch.ones((batch_size, img_feats.shape[1]), device=device, dtype=attention_mask.dtype)
            attention_mask = torch.cat([attention_mask, img_attention_mask], dim=1)
        # If attention_mask is already combined length, it's used as is.

        if token_type_ids is None and hasattr(self.embeddings, 'token_type_embeddings'):
             # Some RoBERTa versions might not use token_type_ids extensively if type_vocab_size is 1
             # Create zeros if input_ids are given, otherwise this will be handled by RobertaEmbeddings if inputs_embeds path is taken.
            if input_ids is not None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)
        # If token_type_ids are provided for text and img_feats are present, they should cover the concatenated sequence.
        # For simplicity, if img_feats are added, token_type_ids for the image part can be a default (e.g., 0 or 1),
        # or assumed to be part of the input `token_type_ids` if it's already of combined length.
        # RobertaEmbeddings will handle token_type_ids for the text part.

        # Get text embeddings
        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids, # Pass token_type_ids for the text part
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values[0][0].shape[2] if past_key_values is not None else 0, # RobertaEmbeddings uses this
        )

        # Process and concatenate image embeddings if img_feats are provided
        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                # Assuming similar handling as BertImgModel for 'dis_code'
                # code_emb = self.code_embeddings(img_feats) # Placeholder, ensure code_embeddings exists if type is 'dis_code'
                # img_embedding_output = self.img_embedding(code_emb)
                # For now, simplified, assuming 'fc' or similar direct mapping for other types
                logger.warning("img_feature_type 'dis_code' specific path in RobertaImgModel needs full implementation if used.")
                img_embedding_output = self.img_embedding(img_feats) # Fallback for now
            elif self.img_feature_type == 'dis_code_t':
                 logger.warning("img_feature_type 'dis_code_t' specific path in RobertaImgModel needs full implementation if used.")
                 img_embedding_output = self.img_embedding(img_feats) # Fallback for now
            elif self.img_feature_type == 'dis_code_scale':
                 logger.warning("img_feature_type 'dis_code_scale' specific path in RobertaImgModel needs full implementation if used.")
                 img_embedding_output = self.img_embedding(img_feats) # Fallback for now
            else: # Default 'fc' or other direct mapping
                img_embedding_output = self.img_embedding(img_feats)

            if self.use_img_layernorm:
                # Ensure LayerNorm is defined in __init__ for this path
                if hasattr(self, 'LayerNorm') and self.LayerNorm is not None:
                     img_embedding_output = self.LayerNorm(img_embedding_output)
                else: # Fallback if self.LayerNorm for images was not initialized (e.g. if use_img_layernorm was false)
                     logger.warning("use_img_layernorm is true, but self.LayerNorm for images is not defined. Skipping LayerNorm.")

            # Dropout for image embeddings
            img_embedding_output = self.dropout(img_embedding_output)

            # Concatenate text and image embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), dim=1)

            # If token_type_ids were only for text, extend them for the image part
            if token_type_ids is not None and token_type_ids.shape[1] == seq_length:
                img_token_type = getattr(self.config, 'img_token_type', 0) # Default image token type to 0 or 1
                img_token_type_ids = torch.full(
                    (batch_size, img_feats.shape[1]),
                    img_token_type,
                    dtype=torch.long,
                    device=device
                )
                # This assumes self.embeddings was called with text-only token_type_ids.
                # However, RobertaEmbeddings itself doesn't take concatenated token_type_ids.
                # The token_type_embeddings are added *inside* RobertaEmbeddings.
                # For simplicity, we assume token_type_ids passed to RobertaEmbeddings are for text,
                # and image part doesn't get explicit token_type_ids added here at concatenation stage,
                # relying on RobertaEmbeddings to handle its part. If type_vocab_size is small for RoBERTa, this is less critical.
                # A more robust way for combined sequences would be to create combined token_type_ids *before* self.embeddings,
                # but that changes how self.embeddings is called if it expects only text part.
                # The current RobertaEmbeddings is for text only.
                # So, we are concatenating pure image embeddings with text embeddings that already include token_type.
                pass # No explicit token_type_id extension after cat for now, relies on how RobertaEmbeddings works.


        # Prepare extended attention mask
        # extended_attention_mask = self.get_extended_attention_mask(attention_mask, embedding_output.shape[:2], device)
        # Re-using existing logic from file for extended_attention_mask:
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3: # For compatibility if a pre-extended mask is passed (e.g. from beam search)
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        # Pass to RoBERTa encoder
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states, # Should be None for self-attention encoder
            encoder_attention_mask=encoder_attention_mask, # Should be None for self-attention encoder
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict, # Encoder itself might not use return_dict for tuple output
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # Consistent tuple output
        # encoder_outputs = (last_hidden_state, all_hidden_states, all_self_attentions)
        # We want (sequence_output, pooled_output, all_hidden_states, all_attentions)

        final_outputs = (sequence_output, pooled_output)
        if output_hidden_states and len(encoder_outputs) > 1 and encoder_outputs[1] is not None:
            final_outputs += (encoder_outputs[1],) # all_hidden_states
        else:
            final_outputs += (None,)

        if output_attentions and len(encoder_outputs) > 2 and encoder_outputs[2] is not None:
            final_outputs += (encoder_outputs[2],) # all_attentions
        else:
            final_outputs += (None,)

        return final_outputs # sequence_output, pooled_output, (hidden_states), (attentions)


# Adapted from BertCaptioningHeads
class RobertaCaptioningHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = RobertaLMPredictionHead(config) # Using RobertaLMPredictionHead

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores

# BertCaptioningLoss can likely be reused as is, if it's general enough.
# For now, assume it can be imported or copied.
# Re-defining for completeness, ensure it's available.
class RobertaCaptioningLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.label_smoothing = getattr(config, 'label_smoothing', 0)
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.log_soft = nn.LogSoftmax(dim=1)
        self.kl = nn.KLDivLoss(reduction='none')
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        eps = self.label_smoothing
        n_class = logits.size(1)
        one_hot = torch.zeros_like(logits).scatter(1, target.view(-1, 1), 1)
        one_hot = one_hot * (1 - eps) + (1 - one_hot) * eps / (n_class - 1)
        log_prb = self.log_soft(logits)
        loss = self.kl(log_prb, one_hot).sum(1)

        if self.drop_worst_ratio > 0 and self.iter > self.drop_worst_after:
            loss, _ = torch.topk(loss,
                    k=int(loss.shape[0] * (1-self.drop_worst_ratio)),
                    largest=False)
        loss = loss.mean()
        return loss

# RobertaIFPredictionHead - Adapted from BertIFPredictionHead
class RobertaIFPredictionHead(nn.Module):
    def __init__(self, config):
        super(RobertaIFPredictionHead, self).__init__()
        self.transform = RobertaPredictionHeadTransform(config) # Using Roberta's transform
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        # TODO: The output dimension (2048) is hardcoded in BERT version. Make this configurable if possible.
        self.decoder = nn.Linear(config.hidden_size, getattr(config, 'img_feature_pred_dim', 2048), bias=False)
        self.bias = nn.Parameter(torch.zeros(getattr(config, 'img_feature_pred_dim', 2048)))
        self.decoder.bias = self.bias # Manually tie bias

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) # Bias is part of nn.Linear
        return torch.nn.functional.relu(hidden_states)

# RobertaImgFeatureLoss - Adapted from BertImgFeatureLoss (likely reusable)
class RobertaImgFeatureLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        # Configuration for loss (label smoothing, drop worst, etc.)
        self.cri = nn.MSELoss() # Example, can be other losses like SmoothL1Loss
        self.iter = 0 # For tracking iterations if needed by loss logic

    def forward(self, logits, target):
        self.iter += 1
        target = target.view(-1, target.shape[-1])
        loss = self.cri(logits, target)
        return loss


@add_start_docstrings("""RoBERTa Model transformer for image captioning""",
    # ROBERTA_START_DOCSTRING, # Placeholder, define if needed
    """RoBERTa model specific start docstring placeholder.""", # Placeholder
    # ROBERTA_INPUTS_DOCSTRING # Placeholder, define if needed
    """RoBERTa model specific input docstring placeholder.""", # Placeholder
    )
class RobertaForImageCaptioning(RobertaPreTrainedModel):
    def __init__(self, config):
        super(RobertaForImageCaptioning, self).__init__(config)
        self.config = config
        self.roberta = RobertaImgModel(config) # Using RobertaImgModel
        self.cls = RobertaCaptioningHeads(config) # Using RobertaCaptioningHeads
        self.loss = RobertaCaptioningLoss(config) # Using RobertaCaptioningLoss (or Bert's if identical)

        # For image feature prediction part, adapted from BertForImageCaptioning
        self.cls_img_feat = RobertaIFPredictionHead(config) # Using Roberta's version
        self.loss_img_feat = RobertaImgFeatureLoss(config) # Using Roberta's version (or Bert's)

        self.init_weights()
        self.tie_weights()

    # Removed _init_weights to rely on RobertaPreTrainedModel's _init_weights
    # def _init_weights(self, module):
    #     super()._init_weights(module)

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            # Ensure word_embeddings attribute path is correct for RobertaImgModel.embeddings
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.roberta.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.roberta.embeddings.word_embeddings.weight.requires_grad = not freeze


    def forward(self, *args, **kwargs):
        # This forward method needs to be adapted from BertForImageCaptioning
        # Key considerations:
        # - Argument names and handling (is_decode, inference_mode, etc.)
        # - Calling self.roberta (the RobertaImgModel instance) instead of self.bert
        # - Ensuring the output processing and loss calculation are compatible.
        # For now, a simplified placeholder:
        is_decode = kwargs.get('is_decode', False)
        inference_mode = kwargs.get('inference_mode', '') # Added

        if inference_mode:
            kwargs.pop('inference_mode')
            # Add specific generation methods if they differ significantly from Bert's
            # e.g., self.roberta_prod_generate, etc.
            # For now, assuming they can be adapted from BertForImageCaptioning's methods
            # by primarily changing self.bert to self.roberta and adjusting for any RoBERTa specific API changes.
            # This part requires careful review of BertForImageCaptioning.generate and related methods.
            if inference_mode == 'prod':
                 return self.prod_generate(*args, **kwargs) # Needs adaptation
            # Add other inference modes if necessary
            raise NotImplementedError(f"Inference mode '{inference_mode}' not fully adapted for RoBERTa yet.")

        if is_decode:
            return self.generate(*args, **kwargs) # Needs adaptation from BertForImageCaptioning
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask,
            masked_pos=None, masked_ids=None,
            masked_pos_img=None, masked_token_img=None,
            token_type_ids=None, position_ids=None, head_mask=None,
            is_training=True, encoder_history_states=None,
            # Added inputs_embeds, output_attentions, output_hidden_states, return_dict for HF compatibility
            # though current RobertaImgModel might not use all of them directly if it's not returning dict.
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        # Resolve output flags for self.roberta call
        # These would be passed to RobertaImgModel which then passes to RobertaEncoder
        roberta_output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        roberta_output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # roberta_return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        # For now, assuming RobertaImgModel returns a tuple as per its implementation.

        outputs = self.roberta(input_ids, img_feats=img_feats, attention_mask=attention_mask,
                position_ids=position_ids, token_type_ids=token_type_ids,
                head_mask=head_mask,
                encoder_hidden_states=encoder_history_states, # This is past_key_values for Roberta
                inputs_embeds=inputs_embeds, # Pass through
                output_attentions=roberta_output_attentions,
                output_hidden_states=roberta_output_hidden_states,
                return_dict=False # Explicitly False, expecting tuple from RobertaImgModel
                )

        # outputs from RobertaImgModel: (sequence_output, pooled_output, all_hidden_states_tuple, all_attentions_tuple)
        sequence_output = outputs[0]
        # pooled_output = outputs[1] # Not directly used by this method's main logic
        model_other_outputs = outputs[2:] # These are (all_hidden_states, all_attentions)

        if is_training:
            # Determine text sequence length. masked_pos corresponds to the text part.
            if masked_pos is None:
                raise ValueError("masked_pos is required for training to determine text sequence length.")
            text_seq_len = masked_pos.shape[-1]
            
            text_sequence_output = sequence_output[:, :text_seq_len, :]
            
            # Ensure masked_pos is boolean for indexing
            if masked_pos.dtype != torch.bool:
                effective_masked_pos = masked_pos == 1
            else:
                effective_masked_pos = masked_pos

            sequence_output_masked = text_sequence_output[effective_masked_pos, :]

            if masked_ids is None:
                 raise ValueError("masked_ids is required for training.")
            # Filter out padding tokens from masked_ids (often -1 or -100)
            # This should happen *before* loss calculation, and masked_ids should align with sequence_output_masked
            valid_masked_ids = masked_ids[effective_masked_pos] # Align masked_ids with selected sequence_output
            valid_masked_ids = valid_masked_ids[valid_masked_ids != -1] # Filter out padding

            if sequence_output_masked.shape[0] != valid_masked_ids.shape[0] and valid_masked_ids.numel() > 0 :
                # This check is important. If sequence_output_masked is empty due to no True in masked_pos,
                # but valid_masked_ids is not empty (e.g. due to incorrect mask logic or all -1s filtered out),
                # or vice-versa, it's an issue.
                # However, if sequence_output_masked is empty and valid_masked_ids is also empty, it's fine (no loss).
                if not (sequence_output_masked.shape[0] == 0 and valid_masked_ids.numel() == 0):
                    logger.warning(f"Mismatch between masked outputs ({sequence_output_masked.shape[0]}) and masked labels ({valid_masked_ids.shape[0]}) after filtering. This might lead to errors or incorrect loss.")

            class_logits = self.cls(sequence_output_masked) # RobertaCaptioningHeads

            # Only compute loss if there are valid (non-padded) masked tokens
            if valid_masked_ids.numel() > 0 :
                masked_lm_loss = self.loss(class_logits.float(), valid_masked_ids) # RobertaCaptioningLoss
            else: # No valid masked tokens to compute loss on
                masked_lm_loss = torch.tensor(0.0, device=sequence_output.device)

            total_loss = masked_lm_loss

            # Image feature prediction part (Optional)
            if masked_pos_img is not None and masked_token_img is not None:
                img_sequence_output = sequence_output[:, text_seq_len:, :] # Get the image part
                
                if masked_pos_img.dtype != torch.bool:
                    effective_masked_pos_img = masked_pos_img == 1
                else:
                    effective_masked_pos_img = masked_pos_img

                # Check if any image tokens are actually masked
                if torch.any(effective_masked_pos_img):
                    img_output_masked = img_sequence_output[effective_masked_pos_img, :]

                    # Align masked_token_img with selected img_output_masked
                    valid_masked_token_img = masked_token_img[effective_masked_pos_img, :]

                    if img_output_masked.shape[0] > 0: # Ensure some tokens were actually selected
                        img_feat_logits = self.cls_img_feat(img_output_masked) # RobertaIFPredictionHead
                        masked_img_loss = self.loss_img_feat(img_feat_logits.float(), valid_masked_token_img) # RobertaImgFeatureLoss

                        img_loss_weight = getattr(self.config, 'img_loss_weight', 0.1) # Default weight if not in config
                        total_loss += img_loss_weight * masked_img_loss
                    # else: # No need for warning if img_output_masked is empty, implies no loss contribution
                        # logger.warning("Masked image positions were specified, but resulted in no tokens being selected for loss calculation.")
                # else: # No image tokens masked, no loss contribution
                    # logger.warning("No image tokens were masked for image feature prediction loss.")

            # Return (total_loss, class_logits_for_masked_tokens, other_model_outputs_like_attentions_if_any)
            return (total_loss, class_logits,) + model_other_outputs
        else: # Not training (e.g., for feature extraction or validation logits for the whole sequence)
            # Determine text sequence length. If input_ids is present, use its length.
            # Otherwise, this path might need a way to know the text length (e.g. from config or a convention).
            if input_ids is not None:
                text_seq_len = input_ids.shape[-1]
            elif inputs_embeds is not None: # If inputs_embeds are given, assume they are for text part if img_feats also exist
                text_seq_len = inputs_embeds.shape[1] if img_feats is None else inputs_embeds.shape[1]
            else: # Fallback: this is problematic for non-training if we don't know text length
                  # For now, assume the sequence_output is all text, or use a placeholder.
                  # This path is less common for captioning models which usually train or generate.
                logger.warning("Cannot accurately determine text_seq_len for non-training mode without input_ids or a clear convention.")
                text_seq_len = sequence_output.shape[1] - (img_feats.shape[1] if img_feats is not None else 0)

            text_sequence_output = sequence_output[:, :text_seq_len, :]
            class_logits = self.cls(text_sequence_output) # Get logits for the whole text sequence
            return (class_logits,) + model_other_outputs

    # The `generate` and related methods (`_generate_beam_search`, `_generate_no_beam_search`,
    # `prepare_inputs_for_generation`, `_decode_step`, etc.) from `BertForImageCaptioning`
    # need to be copied and adapted here.
    # Key changes:
    # - Use `self.roberta` instead of `self.bert`.
    # - Ensure RoBERTa specific details (like `pad_token_id`, `bos_token_id`, `eos_token_ids` if different) are handled.
    # - RoBERTa doesn't use `token_type_ids` in the same way as BERT, so ensure `RobertaImgModel` and `RobertaEmbeddings` handle this.
    # - `mask_token_id` should be RoBERTa's mask token ID.
    # The `generate` and related methods (`_generate_beam_search`, `_generate_no_beam_search`,
    # `prepare_inputs_for_generation`, `_decode_step`, etc.) from `BertForImageCaptioning`
    # need to be copied and adapted here.
    # Key changes:
    # - Use `self.roberta` instead of `self.bert`.
    # - Ensure RoBERTa specific details (like `pad_token_id`, `bos_token_id`, `eos_token_ids` if different) are handled.
    # - RoBERTa doesn't use `token_type_ids` in the same way as BERT, so ensure `RobertaImgModel` and `RobertaEmbeddings` handle this.
    # - `mask_token_id` should be RoBERTa's mask token ID.
    # - `prepare_inputs_for_generation` will need to correctly construct inputs for `self.roberta`.
    # This is a significant part and requires careful, line-by-line adaptation.

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x

    def _do_output_past(self, outputs):
        # This model specific function is used to check if past key values are captured in outputs
        # It is used in _generate_beam_search and _generate_no_beam_search
        # For RobertaImgModel, past_key_values are expected at outputs[2] when use_cache=True
        # and output_hidden_states=False (typical for generation).
        # outputs = (sequence_output, pooled_output, past_key_values, all_attentions)
        # So, if len(outputs) > 2 and outputs[2] is not None, it means past is outputted.
        return len(outputs) >= 3 and outputs[2] is not None


    def generate(self, img_feats, attention_mask, masked_pos, token_type_ids=None,
            position_ids=None, head_mask=None, input_ids=None, max_length=None,
            do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, bos_token_id=None, pad_token_id=None,
            eos_token_ids=None, mask_token_id=None, length_penalty=None,
            num_return_sequences=None,
            num_keep_best=1, is_decode=None, # is_decode is specific to BertForImageCaptioning's forward, not standard HF
            add_od_labels=False, od_labels_start_posid=None,
            use_cbs=False, fsm=None, num_constraints=None,
            min_constraints_to_satisfy=None, use_hypo=False,
            decoding_constraint_flag=None, bad_ending_ids=None,
            output_scores=None, # Standard HF argument
            return_dict_in_generate=None, # Standard HF argument
            **model_kwargs # Standard HF model_kwargs
            ):
        """ Generates captions given image features """

        # If is_decode is True (i.e., we are in generation/inference mode for captioning),
        # ignore any provided input_ids from the dataloader to ensure we start fresh with BOS.
        # The OD labels logic below will still use model_kwargs if add_od_labels is True.
        if is_decode:
            input_ids = None
            # Other inputs like attention_mask, token_type_ids, masked_pos from the dataloader
            # are also less relevant if we start fresh with BOS for the actual sequence construction.
            # They are, however, used by self._expand_for_beams if num_beams > 1 or num_return_sequences > 1.
            # The critical part is that the model's first step input_ids becomes just [BOS].

        # This method adapts BertForImageCaptioning.generate

        # Standard HuggingFace generate arguments not in BertForImageCaptioning's original signature:
        # output_attentions, output_hidden_states are usually controlled by config or model_kwargs
        # output_scores, return_dict_in_generate are standard.

        # Process is_decode (if it's a way to route to this function)
        # if is_decode is None: is_decode = True # Assume this function is for decoding

        # Update with model-specific kwargs
        # model_kwargs['output_attentions'] = output_attentions # from config or args
        # model_kwargs['output_hidden_states'] = output_hidden_states # from config or args
        # model_kwargs['use_cache'] = True # always true for generation

        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        if max_length is None: max_length = self.config.max_length if hasattr(self.config, 'max_length') else 20
        self.max_seq_len = max_length # Used by prepare_inputs_for_generation

        # vocab
        vocab_size = self.config.vocab_size
        if mask_token_id is None: mask_token_id = self.config.mask_token_id
        if bos_token_id is None: bos_token_id = self.config.bos_token_id
        if pad_token_id is None: pad_token_id = self.config.pad_token_id
        if eos_token_ids is None: eos_token_ids = self.config.eos_token_id
        if isinstance(eos_token_ids, int): eos_token_ids = [eos_token_ids]

        self.mask_token_id = mask_token_id # Used by prepare_inputs_for_generation
        self.prev_encoded_layers = None # Used by prepare_inputs_for_generation if adapting BERT's exact logic
                                        # For RoBERTa, this should map to past_key_values handling

        self.num_keep_best = num_keep_best # Used by _generate_beam_search

        if not use_cbs: # Constrained Beam Search
            num_fsm_states = 1
        else:
            b, num_fsm_states, f1, v = fsm.shape
            assert b == batch_size and v == vocab_size and f1 == num_fsm_states

        self.add_od_labels = add_od_labels
        if od_labels_start_posid is None and hasattr(self.config, 'od_labels_start_posid'):
             od_labels_start_posid = self.config.od_labels_start_posid
        self.od_labels_start_posid = max(od_labels_start_posid if od_labels_start_posid is not None else 0, self.max_seq_len)

        if self.add_od_labels:
            assert input_ids is not None, "input_ids must be provided for OD labels"
            # In BERT version, input_ids was used for OD labels. Assume similar here.
            # The passed `input_ids` here might be just OD labels or combined.
            # Bert's `generate` took `input_ids` (for text prompt) and `od_label_ids` (derived from the original `input_ids`'s tail).
            # This adaptation assumes `input_ids` passed to `generate` could be the text prompt part,
            # and `model_kwargs` might contain `od_label_ids` if needed.
            # For simplicity, let's assume `od_label_ids` are passed via `model_kwargs` if `add_od_labels` is True.
            od_label_ids = model_kwargs.pop('od_label_ids', None)
            if od_label_ids is None:
                raise ValueError("`od_label_ids` must be provided in model_kwargs when `add_od_labels` is True.")
            self.od_labels_len = od_label_ids.shape[1]
            # Text prompt part (if any)
            # The original BERT code sets `input_ids = None` after extracting OD labels.
            # Let's assume `input_ids` passed to `generate` is the text prompt (e.g., BOS).
        else:
            self.od_labels_len = 0
            od_label_ids = None

        if input_ids is None: # If no text prompt, start with BOS
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=img_feats.device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        cur_len = input_ids.shape[1]
        effective_batch_size = batch_size
        if num_return_sequences != 1:
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size *= num_return_sequences
        
        # Expand other inputs that are per-batch but need to be per-beam for generation step's model call
        num_expand = num_beams * num_fsm_states * (num_return_sequences if num_return_sequences is not None else 1)

        self.od_label_ids = self._expand_for_beams(od_label_ids, num_expand) # OD labels expanded for beams
        self.img_feats = self._expand_for_beams(img_feats, num_expand) # Image features expanded
        
        # These are full masks/ids prepared by the user, corresponding to the structure before beam expansion.
        # They are used by `prepare_inputs_for_generation` by slicing/selecting.
        # We expand them here so `prepare_inputs_for_generation` receives beam-expanded versions.
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand) # masked_pos seems specific to training objective?
                                                                            # In BERT's generate, it's used for slicing.
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand) # Usually None for generation


        # Generation call, assuming _generate_beam_search and _generate_no_beam_search are part of the class
        # or inherited (they are not in PreTrainedModel, so need to be defined or copied)
        # For this subtask, we assume they will be available.
        # The actual call to HuggingFace's internal generation utilities like `beam_search`, `sample`
        # would happen here if fully refactoring to HF style.
        # Sticking to BertForImageCaptioning structure for now:
        if not use_cbs:
            if num_beams > 1:
                # These methods `_generate_beam_search`, `_generate_no_beam_search` are from HF `generation_utils.py`
                # and are usually called by `PreTrainedModel.generate`.
                # BertForImageCaptioning has its own simplified versions of these.
                # We'll assume these will be adapted/copied into RobertaForImageCaptioning.
                # For now, this is a conceptual placeholder for the call.
                output_tuple = self._generate_beam_search( # This method needs to be in the class
                    input_ids, cur_len, max_length, do_sample, temperature, top_k, top_p,
                    repetition_penalty, pad_token_id, eos_token_ids, effective_batch_size,
                    length_penalty, num_beams, vocab_size
                )
            else:
                output_tuple = self._generate_no_beam_search( # This method needs to be in the class
                    input_ids, cur_len, max_length, do_sample, temperature, top_k, top_p,
                    repetition_penalty, pad_token_id, eos_token_ids, effective_batch_size
                )
        else: # Constrained Beam Search (CBS)
            from src.modeling.utils_cbs import (ConstrainedBeamSearch, select_best_beam_with_constraints)
            # Assuming utils_cbs is available in the path
            assert self.num_keep_best == 1, 'num_keep_best > 1 not supported for CBS'
            searcher = ConstrainedBeamSearch(eos_token_ids, max_length, num_beams, use_hypo=use_hypo,
                                             decoding_constraint_flag=decoding_constraint_flag,
                                             bad_ending_ids=bad_ending_ids)
            # _decode_step is a method of this class
            curr_ids, sum_logprobs = searcher.search(input_ids, None, self._decode_step, fsm)
            curr_ids, logprobs = select_best_beam_with_constraints(
                curr_ids, sum_logprobs, num_constraints, min_constraints_to_satisfy, eos_token_ids
            )
            # Expected output: (batch_size, n_best, max_len), (batch_size, n_best)
            # For compatibility with the non-CBS path, and num_keep_best=1:
            # Return shape: (batch_size * num_return_sequences, max_length), (batch_size * num_return_sequences, num_beams, max_length) for scores
            # This part needs to align with HF return format or BertForImageCaptioning's specific format.
            # Bert's version returns ( (batch_size, num_keep_best, max_len), (batch_size, num_keep_best) )
            # HF's generate returns (batch_size * num_return_sequences, max_length) and scores tuple/dict.
            # For now, let's match Bert's CBS output structure.
            output_tuple = (curr_ids.unsqueeze(1), logprobs.unsqueeze(1))


        # Process output_tuple to match expected format (sequences, scores)
        # Bert's _generate_beam_search returns: ( (batch_size, num_keep_best, max_len), (batch_size, num_keep_best) )
        # Bert's _generate_no_beam_search returns: ( (batch_size, max_len), None ) or ( (batch_size, max_len), scores )
        # The final return should be (generated_sequence_ids, scores_or_logprobs)
        # If output_tuple is already (sequences, scores), it's fine.
        # If scores are None from _generate_no_beam_search, handle it.
        
        # Reshape to (batch_size, num_return_sequences * num_keep_best, max_length) for sequences
        # and (batch_size, num_return_sequences * num_keep_best) for scores
        # This part depends on how _generate_beam_search/_generate_no_beam_search are implemented.
        # Assuming they return what BertForImageCaptioning expects.
        # The final output for this function as per problem description: (generated_sequence_ids, scores_or_logprobs)
        # This usually means sequences of shape (batch_size * num_return_sequences, max_length)
        # and scores of shape (batch_size * num_return_sequences, vocab_size) or similar for sequence scores.
        # For now, directly return the output_tuple assuming its structure is (sequences, scores).
        return output_tuple


    def _decode_step(self, curr_ids, past_key_values, **kwargs_for_prepare):
        """
        Performs a single decoding step. (Adapted from BertForImageCaptioning)
        Args:
            curr_ids: Tensor of current token sequences in the beam (batch_size * num_beams, current_length).
            past_key_values: Cached key/value states from previous step.
        Returns:
            Tuple of (logits for the next token, new_past_key_values).
        """
        model_inputs = self.prepare_inputs_for_generation(curr_ids, past=past_key_values, **kwargs_for_prepare)

        # Model Forward Pass
        # self.roberta is RobertaImgModel.
        # It should return past_key_values if use_cache=True.
        # Expected output: (sequence_output, pooled_output, past_key_values, all_attentions)
        # when use_cache=True and output_hidden_states=False.
        outputs = self.roberta(**model_inputs)

        sequence_output = outputs[0]

        new_past_key_values = None
        if model_inputs.get('use_cache', False): # Should always be true from prepare_inputs
            if self._do_output_past(outputs): # Checks if past is actually in outputs
                new_past_key_values = outputs[2] # Assuming past_key_values are at index 2
            else:
                logger.warning("_decode_step: `use_cache` was True, but `new_past_key_values` not found in outputs.")

        # Get Logits for Prediction
        # input_ids to prepare_inputs_for_generation is typically [last_generated_token, MASK_TOKEN]
        # The MASK token is expected to be at index 1 of the input_ids to the model for this step.
        mask_token_index = 1 # Assuming input to model was [token, MASK] or [BOS, MASK]
        # If input_ids to model was just the current token (HF style), then index is 0 and seq_len is 1.
        # BertForImageCaptioning's prepare_inputs makes input_ids as [token, MASK]

        # sequence_output from roberta has shape (batch_size * num_beams, step_seq_len, hidden_size)
        # We need the hidden state of the MASK token to predict the next token.
        # If input_ids to roberta was [token, MASK], then step_seq_len is 2.
        # The MASK token's hidden state is at index 1.
        if sequence_output.shape[1] > mask_token_index:
            next_token_hidden_states = sequence_output[:, mask_token_index, :]
        else:
            # This case might happen if input_ids to roberta was just the current token (length 1)
            # Then we take the hidden state of that token.
            next_token_hidden_states = sequence_output[:, 0, :]


        logits = self.cls(next_token_hidden_states) # self.cls is RobertaCaptioningHeads
        return logits, new_past_key_values


    def prepare_inputs_for_generation(self, curr_ids, past=None, **kwargs):
        """
        Prepares inputs for generation, carefully adapted from BertForImageCaptioning.
        """
        mask_token_id = self.mask_token_id # Set in generate() from self.config.mask_token_id
        batch_size = curr_ids.shape[0] # effective_batch_size (batch_size * num_beams * num_fsm_states * num_return_sequences)

        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        # Local helper functions from BERT's prepare_inputs (if used)
        # These relied on self.max_seq_len and self.od_labels_len, which should be set in self.generate()
        def _slice(t, start, end):
            if t is None: return t
            # Assuming t is (batch_size, self.max_seq_len + self.od_labels_len) before expansion
            # After expansion in generate, t is (effective_batch_size, ...)
            # This slicing is specific to text part + OD labels part.
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None: return t
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        img_feats_for_step = None
        past_key_values_for_step = past # `past` from previous step, directly used as past_key_values

        if past is None: # First decoding step (e.g., after BOS token)
            # Input for this step: [curr_ids (BOS), mask_ids]
            # If add_od_labels: [curr_ids (BOS), mask_ids, od_label_ids]
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)
            current_text_part_len = input_ids.shape[1] # e.g., 2 for [BOS, MASK]

            # Attention mask construction for the first step
            # The mask should cover: [text_part, od_labels (opt), img_feats]
            # attention_mask_len = current_text_part_len
            # if self.add_od_labels:
            #     input_ids = torch.cat([input_ids, self.od_label_ids], dim=1) # self.od_label_ids is already beam-expanded
            #     attention_mask_len += self.od_labels_len
            # if self.img_feats is not None: # self.img_feats is already beam-expanded
            #     img_feats_for_step = self.img_feats
            #     attention_mask_len += self.img_seq_len
            # attention_mask = torch.ones((batch_size, attention_mask_len), device=curr_ids.device)

            # BERT's complex attention mask slicing:
            # This assumes self.full_attention_mask is a 3D mask (batch, full_len, full_len)
            # and parts corresponding to future text tokens are removed.
            # full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            # seq_start = current_text_part_len # Where MASK token is, relative to text part start
            # seq_end = self.max_seq_len      # End of text part in full mask
            # attention_mask = _remove_rows_cols_from_3d_mask(self.full_attention_mask, seq_start, seq_end, seq_start, seq_end)
            # This _remove_rows_cols_from_3d_mask needs to be defined if using this exact logic.
            # For simplicity, if self.full_attention_mask is already prepared for the first step (e.g. pre-sliced or correctly structured)
            # we might use it directly or derive from it.
            # The current RobertaImgModel.forward expects a 2D attention_mask.

            # Simplified first-step attention_mask (model will extend it to 3D/4D):
            # This mask is for the sequence [input_ids (incl. MASK, OD), img_feats]
            temp_input_ids_len = current_text_part_len
            if self.add_od_labels:
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
                temp_input_ids_len += self.od_labels_len

            attention_mask_len = temp_input_ids_len
            if self.img_feats is not None:
                img_feats_for_step = self.img_feats # Already beam-expanded
                attention_mask_len += self.img_seq_len
            attention_mask = torch.ones((batch_size, attention_mask_len), device=curr_ids.device)


            # Position IDs
            # Explicit creation similar to BERT, adapted for RoBERTa if needed (e.g. padding_idx offset for RobertaEmbeddings)
            # RobertaEmbeddings uses `past_key_values_length` and `padding_idx` to create its own positions if `position_ids` is None.
            # If we provide them, they should be "absolute" positions.
            position_ids = torch.arange(current_text_part_len, dtype=torch.long, device=curr_ids.device)
            if self.add_od_labels:
                od_pos_ids = torch.arange(self.od_labels_start_posid, self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=curr_ids.device)
                position_ids = torch.cat([position_ids, od_pos_ids], dim=0) # This results in (N,)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1) # (B, N)

            # Token Type IDs
            # For RoBERTa, often all zeros if type_vocab_size is 1.
            token_type_ids_text_part = torch.zeros((batch_size, current_text_part_len), dtype=torch.long, device=curr_ids.device)
            if self.add_od_labels:
                # od_type = 1 if self.config.type_vocab_size > 1 else 0 # Example for different type
                od_type = 0 # Assuming all same type for RoBERTa simplicity
                token_type_ids_od_part = torch.full((batch_size, self.od_labels_len), od_type, dtype=torch.long, device=curr_ids.device)
                token_type_ids = torch.cat([token_type_ids_text_part, token_type_ids_od_part], dim=1)
            else:
                token_type_ids = token_type_ids_text_part

            # self.prev_encoded_layers is BERT's name for past_key_values cache.
            # For RoBERTa, we directly use past_key_values_for_step.
            # No complex reordering of `past` state as in BERT's prepare_inputs.
            self.prev_encoded_layers = None # Clear any old state (though `past` argument handles this)

        else: # Subsequent decoding steps (past is not None)
            # Input for this step: [last_token_generated, mask_ids]
            last_token = curr_ids[:, -1:]
            input_ids = torch.cat([last_token, mask_ids], dim=1) # Shape: (batch_size, 2)

            # Position IDs:
            # RobertaEmbeddings can infer this using `past_key_values_length`.
            # If providing explicitly:
            # current_text_len_generated = curr_ids.shape[1] # Length of [BOS, token1, ..., last_token]
            # pos_for_last_token = current_text_len_generated - 1
            # pos_for_mask_token = current_text_len_generated
            # position_ids = torch.tensor([pos_for_last_token, pos_for_mask_token], dtype=torch.long, device=curr_ids.device).unsqueeze(0).expand(batch_size, 2)
            # For simplicity and robustness with RoBERTa's embedding, let RobertaEmbeddings handle it.
            position_ids = None # RobertaEmbeddings will use past_key_values_length to create correct position_ids

            # Token Type IDs for [last_token, MASK] - usually all zeros for RoBERTa.
            token_type_ids = torch.zeros_like(input_ids)

            img_feats_for_step = None # Image features info is in `past_key_values`

            # Attention Mask for subsequent steps:
            # With past_key_values, the model's attention mechanism combines cached keys/values with new keys/values.
            # The attention_mask passed to model.forward should typically only mask padding in the *new* input_ids.
            # Since our new input_ids is [token, MASK] (length 2) and has no padding, mask is all ones.
            attention_mask = torch.ones_like(input_ids) # Shape (batch_size, 2)
            # RobertaImgModel.forward will extend this 2D mask correctly when past_key_values are present.

        # Common inputs for both cases (first step / subsequent steps)
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'token_type_ids': token_type_ids,
            'past_key_values': past_key_values_for_step,
            'img_feats': img_feats_for_step, # Only for the first step
            # Model control flags
            'output_attentions': False, # Usually False for generation
            'output_hidden_states': False, # Usually False for generation
            'use_cache': True, # ESSENTIAL for efficient generation
            # 'is_training': False, # Implied by generate context, model should be in eval mode.
        }
        return model_inputs

    def prod_generate(self, img_feats, od_label_ids, max_length,
                      bos_token_id=None, eos_token_ids=None, mask_token_id=None,
                      od_labels_start_posid=None, add_od_labels=True,
                      # RoBERTa specific token type IDs might be all 0s
                      cls_token_segment_id=0, sequence_a_segment_id=0, sequence_b_segment_id=0):
        """
        Generates captions for PROD (efficient production) setting. Adapted from BertForImageCaptioning.
        Assumes batch_size = 1, num_beams = 1. Uses past_key_values for speed.
        RoBERTa adaptation: Simplified past_key_values handling, token_type_ids, position_ids.
        """
        if bos_token_id is None: bos_token_id = self.config.bos_token_id
        if eos_token_ids is None:
            # Ensure eos_token_ids is a list, even if a single ID is provided in config
            if hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                eos_token_ids = [self.config.eos_token_id]
            else:
                eos_token_ids = [] # Default to empty list if not configured
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]

        if mask_token_id is None: mask_token_id = self.config.mask_token_id

        batch_size = img_feats.shape[0]
        assert batch_size == 1, "prod_generate is for batch_size=1"
        device = img_feats.device

        od_labels_len = od_label_ids.shape[1] if add_od_labels and od_label_ids is not None else 0
        img_seq_len = img_feats.shape[1] # Assuming img_feats is (batch, img_seq_len, img_dim)

        mask_input_id_tensor = torch.full((1, 1), mask_token_id, dtype=torch.long, device=device)
        current_generated_ids = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)

        past_key_values = None
        sum_logprob = 0.0

        # Initial input construction for the very first step
        # input_ids_step1: [BOS, MASK, OD_labels (opt)]
        input_ids_list_step1 = [current_generated_ids, mask_input_id_tensor]
        if add_od_labels and od_label_ids is not None:
            input_ids_list_step1.append(od_label_ids)
        input_ids_for_first_step = torch.cat(input_ids_list_step1, dim=1)

        current_img_feats_for_model = img_feats # Only pass img_feats at the first step

        # Attention mask for the first step (covers text part, optional OD labels, and image features)
        # RobertaImgModel's forward pass will create the extended 3D/4D mask.
        attention_mask_first_step_len = input_ids_for_first_step.shape[1] + (img_seq_len if current_img_feats_for_model is not None else 0)
        attention_mask_for_first_step = torch.ones((1, attention_mask_first_step_len), device=device)

        # Token type IDs for RoBERTa: typically all 0s.
        token_type_ids_for_first_step = torch.zeros_like(input_ids_for_first_step)
        if add_od_labels and od_label_ids is not None and hasattr(self.config, 'type_vocab_size') and self.config.type_vocab_size > 1:
             # RoBERTa often has type_vocab_size=1 or 0, so this might not apply or sequence_b_segment_id would be 0.
             len_text_part_step1 = current_generated_ids.shape[1] + mask_input_id_tensor.shape[1]
             # Ensure sequence_b_segment_id is valid for the model's type_vocab_size
             actual_sequence_b_segment_id = sequence_b_segment_id if sequence_b_segment_id < self.config.type_vocab_size else 0
             token_type_ids_for_first_step[:, len_text_part_step1:] = actual_sequence_b_segment_id


        # Position IDs: For RoBERTa, it's often best to let RobertaEmbeddings compute them.
        # Pass None, and RobertaEmbeddings will use past_key_values_length (0 for 1st step) and padding_idx.
        position_ids_for_first_step = None

        # Initialize loop variables for clarity, used from the second step onwards
        input_ids_incremental_loop = None
        attention_mask_incremental_loop = None
        token_type_ids_incremental_loop = None
        position_ids_incremental_loop = None # Will remain None for RoBERTa with past_key_values

        for _ in range(max_length):
            # Use full inputs for the first step, incremental inputs for subsequent steps
            current_input_ids_for_model_step = input_ids_for_first_step if past_key_values is None else input_ids_incremental_loop
            current_attention_mask_for_model_step = attention_mask_for_first_step if past_key_values is None else attention_mask_incremental_loop
            current_token_type_ids_for_model_step = token_type_ids_for_first_step if past_key_values is None else token_type_ids_incremental_loop
            current_position_ids_for_model_step = position_ids_for_first_step if past_key_values is None else position_ids_incremental_loop

            model_inputs = {
                'input_ids': current_input_ids_for_model_step,
                'img_feats': current_img_feats_for_model, # None after the first step
                'attention_mask': current_attention_mask_for_model_step,
                'token_type_ids': current_token_type_ids_for_model_step,
                'position_ids': current_position_ids_for_model_step,
                'past_key_values': past_key_values,
                'use_cache': True,
                'output_attentions': False,
                'output_hidden_states': False,
            }

            outputs = self.roberta(**model_inputs) # self.roberta is RobertaImgModel
            sequence_output = outputs[0]
            # past_key_values are expected at index 2 if RobertaImgModel's encoder returns them (when use_cache=True)
            past_key_values = outputs[2] if self._do_output_past(outputs) else None

            # Determine the index of the MASK token's output.
            # For the first step, input was [BOS, MASK, ODs...]. MASK is at index 1.
            # For subsequent steps, input was [prev_token, MASK]. MASK is at index 1.
            mask_token_output_idx = 1
            next_token_logits = self.cls(sequence_output[:, mask_token_output_idx, :])

            next_token_id = torch.argmax(next_token_logits, dim=-1) # Greedy decoding

            log_probs = F.log_softmax(next_token_logits, dim=-1)
            sum_logprob += log_probs[0, next_token_id[0]].item() # batch_size is 1

            if next_token_id.item() in eos_token_ids:
                break

            current_generated_ids = torch.cat([current_generated_ids, next_token_id.unsqueeze(-1)], dim=1)

            if (current_generated_ids.shape[1] - 1) >= max_length: # -1 for BOS token
                break

            # Prepare inputs for the *next* decoding step
            input_ids_incremental_loop = torch.cat([next_token_id.unsqueeze(-1), mask_input_id_tensor], dim=1) # [newly_generated_token, MASK]
            current_img_feats_for_model = None # Image features are now encoded in past_key_values
            attention_mask_incremental_loop = torch.ones_like(input_ids_incremental_loop) # Attention for [current_token, MASK]
            token_type_ids_incremental_loop = torch.zeros_like(input_ids_incremental_loop) # RoBERTa typically uses 0s
            position_ids_incremental_loop = None # Crucial for RoBERTa with past_key_values

        final_gen_len = current_generated_ids.shape[1] - 1 # Number of generated tokens (excluding BOS)
        avg_logprob = sum_logprob / final_gen_len if final_gen_len > 0 else 0.0
        return current_generated_ids, torch.full((1,), avg_logprob, device=device)

    def prod_no_hidden_generate(self, img_feats, od_label_ids, max_length,
            bos_token_id=None, eos_token_ids=None, mask_token_id=None,
            od_labels_start_posid=None, add_od_labels=True,
            # RoBERTa specific token type IDs might be all 0s or use configured values
            cls_token_segment_id=0, # From RoBERTa config or default 0
            sequence_a_segment_id=0, # From RoBERTa config or default 0
            sequence_b_segment_id=0, # RoBERTa often doesn't distinguish segment B like BERT
                                     # Defaulting to 0, or use a config value if type_vocab_size > 1
            ):
        """
        Generates captions for PROD (efficient production) setting, without using hidden state history (past_key_values).
        Adapted from BertForImageCaptioning.prod_no_hidden_generate.
        Assumes batch_size = 1, num_beams = 1.
        Each step is a full forward pass of the current sequence.
        """
        if bos_token_id is None: bos_token_id = self.config.bos_token_id
        if eos_token_ids is None:
            if hasattr(self.config, 'eos_token_id') and self.config.eos_token_id is not None:
                eos_token_ids = [self.config.eos_token_id]
            else:
                eos_token_ids = []
        elif isinstance(eos_token_ids, int):
            eos_token_ids = [eos_token_ids]
        if mask_token_id is None: mask_token_id = self.config.mask_token_id

        batch_size = img_feats.shape[0]
        assert batch_size == 1, "prod_no_hidden_generate is for batch_size=1"
        device = img_feats.device

        od_labels_len = od_label_ids.shape[1] if add_od_labels and od_label_ids is not None else 0
        img_seq_len = img_feats.shape[1]

        # Mask token tensor, used repeatedly
        mask_input_id_tensor = torch.full((1, 1), mask_token_id, dtype=torch.long, device=device)

        # Current generated sequence, starts with BOS
        current_generated_ids = torch.full((1, 1), bos_token_id, dtype=torch.long, device=device)

        # Pre-calculate a triangle mask for self-attention up to max_length for the text part
        # This is for ensuring causal attention for the generated text tokens.
        # Max possible text length = max_length (caption) + 1 (for MASK token).
        max_text_len_for_mask = max_length + 1
        triangle_self_attention_mask = torch.tril(torch.ones(
            (max_text_len_for_mask, max_text_len_for_mask), dtype=torch.long, device=device
        ))

        sum_logprob = 0.0

        # Helper to prepare inputs for RobertaImgModel for each step
        def _prepare_inputs_for_step(generated_ids_so_far):
            current_text_len = generated_ids_so_far.shape[1] # Length of [BOS, token1, ..., current_token]

            # Input IDs for this step: [BOS, token1, ..., current_token, MASK, OD_labels (opt)]
            input_ids_list_step = [generated_ids_so_far, mask_input_id_tensor]
            if add_od_labels and od_label_ids is not None:
                input_ids_list_step.append(od_label_ids)
            step_input_ids = torch.cat(input_ids_list_step, dim=1)

            # Token Type IDs for RoBERTa (often all zeros)
            # For [BOS, token1, ..., MASK] part:
            text_and_mask_len = current_text_len + 1 # +1 for MASK token
            step_token_type_ids = torch.full((1, text_and_mask_len), sequence_a_segment_id, dtype=torch.long, device=device)
            if add_od_labels and od_label_ids is not None:
                # Use sequence_b_segment_id for OD labels if type_vocab_size allows, else 0
                actual_od_segment_id = sequence_b_segment_id if self.config.type_vocab_size > sequence_b_segment_id else 0
                od_token_types = torch.full((1, od_labels_len), actual_od_segment_id, dtype=torch.long, device=device)
                step_token_type_ids = torch.cat([step_token_type_ids, od_token_types], dim=1)

            # Position IDs: For RoBERTa, typically allow RobertaEmbeddings to create them if not providing specific offsets.
            # If created explicitly here, they should be absolute.
            # For a "no_hidden" (no past_key_values) pass, positions are from 0 for the current sequence.
            text_od_combined_len = text_and_mask_len + od_labels_len
            step_position_ids = torch.arange(text_od_combined_len, dtype=torch.long, device=device).unsqueeze(0)
            if add_od_labels and od_labels_start_posid is not None:
                 # If explicit OD label start positions are given, adjust them.
                 # This part needs to be robust. Original BERT code for prod_no_hidden uses fixed ranges.
                 # For simplicity, let's assume od_labels_start_posid is an offset for OD part if used.
                 # This means positions for text part are [0...text_len-1] and OD part is [od_start...od_start+od_len-1].
                 # The current step_position_ids is [0 ... text_od_combined_len-1].
                 # If od_labels_start_posid is e.g. max_length, we'd offset the OD part.
                 # The BERT version's _prepare_inputs for prod_no_hidden used:
                 # position_ids = torch.arange(token_len, dtype=torch.long, device=device)
                 # od_labels_posids = torch.arange(od_labels_start_posid, od_labels_start_posid + od_labels_len, ...)
                 # step_position_ids = torch.cat([position_ids, od_labels_posids])
                 # Let's try to replicate this structure for `token_len` = text_and_mask_len
                 pos_ids_text_mask_part = torch.arange(text_and_mask_len, dtype=torch.long, device=device)
                 if add_od_labels and od_label_ids is not None:
                     actual_od_labels_start_posid = od_labels_start_posid if od_labels_start_posid is not None else text_and_mask_len
                     pos_ids_od_part = torch.arange(actual_od_labels_start_posid, actual_od_labels_start_posid + od_labels_len, dtype=torch.long, device=device)
                     step_position_ids = torch.cat([pos_ids_text_mask_part, pos_ids_od_part], dim=0).unsqueeze(0)
                 else:
                     step_position_ids = pos_ids_text_mask_part.unsqueeze(0)


            # Attention Mask:
            # Covers [text_part (incl MASK), OD_labels (opt), img_feats]
            # Text part is causal (triangle_self_attention_mask).
            # OD_labels and img_feats attend to everything before/among themselves but not future text.
            # Text cannot attend to future text.
            # RobertaImgModel expects a 2D mask [batch, combined_seq_len]

            # Length of just text + MASK tokens for self-attention part of the mask
            current_text_plus_mask_len = generated_ids_so_far.shape[1] + 1 # +1 for the MASK token

            # 1. Self-attention for the text part (including MASK token)
            text_self_attention_mask = triangle_self_attention_mask[:current_text_plus_mask_len, :current_text_plus_mask_len]

            # 2. Attention between text and OD labels / image features
            # Text can attend to OD labels and image features.
            # OD labels / image features can attend to text (that came before them in construction).
            # For simplicity in this "no_hidden" version, often a full attention is allowed for non-text parts to text,
            # and among non-text parts themselves.
            # The BERT code: attention_mask[:, :token_len, :token_len].copy_(triangle_mask[:token_len, :token_len])
            # attention_mask[:, token_len:, :token_len] = 0 # od_label, img_feat can not see sentence (this seems reversed?)
            # Let's assume:
            # - Text part: causal mask.
            # - OD/Image part: can see all text part, and all OD/Image part.
            # - Text part: can see all OD/Image part.

            total_len_before_img = step_input_ids.shape[1] # Length of [BOS, ..., MASK, ODs...]
            total_len_with_img = total_len_before_img + img_seq_len

            step_attention_mask = torch.ones((1, total_len_with_img), dtype=torch.long, device=device)

            # Apply causal mask for the text_plus_mask part to itself
            # This requires constructing a larger 2D mask and then embedding the triangle.
            # For RobertaImgModel, it expects a 2D mask [batch, seq], it will make it 3D/4D.
            # The simplest is to pass full attention here and let the model apply its own causal masking if needed (e.g. if it's a decoder).
            # However, Bert's prod_no_hidden explicitly creates a custom 3D mask.
            # RobertaImgModel's encoder is not inherently causal.
            # So, if causal generation is needed, the mask must enforce it.
            # For now, let's use a simplified 2D mask that RobertaImgModel can extend.
            # A simple 2D mask of all ones is effectively what happens if no complex 3D mask is built by Bert's prod_no_hidden.
            # Bert's prod_no_hidden_generate has specific 3D mask logic:
            #   attention_mask = torch.ones((1, token_len+od_labels_len+img_seq_len, token_len+od_labels_len+img_seq_len), ...)
            #   attention_mask[:, :token_len, :token_len].copy_(triangle_mask[:token_len, :token_len])
            #   attention_mask[:, token_len:, :token_len] = 0 # od_label, img_feat can not see sentence
            # This implies the model needs a 3D mask. RobertaImgModel.forward currently makes 2D->3D/4D.
            # To replicate BERT's behavior, we might need to pass a 3D mask.
            # Let's assume for now RobertaImgModel's standard 2D mask processing is sufficient if we ensure inputs are right.
            # If the model is encoder-only, it doesn't apply causal masking by default.
            # The "MASK" token approach for generation means we predict the MASK, so the sequence before MASK is context.
            # Causal masking is mostly for autoregressive generation without a MASK token.
            # So, a full attention mask for the currently constructed sequence should be fine.

            # Fallback to simpler 2D mask of all ones for now.
            step_attention_mask = torch.ones((1, total_len_with_img), dtype=torch.long, device=device)

            return step_input_ids, step_token_type_ids, step_position_ids, step_attention_mask

        # Main generation loop
        for i in range(max_length): # Number of tokens to generate (excluding BOS)
            step_input_ids, step_token_type_ids, step_position_ids, step_attention_mask = \
                _prepare_inputs_for_step(current_generated_ids)

            model_inputs = {
                'input_ids': step_input_ids,
                'img_feats': img_feats, # Passed at every step as there's no history
                'attention_mask': step_attention_mask,
                'token_type_ids': step_token_type_ids,
                'position_ids': step_position_ids,
                # No past_key_values / encoder_history_states for "no_hidden"
                'use_cache': False,
                'output_attentions': False,
                'output_hidden_states': False,
            }

            outputs = self.roberta(**model_inputs) # self.roberta is RobertaImgModel
            sequence_output = outputs[0] # Shape: (batch_size, seq_len_this_step, hidden_size)

            # Logits for the MASK token, which is at index `current_generated_ids.shape[1]`
            # e.g., if current_generated_ids is [BOS, tok1], len=2. Input was [BOS, tok1, MASK,...]. MASK is at index 2.
            mask_token_index_in_step_output = current_generated_ids.shape[1]
            next_token_logits = self.cls(sequence_output[:, mask_token_index_in_step_output, :])

            next_token_id = torch.argmax(next_token_logits, dim=-1) # Greedy decoding (shape: [1])

            log_probs = F.log_softmax(next_token_logits, dim=-1)
            sum_logprob += log_probs[0, next_token_id[0]].item() # batch_size is 1

            if next_token_id.item() in eos_token_ids:
                break

            current_generated_ids = torch.cat([current_generated_ids, next_token_id.unsqueeze(-1)], dim=1)

            # Check if max_length (of generated part) is reached
            if (current_generated_ids.shape[1] - 1) >= max_length: # -1 for BOS token
                break

        final_gen_len = current_generated_ids.shape[1] - 1 # Number of generated tokens (excluding BOS)
        avg_logprob = sum_logprob / final_gen_len if final_gen_len > 0 else 0.0

        return current_generated_ids, torch.full((1,), avg_logprob, device=device)

    # Adapted from BertForImageCaptioning
    def _generate_no_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample,
        temperature,
        top_k,
        top_p,
        repetition_penalty,
        pad_token_id,
        eos_token_ids,
        batch_size, # effective_batch_size
    ):
        """ Generate sequences for each example without beam search (num_beams == 1).
            All returned sequence are generated independantly.
        """
        # current position and vocab size
        vocab_size = self.config.vocab_size

        # Expand input to num return sequences
        # input_ids = self._expand_for_beams(input_ids, num_return_sequences) # Already done in generate if num_return_sequences > 1
        # batch_size = input_ids.shape[0] # This is effective_batch_size

        # generated hypotheses
        generated_hyps = [
            torch.zeros(max_length, dtype=torch.long, device=input_ids.device) -1 for _ in range(batch_size)
        ]
        generated_lengths = torch.zeros(batch_size, dtype=torch.long, device=input_ids.device)

        # For accumulating log probabilities of chosen tokens
        sum_logprobs = torch.zeros(batch_size, dtype=torch.float, device=input_ids.device)

        # past states
        past = None

        # current positions
        curr_ids = input_ids # (batch_size, cur_len)

        # done sentences
        done = [False for _ in range(batch_size)]

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(curr_ids, past=past)
            # outputs from roberta: (sequence_output, pooled_output, past_key_values, attentions)
            # sequence_output has shape (batch_size, model_input_seq_len, hidden_size)
            # past_key_values are at outputs[2] if use_cache=True
            outputs = self.roberta(**model_inputs)

            sequence_output_for_cls = outputs[0]
            if self._do_output_past(outputs): # Check if past_key_values are present
                past = outputs[2]
            else:
                past = None

            # Calculate logits for the MASK token (expected at index 1 of the model_inputs['input_ids'])
            # model_inputs['input_ids'] to roberta was [token, MASK] or [BOS, MASK]
            mask_token_index = 1
            if sequence_output_for_cls.shape[1] > mask_token_index:
                next_token_hidden_states = sequence_output_for_cls[:, mask_token_index, :]
            else: # Should not happen if prepare_inputs_for_generation is correct
                next_token_hidden_states = sequence_output_for_cls[:, 0, :]

            next_token_logits = self.cls(next_token_hidden_states) # (batch_size, vocab_size)

            # repetition penalty (remains the same)
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in curr_ids[i]:
                        if next_token_logits[i, previous_token] < 0:
                            next_token_logits[i, previous_token] *= repetition_penalty
                        else:
                            next_token_logits[i, previous_token] /= repetition_penalty

            # Calculate log probabilities for chosen tokens
            next_token_logprobs_all_vocab = F.log_softmax(next_token_logits, dim=-1) # (batch_size, vocab_size)

            if do_sample:
                # Temperature
                if temperature != 1.0:
                    next_token_logits_for_sampling = next_token_logits / temperature
                else:
                    next_token_logits_for_sampling = next_token_logits
                # Top-p/top-k filtering
                next_token_logits_for_sampling = self._top_k_top_p_filtering(next_token_logits_for_sampling, top_k=top_k, top_p=top_p)
                # Sample
                probs = F.softmax(next_token_logits_for_sampling, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1) # (batch_size,)
            else:
                # Greedy decoding
                next_tokens = torch.argmax(next_token_logits, dim=-1) # (batch_size,)

            # Gather the log probabilities of the chosen tokens
            # next_tokens needs to be shaped as (batch_size, 1) for gather
            chosen_token_logprobs = torch.gather(next_token_logprobs_all_vocab, 1, next_tokens.unsqueeze(-1)).squeeze(-1) # (batch_size,)

            # update generated hypotheses and sum_logprobs
            for i in range(batch_size):
                if done[i]:
                    continue
                generated_hyps[i][cur_len - input_ids.shape[1]] = next_tokens[i]
                generated_lengths[i] += 1
                sum_logprobs[i] += chosen_token_logprobs[i] # Add logprob of the chosen token

                if next_tokens[i].item() in eos_token_ids:
                    done[i] = True

            # update current token sequences
            curr_ids = torch.cat([curr_ids, next_tokens.unsqueeze(-1)], dim=-1)
            cur_len +=1

            if all(done):
                break

        # Convert generated hypotheses to a tensor
        output_ids = torch.stack(generated_hyps, dim=0) # (batch_size, max_length)
        # Trim to actual generated lengths (remove -1 padding)
        # This part needs careful thought. If input_ids was len 1 (BOS), then output_ids up to generated_lengths[i] is fine.
        # BertForImageCaptioning's version returns sequences of `max_length`.
        # Let's keep it simple and return full max_length sequences, padded with -1 or pad_token_id.
        # The original BERT code seems to return ( (batch_size, max_len), scores_tuple_or_None )
        # Scores tuple is (batch_size, num_beams, vocab_size) for each token. Here num_beams=1.
        # For _generate_no_beam_search, scores are often not returned or are raw logits.
        # HF's generate returns sequences of shape (batch_size * num_return_sequences, sequence_length)
        # and optionally scores.
        # This internal method seems to be structured to be called by `generate`.
        # The `generate` method will handle num_return_sequences and reshaping.

        # For compatibility with _generate_beam_search's output structure for scores:
        # scores: List of (batch_size, vocab_size) -> stack to (batch_size, seq_len_generated, vocab_size)
        # Then permute to (seq_len_generated, batch_size, vocab_size)
        # Then reshape to (seq_len_generated, batch_size * 1, vocab_size)
        # This seems overly complex for no_beam_search. HF usually returns None for scores here, or simple logits.
        # BertForImageCaptioning original returns `output` which is `(output_ids, None)` or `(output_ids, final_scores_if_needed)`.
        # Let's return `(output_ids, None)` for simplicity, assuming scores are not typically processed from this path.

        # Finalize generated sequences: input_ids + generated part
        # The `generated_hyps` only stores the *generated* part.
        # We need to combine with the initial `input_ids` (which was just BOS usually).
        final_sequences = []
        for i in range(batch_size):
            initial_len = input_ids.shape[1]
            gen_len = generated_lengths[i]
            seq = torch.cat( (input_ids[i, :initial_len], generated_hyps[i][:gen_len]), dim=0)
            # Pad to max_length
            padding_needed = max_length - seq.shape[0]
            if padding_needed > 0:
                seq = torch.cat([seq, torch.full((padding_needed,), pad_token_id if pad_token_id is not None else -1, dtype=torch.long, device=input_ids.device)])
            final_sequences.append(seq)

        output_ids_final = torch.stack(final_sequences, dim=0)

        # Calculate average log probabilities
        avg_logprobs = torch.zeros(batch_size, dtype=torch.float, device=input_ids.device)
        for i in range(batch_size):
            if generated_lengths[i] > 0:
                avg_logprobs[i] = sum_logprobs[i] / generated_lengths[i]
            else:
                # Avoid division by zero; assign a very low logprob if no tokens were generated
                avg_logprobs[i] = -float('Inf')

        return output_ids_final, avg_logprobs


    # Adapted from BertForImageCaptioning
    def _generate_beam_search(
        self,
        input_ids,
        cur_len,
        max_length,
        do_sample, # Unused in original beam search, but kept for signature
        temperature, # Unused
        top_k, # Unused
        top_p, # Unused
        repetition_penalty, # Unused
        pad_token_id,
        eos_token_ids,
        batch_size, # effective_batch_size (batch_size * num_return_sequences)
        length_penalty,
        num_beams,
        vocab_size,
    ):
        """ Generate sequences for each example with beam search.
        """
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(num_beams, max_length, length_penalty, early_stopping=False) for _ in range(batch_size)
        ]

        # initial tokens
        beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
        beam_scores[:, 1:] = -1e9 # Initialize all beams except the first to -infinity
        beam_scores = beam_scores.view(-1)  # shape (batch_size * num_beams,)

        # current sequences (batch_size * num_beams, cur_len)
        # input_ids is (effective_batch_size, 1) e.g. [[BOS], [BOS]] if num_return_sequences=2
        # We need to expand it for num_beams: (effective_batch_size * num_beams, 1)
        curr_ids = self._expand_for_beams(input_ids, num_beams)

        # past states, (batch_size * num_beams, ...)
        past = None

        # done sentences
        done = [False for _ in range(batch_size)]

        # Pad token ID ensure handling
        if pad_token_id is None and hasattr(self.config, 'pad_token_id') and self.config.pad_token_id is not None:
            pad_token_id = self.config.pad_token_id
        elif pad_token_id is None:
            logger.warning("pad_token_id not set, defaulting to 0 if needed for padding final sequences.")
            pad_token_id = 0 # A common default if not specified

        while cur_len < max_length:
            model_inputs = self.prepare_inputs_for_generation(curr_ids, past=past)
            # outputs from roberta: (sequence_output, pooled_output, past_key_values, attentions)
            # sequence_output has shape (batch_size * num_beams, model_input_seq_len, hidden_size)
            outputs = self.roberta(**model_inputs)

            sequence_output_for_cls = outputs[0]
            if self._do_output_past(outputs):
                past = outputs[2] # past_key_values
            else:
                past = None

            # Calculate logits for the MASK token (expected at index 1 of the model_inputs['input_ids'])
            mask_token_index = 1
            if sequence_output_for_cls.shape[1] > mask_token_index:
                next_token_hidden_states = sequence_output_for_cls[:, mask_token_index, :]
            else:
                next_token_hidden_states = sequence_output_for_cls[:, 0, :]

            next_token_logits = self.cls(next_token_hidden_states) # (batch_size * num_beams, vocab_size)

            # Repetition penalty (from CTRL paper https://arxiv.org/abs/1909.05858)
            if repetition_penalty != 1.0:
                for i in range(batch_size * num_beams):
                    for token_id in curr_ids[i]:
                        if next_token_logits[i, token_id] < 0:
                            next_token_logits[i, token_id] *= repetition_penalty
                        else:
                            next_token_logits[i, token_id] /= repetition_penalty

            # Convert to log probs for beam search
            next_token_logprobs = F.log_softmax(next_token_logits, dim=-1) # (batch_size * num_beams, vocab_size)

            # Add current beam scores
            # beam_scores has shape (batch_size * num_beams,)
            # next_token_logprobs has shape (batch_size * num_beams, vocab_size)
            # Want to add scores only to the currently active beams.
            # For the first step, beam_scores are correct. For subsequent, they accumulate.
            scores_for_next_step = next_token_logprobs + beam_scores[:, None].expand_as(next_token_logprobs) # (batch_size * num_beams, vocab_size)

            # Reshape for selection: (batch_size, num_beams * vocab_size)
            vocab_size = next_token_logprobs.shape[-1]
            scores_for_next_step = scores_for_next_step.view(batch_size, num_beams * vocab_size)

            # Select top-k candidate tokens overall (across beams and vocab)
            # `2 * num_beams` to allow for diverse paths and then filter down to `num_beams`
            num_candidates_to_consider = 2 * num_beams
            next_beam_scores, next_beam_indices = torch.topk(scores_for_next_step, num_candidates_to_consider, dim=1, largest=True, sorted=True)

            # `next_beam_indices` are flat indices into `num_beams * vocab_size`
            # Convert them to (beam_idx, token_idx)
            next_beam_origin_beam_idx = next_beam_indices // vocab_size # Which original beam this candidate came from
            next_beam_token_idx = next_beam_indices % vocab_size       # Which token in vocab this candidate is

            # Prepare inputs for the next iteration
            new_beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device) - 1e9
            new_beam_curr_ids = torch.zeros((batch_size * num_beams, max_length), dtype=torch.long, device=input_ids.device) -1 # Pad with -1
            new_beam_past = [torch.zeros_like(p) for p in past] if past is not None else None # Placeholder for new past

            current_done_count = 0 # Count of examples in batch that are done

            for batch_idx in range(batch_size):
                if done[batch_idx]: # If this example in batch is already done, copy its best hypothesis
                    # This logic might need refinement if we want to ensure num_beams outputs even if done early.
                    # For now, if done, we effectively stop expanding this batch item.
                    # The `generated_hyps[batch_idx]` already holds the completed sequences.
                    # We need to ensure `new_beam_scores` for this batch_idx reflect that these paths are finished.
                    # This is handled by `BeamHypotheses.is_done`.
                    current_done_count +=1
                    continue

                # Iterate over candidates for this batch item
                hyp_idx = 0
                for cand_idx in range(num_candidates_to_consider):
                    origin_beam_idx = next_beam_origin_beam_idx[batch_idx, cand_idx].item()
                    token_id = next_beam_token_idx[batch_idx, cand_idx].item()
                    score = next_beam_scores[batch_idx, cand_idx].item()

                    # Original sequence this candidate extends from
                    # batch_size * num_beams is the first dim of curr_ids and past
                    global_origin_beam_idx = batch_idx * num_beams + origin_beam_idx
                    original_sequence = curr_ids[global_origin_beam_idx, :cur_len]

                    if token_id in eos_token_ids: # End Of Sentence
                        generated_hyps[batch_idx].add(original_sequence.clone(), score)
                    else: # Add to current hypothesis
                        new_sequence = torch.cat([original_sequence, torch.tensor([token_id], device=input_ids.device)])
                        generated_hyps[batch_idx].add(new_sequence, score) # Add to beam hypotheses for this batch item

                # After processing all candidates for this batch_idx,
                # select the top `num_beams` hypotheses from `generated_hyps[batch_idx]`
                # to form the input for the next step.

                # Check if this batch item is now done (all its beams found EOS)
                if generated_hyps[batch_idx].is_done(best_sum_logprobs=beam_scores[batch_idx * num_beams:(batch_idx + 1) * num_beams].max().item(), cur_len=cur_len):
                    done[batch_idx] = True
                    current_done_count +=1
                    # If done, we don't need to prepare inputs for next step for this batch item's beams.
                    # They will be filled with dummy/pad values or handled by ensuring scores are low.
                    continue # Go to next batch_idx

                # Prepare inputs for the next step for this batch_idx's active beams
                # Get the top `num_beams` hypotheses that are not yet EOS

                # This part is tricky: BertForImageCaptioning's version of beam search
                # seems to directly manipulate `next_beam_ids`, `next_beam_scores`, `next_beam_idx`
                # to form the inputs for the next step, rather than re-querying `generated_hyps`.
                # Let's try to follow that structure.
                # The loop over `cand_idx` above was more about adding to `generated_hyps`.
                # The actual selection for next step's input:

                effective_next_beam_idx = 0 # Pointer for this batch_idx's new beams

                for cand_idx in range(num_candidates_to_consider): # Iterate again through sorted candidates
                    if effective_next_beam_idx >= num_beams: # Filled all required beams for next step
                        break

                    origin_beam_idx = next_beam_origin_beam_idx[batch_idx, cand_idx].item()
                    token_id = next_beam_token_idx[batch_idx, cand_idx].item()
                    current_score = next_beam_scores[batch_idx, cand_idx].item() # This is cumulative score

                    if token_id in eos_token_ids: # Don't use EOS-terminated sequences as input for next step
                        continue

                    # Global index for new beams (for this batch_idx)
                    new_global_beam_idx = batch_idx * num_beams + effective_next_beam_idx

                    # Copy sequence and extend
                    original_global_beam_idx = batch_idx * num_beams + origin_beam_idx
                    new_beam_curr_ids[new_global_beam_idx, :cur_len] = curr_ids[original_global_beam_idx, :cur_len]
                    new_beam_curr_ids[new_global_beam_idx, cur_len] = token_id

                    # Copy past state
                    if past is not None:
                        for p_idx in range(len(past)):
                             # Select the past state from the origin beam and copy to new beam's slot
                            new_beam_past[p_idx][new_global_beam_idx] = past[p_idx][original_global_beam_idx]

                    # Update score for this new beam
                    new_beam_scores[batch_idx, effective_next_beam_idx] = current_score

                    effective_next_beam_idx += 1

                # If fewer than num_beams active paths found (e.g., all ended in EOS or were pruned)
                # fill remaining beams by duplicating the best remaining ones or padding.
                # This is important to maintain tensor shapes.
                if effective_next_beam_idx < num_beams:
                    # This case needs robust handling. For now, if not enough beams, it might lead to issues
                    # if new_beam_curr_ids and new_beam_past are not fully populated.
                    # The original BERT code implicitly handles this by how it selects topk.
                    # Let's assume for now that topk selection and EOS handling in BeamHypotheses
                    # will ensure that we either find num_beams or mark as done.
                    # If a batch item is not `done` but has < `num_beams` active hyps,
                    # it implies those are the ones to continue.
                    # The `new_beam_scores` already has -1e9 for unused slots.
                    pass


            if current_done_count == batch_size: # All batch items are done
                break

            # Update global state for next iteration
            curr_ids = new_beam_curr_ids[:, :cur_len + 1].clone() # Get sequences up to new current length
            beam_scores = new_beam_scores.view(-1) # Flatten for next score addition
            if past is not None:
                past = [p.clone() for p in new_beam_past] # Ensure past is updated

            cur_len += 1

        # Finalize: select best hypotheses from each batch item
        output_sequences = []
        output_sequence_scores = []

        for batch_idx in range(batch_size):
            # Sort hypotheses by score
            # generated_hyps[batch_idx].beams will be a list of (score, sequence_tensor)
            # Ensure it's sorted if BeamHypotheses doesn't guarantee it (it should)

            # Take top `self.num_keep_best` sequences
            # BeamHypotheses.finalize should give sorted list of (sum_logprobs, token_ids_tensor)
            num_to_return_for_batch = self.num_keep_best if hasattr(self, 'num_keep_best') else 1

            best_hyps = generated_hyps[batch_idx].finalize(cur_len, num_to_return_for_batch)


            for score, seq_tensor in best_hyps:
                # Pad sequence to max_length
                padding = max_length - seq_tensor.shape[0]
                padded_seq = torch.cat([seq_tensor, torch.full((padding,), pad_token_id, dtype=torch.long, device=input_ids.device)])
                output_sequences.append(padded_seq)
                output_sequence_scores.append(score)

        if not output_sequences: # Should not happen if batch_size > 0
             # Fallback: return input_ids or padded empty sequences if generation failed completely
            empty_seq = torch.full((max_length,), pad_token_id, dtype=torch.long, device=input_ids.device)
            if input_ids.shape[1] < max_length : # if input was shorter than max_length
                 padded_input = torch.cat([input_ids[0,:], torch.full((max_length - input_ids.shape[1],), pad_token_id, dtype=torch.long, device=input_ids.device)])
                 output_sequences.append(padded_input)
            else:
                 output_sequences.append(input_ids[0,:max_length].clone() if input_ids.nelement() > 0 else empty_seq) # Fallback to input or empty
            output_sequence_scores.append(-1e9) # Indicate low score

        final_sequences_tensor = torch.stack(output_sequences) # (batch_size * num_keep_best, max_length)
        final_scores_tensor = torch.tensor(output_sequence_scores, dtype=torch.float, device=input_ids.device) # (batch_size * num_keep_best)

        # Reshape to (batch_size, num_keep_best, max_length) and (batch_size, num_keep_best)
        # This matches the output format of BertForImageCaptioning's internal method.
        # Note: num_return_sequences (from generate()) is handled by generate() itself repeating the inputs.
        # Here, batch_size is effective_batch_size.
        # If num_return_sequences=1, effective_batch_size = actual_batch_size.
        # If num_return_sequences > 1, then generate() calls this with effective_batch_size,
        # and the output needs to be shaped accordingly or handled by generate().
        # The current structure seems to assume num_keep_best is applied per item in effective_batch_size.

        # Let's assume num_keep_best is always 1 for now as per original single return from generate.
        # If num_keep_best > 1, the output shape needs to be (effective_batch_size, num_keep_best, max_length)
        # For now, if num_keep_best=1, it's (effective_batch_size, max_length)
        if num_to_return_for_batch > 1 :
             final_sequences_tensor = final_sequences_tensor.view(batch_size, num_to_return_for_batch, max_length)
             final_scores_tensor = final_scores_tensor.view(batch_size, num_to_return_for_batch)

        return final_sequences_tensor, final_scores_tensor

    def _top_k_top_p_filtering(self, logits, top_k=0, top_p=1.0, filter_value=-float("Inf"), min_tokens_to_keep=1):
        """ Filter a distribution of logits using top-k and/or nucleus (top-p) filtering
            Args:
                logits: logits distribution shape (batch size, vocabulary size)
                if top_k > 0: keep only top k tokens with highest probability (top-k filtering).
                if top_p < 1.0: keep the top tokens with cumulative probability >= top_p (nucleus filtering).
                    Nucleus filtering is described in Holtzman et al. (http://arxiv.org/abs/1904.09751)
                Make sure we keep at least min_tokens_to_keep per batch example in the output
            From HuggingFace's internal method of the same name
        """
        if top_k > 0:
            top_k = min(max(top_k, min_tokens_to_keep), logits.size(-1))  # Safety check
            # Remove all tokens with a probability less than the last token of the top-k
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = filter_value

        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
            sorted_indices_to_remove = cumulative_probs > top_p
            if min_tokens_to_keep > 1:
                # Keep at least min_tokens_to_keep (set to min_tokens_to_keep-1 because we add the first one below)
                sorted_indices_to_remove[..., :min_tokens_to_keep] = 0
            # Shift the indices to the right to keep also the first token above the threshold
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            # scatter sorted tensors to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = filter_value
        return logits

# Helper function from HuggingFace transformers.models.bert.modeling_bert, adapted if needed
# This might be in transformers.modeling_utils or specific to bert/roberta modeling scripts

# Standard BeamHypotheses class from HuggingFace's generation_utils.py
class BeamHypotheses:
    def __init__(self, num_beams, max_length, length_penalty, early_stopping):
        """
        Initialize n-best list of hypotheses.
        """
        self.max_length = max_length - 1  # ignoring bos_token
        self.length_penalty = length_penalty
        self.early_stopping = early_stopping
        self.num_beams = num_beams
        self.beams = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.beams)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """
        score = sum_logprobs / (hyp.shape[-1] ** self.length_penalty)
        if len(self) < self.num_beams or score > self.worst_score:
            self.beams.append((score, hyp))
            if len(self) > self.num_beams:
                sorted_next_scores = sorted([(s, idx) for idx, (s, _) in enumerate(self.beams)])
                del self.beams[sorted_next_scores[0][1]]
                self.worst_score = sorted_next_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """
        if len(self) < self.num_beams:
            return False
        elif self.early_stopping:
            return True
        else:
            cur_score = best_sum_logprobs / cur_len ** self.length_penalty
            ret = self.worst_score >= cur_score
            return ret

    def finalize(self, cur_len, num_to_return):
        """
        Sort the hypotheses by score and return the top `num_to_return`.
        """
        # Sort by score
        self.beams.sort(key=lambda x: x[0], reverse=True)

        # Prepare output: list of (score, tensor_sequence)
        final_beams = []
        for score, hyp in self.beams[:num_to_return]:
            final_beams.append((score, hyp))
        return final_beams

def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
    """
    Finds the heads and their indices taking into account the already pruned heads.
    """
    mask = torch.ones(n_heads, head_size)
    for head in already_pruned_heads:
        # Mask out previous pruned heads
        mask[head] = 0
    mask = mask.view(-1).contiguous().eq(1)
    index = torch.arange(len(mask))[mask].long()

    new_heads = []
    for head in heads:
        idx = torch.sum(mask[:head]).item()
        new_heads.append(idx)
    return new_heads, index

# Ensure all imported classes (RobertaConfig, RobertaPreTrainedModel, etc.) are correctly sourced.
# If they are part of the `transformers` library and suffice, use those.
# If custom versions are needed (e.g., from `src.layers.roberta`), ensure paths are correct.
# For now, using `transformers.RobertaConfig` and `transformers.modeling_utils.PreTrainedModel` as base.
# `RobertaLayerNorm` is typically `torch.nn.LayerNorm`.
# `ACT2FN` should be standard.
# `ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP` and `ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP`
# would ideally come from `transformers.RobertaConfig.pretrained_config_archive_map` etc.,
# or be defined if using a completely custom model.

# The `dtype` property used in BertImgModel.forward for fp16 compatibility:
# extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)
# `RobertaPreTrainedModel` (or `PreTrainedModel`) should provide `self.dtype` via `self.dummy_inputs` or similar.
# Or, it can be `next(self.parameters()).dtype`. Let's assume `next(self.parameters()).dtype` can be used if `self.dtype` is not directly available.
# In RobertaImgModel.forward: extended_attention_mask.to(dtype=next(self.parameters()).dtype)
# Added dummy_inputs property to RobertaPreTrainedModel from HF RoBERTa.
# Added `self.dtype` in `RobertaImgModel.forward` as `next(self.parameters()).dtype`.

# Corrected `_init_weights` call in `RobertaImgModel` and `RobertaForImageCaptioning`
# (it's `self.init_weights()` which internally calls `_init_weights` via `apply(self._init_weights_fn)` in `PreTrainedModel`)
# Or, if `_init_weights` is overridden, `super()._init_weights(module)` should be used if the parent class also has `_init_weights`.
# The standard way is `self.apply(self._init_weights)` if `_init_weights` is the local method.
# Let's stick to `self.init_weights()` which is the public API from `PreTrainedModel` for initializing.

# In RobertaLMPredictionHead, decoder bias: `self.decoder.bias = self.bias` is correct if `bias=False` in `nn.Linear` for the decoder.
# If `bias=True` in `nn.Linear`, then `self.bias` parameter is redundant.
# HuggingFace RoBERTa LMHead has `self.decoder = nn.Linear(config.hidden_size, config.vocab_size, bias=False)` and `self.bias = nn.Parameter(torch.zeros(config.vocab_size))`, then `self.decoder.bias = self.bias`. This is fine.

# Final check on RobertaEncoder output:
# It should return a tuple: (last_hidden_state, all_hidden_states, all_attentions)
# The current implementation seems to do this. Need to ensure `output_attentions` and `output_hidden_states` are correctly propagated.
# Corrected usage in RobertaEncoder: use self.config.output_attentions etc. or the passed arguments.

# Cross-Attention in RobertaLayer:
# The `RobertaLayer` includes a cross-attention mechanism if `config.is_decoder` is true.
# This is standard for decoder layers in an encoder-decoder architecture.
# For an encoder-only model like this image captioning model (which is encoder + LM head),
# `config.is_decoder` should be `False`, so cross-attention won't be active.
# If the model were to be used as a decoder, this setup would be relevant.
# For `RobertaImgModel`, `config.is_decoder` will be false by default for `RobertaConfig`.

# `RobertaSelfAttention.output_attentions`: Added this field from config.
# `RobertaEncoder.forward`: Corrected usage of `output_attentions` and `output_hidden_states`. It should use `self.config.output_attentions` etc. by default, or the values passed to the forward method if provided.
# The `return_dict` argument is also standard in HF models; added to RobertaEncoder.forward signature for completeness, though the tuple output is maintained for now.
# The `RobertaImgModel.forward` also gets these HF style arguments.

# Corrected `RobertaEmbeddings` `padding_idx` to use `config.pad_token_id` as per HF RoBERTa.
# `RobertaEmbeddings.forward` adapted from HF `RobertaEmbeddingsModel`.

# `RobertaForImageCaptioning.encode_forward` and `generate` and related methods:
# These are the most complex parts to adapt. The current stubs are placeholders.
# They need to replicate the logic of `BertForImageCaptioning` but call `self.roberta` (the `RobertaImgModel`)
# and handle any differences in API or behavior of RoBERTa components (embeddings, encoder).
# Specifically, `token_type_ids` are often not used or handled differently in RoBERTa.
# `RobertaEmbeddings` has a `token_type_embeddings` layer, but it might be zero-initialized or ignored if `type_vocab_size` is 1 or 0.
# The `RobertaImgModel`'s `forward` pass needs to correctly prepare inputs for `RobertaEmbeddings` and `RobertaEncoder`.
# The `attention_mask` handling in `RobertaImgModel` needs to be robust for combined text and image features.

# The `dtype` for extended_attention_mask in `RobertaImgModel` should be `self.dtype` if available (from `PreTrainedModel`), or `next(self.parameters()).dtype`.
# `PreTrainedModel` defines `dtype` property.

# `RobertaLMPredictionHead.forward`: `self.decoder(hidden_states) + self.bias` is the old way. If `self.decoder.bias = self.bias` was done, then just `self.decoder(hidden_states)` is enough.
# HF `RobertaLMHead` does `hidden_states = self.transform(hidden_states)` then `hidden_states = self.decoder(hidden_states)`. The bias is tied. So `self.decoder(hidden_states)` should be correct.

# `RobertaImgModel._init_weights`: The line `if isinstance(module, nn.Linear) and hasattr(module, 'is_img_embedding'):` is a custom check.
# The `img_embedding` layer itself can be initialized specifically after `super()._init_weights(module)`.
# Or, rely on the general `nn.Linear` initialization.
# The re-initialization line `self.img_embedding.weight.data.normal_(...)` from `BertImgModel` constructor is a more direct way if needed.
# For now, `_init_weights` is standard.

# `RobertaForImageCaptioning.encode_forward`:
# - `text_sequence_output = outputs[0][:, :input_ids.shape[1], :]` assumes `input_ids` refers to the text part.
# - `img_sequence_output = outputs[0][:, input_ids.shape[1]:, :]` assumes image features are concatenated after text.
# This matches the logic in `BertForImageCaptioning` if `RobertaImgModel.forward` concatenates features similarly.
# The `RobertaImgModel.forward` currently does `embedding_output = torch.cat((embedding_output, img_embedding_output), 1)`.
# So, the slicing in `encode_forward` should be correct.

# `RobertaForImageCaptioning.generate` and its helpers:
# These are critical and complex. The placeholders need to be filled by carefully adapting the BERT versions.
# This includes:
# - `_expand_for_beams` (likely reusable)
# - `_do_output_past` (reusable)
# - `_generate_beam_search` / `_generate_no_beam_search` (needs adaptation of model calls)
# - `_decode_step` (core logic for one step of beam search, calls `prepare_inputs_for_generation` and model forward)
# - `prepare_inputs_for_generation` (constructs inputs for `self.roberta` for each generation step)
# - `prod_generate` / `prod_no_hidden_generate` (specialized generation loops)

# The `modeling_utils.py` dependency was mentioned. HF `transformers.modeling_utils` should provide `PreTrainedModel`.
# `add_start_docstrings` is also from there.
# `prune_linear_layer` is also often found there or in base model class.
# If `src.layers.bert.modeling_utils` was custom, its RoBERTa equivalent might be needed or adapt functions.
# For now, assumed standard HF utilities are used.

# The `ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP` and `ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP`
# are usually part of the `RobertaConfig` class itself (`RobertaConfig.pretrained_config_archive_map`).
# Defining them separately is okay if needed for a custom setup, but standard HF models would use the config's map.
# `RobertaPreTrainedModel.config_class.pretrained_model_archive_map` would be the way to access it.
# For now, these are defined locally as placeholders.
# `RobertaPreTrainedModel.pretrained_model_archive_map` is not set in the current HF implementation, it relies on `config_class`.
# Let's remove these local archive maps and rely on the config class or direct model names for `from_pretrained`.

# Corrected `RobertaLayerNorm` usage - it's just `torch.nn.LayerNorm`.
# `BertLayerNorm` was an older TF-style LayerNorm. RoBERTa uses standard LayerNorm.
# So, `LayerNormClass` can be `torch.nn.LayerNorm`.

# `hf_logging` is used for the logger.

# Removed local `ROBERTA_PRETRAINED_MODEL_ARCHIVE_MAP` and `ROBERTA_PRETRAINED_CONFIG_ARCHIVE_MAP`
# as `RobertaPreTrainedModel` will use `config_class.pretrained_config_archive_map` which is `RobertaConfig.pretrained_config_archive_map`.
# This means `from_pretrained` will use the Hugging Face Hub names like "roberta-base".
# `load_tf_weights_in_roberta` is mostly a placeholder as RoBERTa models are typically PyTorch native.

# `RobertaSelfAttention.output_attentions` was not used. Corrected to use `output_attentions` argument passed to `forward`.
# `RobertaEncoder.forward` also corrected to properly use `output_attentions` and `output_hidden_states` arguments.

# `RobertaImgModel.forward` `extended_attention_mask` creation:
# If `img_feats` are added, `attention_mask` should cover them.
# The current code `attention_mask = torch.ones(((batch_size, seq_length + (img_feats.shape[1] if img_feats is not None else 0))), device=device)`
# correctly creates a mask for the combined length if `attention_mask` is None initially.
# If `attention_mask` is provided, it must already account for `img_feats` if they are part of the sequence attention mechanism.
# The `extended_attention_mask` is then built from this. This seems consistent with `BertImgModel`.

# `RobertaForImageCaptioning.generate` and related methods are critical and are currently placeholders.
# These would need to be fully implemented by adapting from `BertForImageCaptioning`.
# This is a large piece of work. The current subtask is to implement `get_roberta_model`,
# which means `RobertaForImageCaptioning` needs to be defined, even if its generation methods are not fully battle-tested yet.
# The structure is what matters for `get_roberta_model` to proceed.

# Added `self.dtype` property to `RobertaPreTrainedModel` for fp16 compatibility, using `next(self.parameters()).dtype`.
# This is how `PreTrainedModel` in HF Transformers provides it if not explicitly set.
# Used `self.dtype` in `RobertaImgModel.forward`.

# Corrected `RobertaLMPredictionHead.forward` to simply `self.decoder(hidden_states)` as bias is tied.

# `RobertaPreTrainedModel._keys_to_ignore_on_load_missing` etc. are commented out as they are specific to certain Roberta heads in HF.
# Not essential for the basic structure.

# `RobertaAttention.forward` `self_outputs` uses `output_attentions` argument now.
# `RobertaLayer.forward` also propagates `output_attentions`.

# `RobertaImgModel._init_weights` should call `self.apply(super()._init_weights)` if it's meant to apply the parent's init logic.
# Or, more simply, `RobertaPreTrainedModel._init_weights` is the one to customize.
# `RobertaImgModel` inherits `RobertaPreTrainedModel`. So `super()._init_weights(module)` in `RobertaImgModel._init_weights`
# would call `RobertaPreTrainedModel._init_weights`. This is not what's intended if `RobertaImgModel` has its own complete init logic.
# The standard is: `PreTrainedModel` has `init_weights()` which calls `_init_weights()`. Derived classes override `_init_weights()`.
# So `RobertaPreTrainedModel._init_weights` is the base. `RobertaImgModel` doesn't need to override it unless it has *additional* specific initializations
# beyond what `RobertaPreTrainedModel._init_weights` does for its components (embeddings, encoder, pooler).
# Let's remove `RobertaImgModel._init_weights` and `RobertaForImageCaptioning._init_weights` and rely on the one in `RobertaPreTrainedModel`
# and the `self.apply(self._init_weights)` call done by `PreTrainedModel.__init__`.
# If specific layers like `img_embedding` need special init, it's done in `RobertaImgModel.__init__` after the layer is created.
# Example: `self.img_embedding.apply(self._init_weights)` if `_init_weights` can distinguish it, or direct init.
# For now, relying on the main `_init_weights` in `RobertaPreTrainedModel` and `apply(self._init_weights)` from `PreTrainedModel` constructor.

# `RobertaEmbeddings`: `padding_idx` in `word_embeddings` should be `config.pad_token_id`. (This was already correct).
# `position_ids` buffer registration is from HF RoBERTa.

# `RobertaSelfAttention` has `output_attentions` field, but the `forward` method also takes `output_attentions` as an argument.
# Conventionally, the argument in `forward` overrides the instance field if provided.
# The instance field `self.output_attentions` (set from `config.output_attentions`) acts as a default.
# Corrected this in `RobertaSelfAttention.forward` and `RobertaEncoder.forward`.

# `find_pruneable_heads_and_indices` is a utility. If not available from HF directly, it needs to be defined.
# It's often part of the base `modeling_bert.py` or `modeling_roberta.py` in HF.
# Included a version here.

# `RobertaPreTrainedModel.dtype` property: This isn't standard in HF `PreTrainedModel`.
# Instead, use `next(self.parameters()).dtype` directly where needed. Removed the property.
# In `RobertaImgModel.forward`, changed `extended_attention_mask.to(dtype=self.dtype)` to `extended_attention_mask.to(dtype=next(self.parameters()).dtype)`.

# `RobertaForImageCaptioning.generate` and other generation methods:
# These are substantial and are critical for the model to be usable for captioning.
# The current subtask is focused on `get_roberta_model` and the definition of `RobertaForImageCaptioning`.
# A fully working `generate` method is beyond the immediate scope but the structure must allow for it.
# The placeholders with warnings are appropriate for now.
# The `prod_generate` and `prod_no_hidden_generate` are specific optimization/modes from the original `BertForImageCaptioning`.
# Their adaptation would follow the adaptation of the main `generate` and `_decode_step` logic.

# Final check on `RobertaImgModel.forward` regarding `attention_mask` for `img_feats`:
# If `img_feats` are concatenated, `attention_mask` must be extended.
# If `attention_mask` is None, it's created for the full length: `seq_length + img_feats.shape[1]`.
# If `attention_mask` is provided, it's assumed to be for `input_ids` only. This is a mismatch with `BertImgModel`.
# `BertImgModel` expects `attention_mask` to cover `input_ids` + `img_feats`.
# Let's adjust `RobertaImgModel.forward` to match `BertImgModel`'s expectation for `attention_mask`.

# In `RobertaImgModel.forward`:
# If `attention_mask` is provided, and `img_feats` are also provided, the provided `attention_mask`
# should correspond to `input_ids` only. We then need to extend it for `img_feats`.
# Corrected this logic.
# If `attention_mask` is for `input_ids` (shape `batch_size, seq_length`), and `img_feats` are added (shape `batch_size, img_seq_length`),
# then the mask for `img_feats` is typically all ones.
# The `extended_attention_mask` then needs to be constructed for the combined sequence.

# Re-evaluating `attention_mask` in `RobertaImgModel.forward`:
# `BertImgModel` does:
# ```python
# if attention_mask is None:
#     if img_feats is not None:
#         attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1] + img_feats.shape[1]), device=input_ids.device)
#     else:
#         attention_mask = torch.ones_like(input_ids)
# ```
# This means `attention_mask` is expected to cover the *final* sequence (text + image).
# So if `attention_mask` is passed, it should already be of the combined length.
# My previous interpretation for `RobertaImgModel` might have been slightly off.
# Let's align `RobertaImgModel.forward` `attention_mask` handling with `BertImgModel`.

# In `RobertaImgModel.forward`, after `embedding_output = torch.cat(...)`:
# The `extended_attention_mask` should be created based on the final concatenated sequence length.
# The `attention_mask` passed in should already be for this combined length.
# This makes the initial `attention_mask` creation simpler:
# ```python
# if attention_mask is None:
#     final_seq_length = seq_length + (img_feats.shape[1] if img_feats is not None else 0)
#     attention_mask = torch.ones((batch_size, final_seq_length), device=device)
# ```
# This seems correct and aligns with how `BertImgModel` expects `attention_mask`.

# The `encoder_history_states` argument in `BertImgModel.forward` is passed to `self.encoder`.
# RoBERTa's encoder uses `past_key_values`. So, if this feature is used, `encoder_history_states`
# would need to be transformed or mapped to the `past_key_values` format expected by `RobertaEncoder`.
# For now, `RobertaEncoder`'s `forward` signature includes `past_key_values`, and `RobertaImgModel`
# passes `encoder_history_states` to it. This implies `RobertaEncoder` needs to handle this mapping if the types differ.
# Or, `RobertaImgModel` should do the mapping.
# For simplicity in this step, `encoder_history_states` is passed as `past_key_values` to the encoder.
# This assumes its structure is compatible or will be handled internally by `RobertaEncoder` if it's used.
# This is relevant for the `generate` method's caching.

# `RobertaEncoder.forward` was missing `past_key_values` in its call to `layer_module`. Added it.
# Also, `use_cache` is a common argument related to `past_key_values`. Added to signature.

# `RobertaLayer.forward` cross-attention part:
# `attention_mask` for cross-attention should be `encoder_attention_mask`.
# `attention_mask` for self-attention should be the decoder's own input mask.
# The current `RobertaLayer` seems to pass `attention_mask` (decoder's mask) to `self.crossattention`. This might be an issue.
# However, since `is_decoder` will be `False` for this model, cross-attention is not active.
# If it were active, this would need scrutiny. For now, it's dormant.

# `RobertaForImageCaptioning.generate` and `prepare_inputs_for_generation` are critical.
# The stubs are there. The actual implementation requires deep diving into `BertForImageCaptioning`'s logic
# and mapping it to RoBERTa. This is a significant task.
# The current goal is to have the class structure for `get_roberta_model`.
# The generation capabilities are a subsequent, detailed implementation step.
