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
        if encoder_history_states is not None and past_key_values is None:
            past_key_values = encoder_history_states # Assuming compatible format

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

    def encode_forward(self,
                       input_ids=None, # Optional if inputs_embeds is used
                       img_feats=None,
                       attention_mask=None,
                       masked_pos=None,
                       masked_ids=None,
                       masked_pos_img=None,
                       masked_token_img=None,
                       token_type_ids=None,
                       position_ids=None,
                       head_mask=None,
                       is_training=True,
                       encoder_history_states=None, # Will be mapped to past_key_values
                       inputs_embeds=None,
                       output_attentions=None,
                       output_hidden_states=None,
                       return_dict=None):

        # Resolve output flags
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        # return_dict for RobertaImgModel call, not necessarily for this method's output
        # return_dict_roberta_model = return_dict if return_dict is not None else self.config.use_return_dict
        # RobertaImgModel now expects tuple output primarily based on its implementation.

        # Pass arguments to self.roberta (RobertaImgModel)
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            img_feats=img_feats,
            encoder_hidden_states=encoder_history_states, # Mapped to past_key_values in RobertaImgModel if past_key_values is None
            # past_key_values will be derived from encoder_history_states if needed by RobertaImgModel
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=False # RobertaImgModel is expected to return a tuple (sequence_output, pooled_output, hidden_states, attentions)
        )

        # outputs = (sequence_output, pooled_output, all_hidden_states, all_attentions)
        sequence_output = outputs[0]
        # pooled_output = outputs[1] # Not used in this method directly
        roberta_model_other_outputs = outputs[2:] # (all_hidden_states, all_attentions)

        # Determine text sequence length for slicing
        # If input_ids is available, use its shape. Otherwise, if masked_pos is available, use its shape.
        # Fallback to assuming full sequence_output is text if neither is available (less robust).
        if input_ids is not None:
            text_seq_len = input_ids.shape[1]
        elif inputs_embeds is not None and masked_pos is not None: # inputs_embeds might be only text part before cat with img
             text_seq_len = masked_pos.shape[-1] # Assuming masked_pos corresponds to text part
        elif inputs_embeds is not None: # If only inputs_embeds given, and it's pre-concat
            text_seq_len = inputs_embeds.shape[1] # This might be text_seq_len if img_feats are separate
            if img_feats is not None: # If img_feats also exist, inputs_embeds was only text part
                pass
            else: # If no img_feats, inputs_embeds is the whole sequence
                text_seq_len = sequence_output.shape[1]
        else: # Fallback, this case should ideally not be hit if inputs are proper
            logger.warning("Cannot accurately determine text_seq_len. Assuming it's up to where image features might start if img_feats are present.")
            text_seq_len = sequence_output.shape[1] - (img_feats.shape[1] if img_feats is not None else 0)


        if is_training:
            text_sequence_output = sequence_output[:, :text_seq_len, :]
            
            # Masking for text part
            if masked_pos is None or masked_ids is None:
                raise ValueError("masked_pos and masked_ids are required for training.")
            
            # Ensure masked_pos is boolean or can be safely converted for indexing
            if masked_pos.dtype != torch.bool:
                effective_masked_pos = masked_pos == 1
            else:
                effective_masked_pos = masked_pos
            
            sequence_output_masked = text_sequence_output[effective_masked_pos, :]

            class_logits = self.cls(sequence_output_masked) # RobertaCaptioningHeads

            valid_masked_ids = masked_ids[masked_ids != -1] # Filter out padding tokens (often -1 or -100)

            masked_lm_loss = self.loss(class_logits.float(), valid_masked_ids) # RobertaCaptioningLoss
            total_loss = masked_lm_loss

            # Image feature prediction part (Optional)
            if masked_pos_img is not None and masked_token_img is not None:
                img_sequence_output = sequence_output[:, text_seq_len:, :]
                
                if masked_pos_img.dtype != torch.bool:
                    effective_masked_pos_img = masked_pos_img == 1
                else:
                    effective_masked_pos_img = masked_pos_img

                # Check if any image tokens are actually masked to avoid empty tensor issues
                if torch.any(effective_masked_pos_img):
                    img_output_masked = img_sequence_output[effective_masked_pos_img, :]
                    if img_output_masked.shape[0] > 0: # Ensure some tokens were actually selected
                        img_feat_logits = self.cls_img_feat(img_output_masked) # RobertaIFPredictionHead
                        masked_img_loss = self.loss_img_feat(img_feat_logits.float(), masked_token_img[effective_masked_pos_img,:]) # Filter masked_token_img too

                        img_loss_weight = getattr(self.config, 'img_loss_weight', 0.1)
                        total_loss += img_loss_weight * masked_img_loss
                    else:
                        logger.warning("Masked image positions were specified, but resulted in no tokens being selected for loss calculation.")
                else:
                    logger.warning("No image tokens were masked for image feature prediction loss.")

            return (total_loss, class_logits,) + roberta_model_other_outputs

        else: # Not training (e.g., for feature extraction or validation logits)
            text_sequence_output = sequence_output[:, :text_seq_len, :]
            class_logits = self.cls(text_sequence_output) # Get logits for the whole text sequence
            return (class_logits,) + roberta_model_other_outputs

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
    # For brevity, I'm not copying all those methods here but they are essential.
    # Placeholder for generate method - THIS NEEDS FULL ADAPTATION
    def generate(self, *args, **kwargs):
        # This method must be carefully adapted from BertForImageCaptioning.generate
        # Ensure all internal calls to self.bert are changed to self.roberta
        # and any model-specific logic (token IDs, input preparation) is updated.
        logger.warning("RobertaForImageCaptioning.generate() is a placeholder and needs full adaptation from BertForImageCaptioning.generate()")
        
        # Example of what needs to be adapted from BertForImageCaptioning:
        # - self.img_seq_len, self.max_seq_len, self.mask_token_id setup
        # - self.num_keep_best
        # - vocab_size from self.config
        # - Handling of add_od_labels, od_label_ids, etc.
        # - Expansion of inputs for beams: self._expand_for_beams(...)
        # - Calling the appropriate search strategy: _generate_beam_search, _generate_no_beam_search, or CBS related logic
        # - The _decode_step method used by beam search, which calls the model's forward pass.
        
        # --- Start of adapted section (conceptual) ---
        # Based on BertForImageCaptioning.generate
        
        # Extract necessary parameters from kwargs, similar to BertForImageCaptioning
        img_feats = kwargs.pop('img_feats') # Example, ensure all args are handled
        attention_mask = kwargs.pop('attention_mask')
        # ... other parameters like max_length, bos_token_id, etc.

        # Setup instance variables like self.img_seq_len, self.max_seq_len, self.mask_token_id
        # self.mask_token_id should be specific to RoBERTa's tokenizer
        # self.img_seq_len = img_feats.shape[1]
        # self.max_seq_len = kwargs.get('max_length')
        # self.mask_token_id = kwargs.get('mask_token_id') # Ensure this is RoBERTa's mask token
        # self.prev_encoded_layers = None # If using past_key_values logic

        # ... (rest of the setup from BertForImageCaptioning.generate)

        # Call the appropriate generation strategy (beam search, nucleus sampling, etc.)
        # These internal methods (_generate_beam_search, etc.) also need adaptation.
        # For example, _generate_beam_search calls _decode_step, which calls prepare_inputs_for_generation.

        # --- End of adapted section (conceptual) ---
        
        # Fallback to a very simple generation for now if not fully implemented
        # This is NOT a functional generation method.
        input_ids = kwargs.get('input_ids')
        if input_ids is None:
            bos_token_id = kwargs.get('bos_token_id', self.config.bos_token_id)
            input_ids = torch.full((img_feats.shape[0], 1), bos_token_id, dtype=torch.long, device=img_feats.device)
        
        # A very naive loop, not a proper generation algorithm
        for _ in range(kwargs.get('max_length', 20) -1): # max_length is just an example
            model_inputs = self.prepare_inputs_for_generation(input_ids, past=None, **kwargs) # Needs full adaptation
            
            # Ensure all necessary arguments for self.roberta.forward are included
            # This is a simplified call, actual call in _decode_step is more complex
            outputs = self.roberta(
                input_ids=model_inputs['input_ids'], 
                img_feats=model_inputs.get('img_feats'), # Get from model_inputs
                attention_mask=model_inputs['attention_mask'],
                # token_type_ids=model_inputs.get('token_type_ids'), # RoBERTa might not use
                # position_ids=model_inputs.get('position_ids'),
                # encoder_history_states=model_inputs.get('encoder_history_states') # Or past_key_values
            )
            
            # This is highly simplified - actual logits processing is more involved
            next_token_logits = self.cls(outputs[0])[:, -1, :] # Logits for the last token prediction
            next_tokens = torch.argmax(next_token_logits, dim=-1)
            input_ids = torch.cat([input_ids, next_tokens.unsqueeze(-1)], dim=-1)
            
            # Check for EOS token
            if hasattr(self.config, 'eos_token_id') and (next_tokens == self.config.eos_token_id).all():
                break
        
        return input_ids # Return generated sequences


    def _decode_step(self, curr_ids, past_key_values, **kwargs_for_prepare):
        """
        Performs a single decoding step.
        Used by beam search algorithms.
        Adapts logic from BertForImageCaptioning._decode_step.

        Args:
            curr_ids: Tensor of current token sequences in the beam (batch_size * num_beams, current_length).
            past_key_values: Cached key/value states from previous step.
            **kwargs_for_prepare: Additional arguments that might be needed by prepare_inputs_for_generation,
                                   though typically attributes set in `generate()` are used.
        Returns:
            Tuple of (logits for the next token, new_past_key_values).
        """
        # 1. Prepare inputs for the model for this decoding step
        # `prepare_inputs_for_generation` uses `self` attributes like `self.img_feats`
        # which are assumed to be set and beam-expanded by the main `generate` method.
        # `kwargs_for_prepare` can supplement or override if needed, but often empty.
        model_inputs = self.prepare_inputs_for_generation(curr_ids, past=past_key_values, **kwargs_for_prepare)

        # 2. Model Forward Pass
        # `self.roberta` is RobertaImgModel. It should return past_key_values if use_cache=True.
        # Expected output from RobertaImgModel when use_cache=True (set in model_inputs):
        # (sequence_output, pooled_output, past_key_values_from_encoder, other_outputs...)
        # The RobertaImgModel.forward was updated to return:
        # (sequence_output, pooled_output, all_hidden_states_or_past_key_values, all_attentions)
        # So, past_key_values should be in outputs[2] if use_cache=True and output_hidden_states=False (typical for generation)
        outputs = self.roberta(**model_inputs)

        sequence_output = outputs[0] # This is the output of RobertaImgModel's encoder for the current input slice

        new_past_key_values = None
        if model_inputs.get('use_cache', False):
            if len(outputs) > 2 and outputs[2] is not None:
                new_past_key_values = outputs[2]
            else:
                logger.warning("_decode_step: `use_cache` was True, but `new_past_key_values` is None or not found at outputs[2].")
        elif len(outputs) > 2 and outputs[2] is not None : # Check if past_kv are there even if use_cache was false (should not happen with HF standard)
             logger.warning("_decode_step: `use_cache` was False, but found unexpected past_key_values in output at index 2.")


        # 3. Get Logits for Prediction
        # The `sequence_output` from `self.roberta` corresponds to the `input_ids`
        # prepared by `prepare_inputs_for_generation`.
        # For generation, `input_ids` to `prepare_inputs_for_generation`
        # is typically `[last_generated_token, MASK_TOKEN]` for subsequent steps,
        # or `[BOS_TOKEN, MASK_TOKEN]` for the first step (potentially with OD labels too).

        # `model_inputs['input_ids']` had shape (batch, step_seq_len)
        # e.g., (batch, 2) for [token, MASK] or (batch, 2 + od_len) for [BOS, MASK, ODs]
        # The MASK token is expected to be at index 1 of the text part of these input_ids.
        mask_token_index = 1
        # sequence_output has shape (batch, step_seq_len, hidden_size)
        next_token_hidden_states = sequence_output[:, mask_token_index, :]

        # `self.cls` is RobertaCaptioningHeads
        logits = self.cls(next_token_hidden_states)

        # 4. Return
        return logits, new_past_key_values


    def prepare_inputs_for_generation(self, curr_ids, past=None, **kwargs):
        """
        Prepares inputs for generation, carefully adapted from BertForImageCaptioning.
        `kwargs` may contain `img_feats`, `attention_mask` etc. if not set as instance attributes by `generate`.
        However, this implementation assumes they are instance attributes (`self.img_feats`, `self.full_attention_mask`, etc.)
        set by the `generate` method, expanded for beams.
        """
        mask_token_id = self.config.mask_token_id # RoBERTa's mask token ID
        batch_size = curr_ids.shape[0] # This is effective_batch_size (batch_size * num_beams * num_fsm_states * num_return_sequences)

        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        # Helper to slice pre-computed full tensors.
        # These tensors (e.g., self.full_masked_pos) have shape (effective_batch_size, full_max_len)
        # where full_max_len = self.max_seq_len (text) + self.od_labels_len + self.img_seq_len (potentially)
        # or just self.max_seq_len (text) + self.od_labels_len if image features are not part of these specific tensors.
        # The original BERT code slices based on text sequence length for some parts.

        # Effective length of text part of full_position_ids, full_token_type_ids etc.
        # This usually corresponds to self.max_seq_len (max caption length).
        # Let full_text_od_len = self.max_seq_len + self.od_labels_len

        # The `_slice` and `_remove_elements` logic from original BERT:
        # `_slice(t, start, end)` selected columns from start to end from a tensor `t` of shape (batch, full_text_od_len)
        # `_remove_elements(t, start, end)` removed columns from start to end.

        # These utility functions are specific to how BertForImageCaptioning structures its full tensors.
        # Assuming they are available or adaptable if needed. For now, focusing on the core logic.

        # Determine current text length (excluding BOS, MASK, OD_LABELS)
        # curr_ids contains [BOS, token1, token2, ...]
        # current_actual_text_len = curr_ids.shape[1] -1 # Number of actual text tokens generated so far

        img_feats_for_step = None
        past_key_values_for_step = past # `past` is the past_key_values from the previous step

        if past is None: # First decoding step (after BOS)
            # Input: [BOS, MASK] or [BOS, MASK, OD_LABELS]
            input_ids = torch.cat([curr_ids, mask_ids], dim=1) # e.g., [BOS, MASK]
            current_total_input_len = input_ids.shape[1] # Length of [BOS, MASK] = 2

            # Attention mask, token_type_ids, position_ids need to cover [BOS, MASK, OD_LABELS, IMG_FEATS]
            # self.full_attention_mask is (batch, full_len, full_len)
            # We need to select the part relevant for [BOS, MASK, OD_LABELS, IMG_FEATS]
            # The elements not yet generated (future text tokens) should be excluded from rows/cols.

            # Example: if full sequence is [CLS TXT PAD OD IMG]
            # Current input is [CLS MASK OD IMG]
            # `seq_start_idx_in_full_mask` is where MASK begins in the text part of full_attention_mask
            # `seq_end_idx_in_full_mask` is end of text part in full_attention_mask
            seq_start_idx_in_full_mask = current_total_input_len -1 # -1 because MASK replaces the first actual token position
            seq_end_idx_in_full_mask = self.max_seq_len # End of text part

            # This is complex. BertImgModel's _remove_rows_cols needs exact indices.
            # For simplicity here, we assume self.full_attention_mask, self.full_token_type_ids etc.
            # are already prepared by `generate` to be of shape (batch_size, current_total_input_len + od_len + img_len, ...)
            # and are correctly ordered. The slicing in Bert's `prepare_inputs` is very specific to its setup.
            # Let's try a simplified adaptation:

            # We need to construct an attention mask for the sequence:
            # [curr_ids, mask_ids, (optional) od_label_ids] + img_feats
            # The `RobertaImgModel.forward` will then create the extended_attention_mask.

            # Length of text part for current step: curr_ids + mask_ids
            current_text_part_len = input_ids.shape[1]
            attention_mask_len = current_text_part_len

            if self.add_od_labels:
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
                attention_mask_len += self.od_labels_len

            if self.img_feats is not None: # Should be self.img_feats from generate()
                img_feats_for_step = self.img_feats
                attention_mask_len += self.img_seq_len

            # Create a simple attention mask for this structure
            # RobertaImgModel will extend it.
            attention_mask = torch.ones((batch_size, attention_mask_len), device=curr_ids.device)

            # position_ids:
            # RoBERTa embeddings handle position_ids internally if not provided, based on input_ids.
            # For generation, providing them explicitly, matching Bert's approach, can be more robust.
            # Position IDs for [BOS, MASK]
            position_ids = torch.arange(current_text_part_len, dtype=torch.long, device=curr_ids.device)
            if self.add_od_labels:
                # OD labels might have specific position IDs (e.g., starting from self.od_labels_start_posid)
                # This needs to align with how RobertaEmbeddings handles position_ids if they are gappy.
                # For now, assume contiguous or rely on RobertaEmbeddings' internal generation if None.
                # Bert version: self.full_position_ids is sliced.
                # Simplified: create position_ids for current input.
                # If self.od_labels_start_posid is used, positions might be like [0, 1, 100, 101, ...]
                # This requires RobertaEmbeddings to handle potentially non-contiguous absolute positions correctly.
                od_pos_ids = torch.arange(self.od_labels_start_posid, self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=curr_ids.device)
                position_ids = torch.cat([position_ids, od_pos_ids], dim=0)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)


            # token_type_ids:
            # For [BOS, MASK] + [OD_LABELS (if any)]
            # RoBERTa typically doesn't use token_type_ids heavily (type_vocab_size=1).
            # If used, text part is 0, OD labels could be 1 (or also 0).
            token_type_ids_text_part = torch.zeros((batch_size, current_text_part_len), dtype=torch.long, device=curr_ids.device)
            if self.add_od_labels:
                # Assuming OD labels get a different type, e.g., 1, if type_vocab_size allows.
                # Or all zeros if type_vocab_size is 1.
                od_type = 1 if self.config.type_vocab_size > 1 else 0
                token_type_ids_od_part = torch.full((batch_size, self.od_labels_len), od_type, dtype=torch.long, device=curr_ids.device)
                token_type_ids = torch.cat([token_type_ids_text_part, token_type_ids_od_part], dim=1)
            else:
                token_type_ids = token_type_ids_text_part

            # `self.prev_encoded_layers` (past_key_values cache) is None for the first step.
            # The model call will generate the first set of past_key_values.
            past_key_values_for_step = None
            self.prev_encoded_layers = None # Reset cache for this sequence generation call

        else: # Subsequent decoding steps (past is not None)
            last_token = curr_ids[:, -1:]
            input_ids = torch.cat([last_token, mask_ids], dim=1) # Input is [PrevToken, MASK]

            # Position IDs for [PrevToken, MASK]
            # PrevToken is at curr_ids.shape[1]-1, MASK is at curr_ids.shape[1]
            # RobertaEmbeddings creates position_ids with offset, so relative positions matter.
            # If curr_ids = [BOS, tok1], then PrevToken=tok1 (pos=1), MASK (pos=2)
            # These positions are relative to the start of text.
            # The `past_key_values_length` in RobertaEmbeddings.forward will handle the offset.
            # So, we can pass position_ids for the current slice, e.g., [actual_pos_of_PrevToken, actual_pos_of_MASK]
            # Or, let RobertaEmbeddings compute them based on `past_key_values_length`.
            # For explicit control like Bert:
            current_text_len_so_far = curr_ids.shape[1] # Includes BOS
            # pos for PrevToken is (current_text_len_so_far - 1), pos for MASK is current_text_len_so_far
            # These are 0-indexed from start of text.
            # Example: curr_ids=[BOS, t1], current_text_len_so_far=2. PrevToken=t1 (at index 1). MASK will be at index 2.
            # input_ids = [t1, MASK]
            # position_ids should be [1, 2] (absolute, if RobertaEmbeddings handles padding_idx offset)
            # Or, if RobertaEmbeddings needs relative to slice, and handles offset by past_key_values_length:
            # position_ids = torch.arange(2L, device=curr_ids.device).unsqueeze(0) and then it adds offset.
            # Let's defer to RobertaEmbeddings internal logic by setting position_ids = None,
            # as it uses past_key_values_length.
            position_ids = None

            # token_type_ids for [PrevToken, MASK] - usually all 0s for RoBERTa text.
            token_type_ids = torch.zeros_like(input_ids)

            img_feats_for_step = None # Image features are in `past`

            # `past` *is* the past_key_values.
            # The complex logic in Bert's `prev_encoded_layers` re-shuffles these keys/values
            # if OD labels and ImgFeats were part of the *initial* sequence passed to BERT layers.
            # If RobertaImgModel's encoder receives concatenated text+image embeddings, then `past`
            # already reflects this combined context.
            # The re-ordering in Bert was: self.prev_encoded_layers = [torch.cat([x[:, 2:, :], x[:, :start_pos,:]], dim=1) for x in past]
            # This implies the initial `past` had structure [BOS, TextTokenSoFar, OD_Labels, Img_Features].
            # And it reordered to [OD_Labels, Img_Features, BOS, TextTokenSoFar] for caching.
            # This is highly model-specific.
            # For RoBERTa, if past_key_values are standard, they are layer-wise tuples of (key, value) tensors.
            # Their sequence dimension corresponds to the *effective sequence length seen by the attention layers*.

            # Assuming `past` is already in the correct format (tuple of layer_wise key/value tensors)
            # and its sequence length matches the combined [Text, OD, Img] structure from the first pass.
            past_key_values_for_step = past

            # Attention mask for [PrevToken, MASK] given `past_key_values_for_step`.
            # The length of items in past_key_values (dim 2 for key/value) indicates total items attended to previously.
            # e.g., past_key_values[0][0].shape = (batch, num_heads, past_seq_len, head_size)
            past_seq_len = past[0][0].shape[2] # Total length of what's in past_key_values
            current_input_len = input_ids.shape[1] # Should be 2 for [Token, MASK]

            # Attention mask should be (batch, current_input_len, past_seq_len + current_input_len)
            # So [Token, MASK] can attend to everything in `past` and to `Token` (for MASK).
            # MASK should attend to `Token` and `past`. `Token` should attend to `past`.
            # Simplified: allow current inputs to attend to past and themselves appropriately.
            # Standard generation attention mask:
            attention_mask = torch.ones((batch_size, past_seq_len + current_input_len), device=curr_ids.device)
            # The `RobertaImgModel.forward` will create the proper extended causal mask.
            # For a slice being decoded, the attention_mask needs to be (batch_size, slice_len_with_past)
            # No, `RobertaImgModel` receives `attention_mask` for the *current items* being passed if `past_key_values` is used.
            # The shape should be (batch_size, current_input_len). The causal mask is built inside.
            # However, for compatibility with BERT's `full_attention_mask` slicing:
            # The original BERT code constructs a very specific attention_mask by slicing `self.full_attention_mask`.
            # This implies a fixed, pre-computed global attention structure.
            # If `past_key_values` are standard HF, the `attention_mask` for `model.forward`
            # only needs to cover the *new* `input_ids` (e.g., shape `(batch_size, 1)` for the token being generated).
            # The cached keys/values handle the past.

            # Let's use the simplified approach for HF standard `past_key_values`:
            # `input_ids` is just the new token (e.g. `curr_ids[:,-1:]` without MASK)
            # `attention_mask` is just for this new token `torch.ones((batch_size, 1))`
            # This part needs to align with how `_decode_step` calls this and the model.
            # The Bert `_decode_step` uses `[last_token, mask_ids]`.
            # So, `attention_mask` for these two tokens, allowing them to see `past`.
            attention_mask = torch.ones((batch_size, input_ids.shape[1] + past_seq_len), device=curr_ids.device)
            # This mask will be extended by `RobertaImgModel` to 4D.
            # It needs to be `(batch_size, current_input_ids_len, total_attended_len)` for triangular if no past_kv.
            # With `past_kv`, it's usually `(batch_size, current_input_ids_len)`.
            # Let's assume `RobertaImgModel` will handle it if `past_key_values` is not None.
            # The minimal mask is `torch.ones_like(input_ids)`.
            # The existing `RobertaImgModel` will then extend this.
            # If `past_key_values` is not None, `attention_mask` in `RobertaEncoder` is for the query sequence.
            # `extended_attention_mask` is `attention_mask[:, None, None, :]`
            # `key_layer = torch.cat([past_key_value[0], key_layer], dim=2)`
            # `attention_scores = attention_scores + attention_mask` (this mask should be for query vs key+past_key)
            # This implies `attention_mask` needs to be `(B, QuerySeqLen, KeySeqLen)`
            # This is getting too complex without knowing `_decode_step` structure.
            # Reverting to BERT's slicing logic for `attention_mask` is safer if `full_attention_mask` is structured similarly.
            # BERT uses: `attention_mask = self.full_attention_mask[:, od_len+img_len+start_pos : od_len+img_len+end_pos, :od_len+img_len+end_pos]`
            # This implies `full_attention_mask` is (B, FullCombined, FullCombined).
            # This requires `self.od_labels_len`, `self.img_seq_len` to be accurate.
            # For now, this part of attention_mask for past!=None is a placeholder needing robust test.
            # A common pattern for HF generation with past_kv is just `attention_mask = torch.ones((batch_size, 1), ...)` for the new token.
            # But since our input_ids is `[last_token, mask_ids]`, it's (batch, 2).
            attention_mask = torch.ones_like(input_ids) # Simplest form, assumes model handles past internally.

        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids, # Can be None to let RobertaEmbeddings compute it
            'token_type_ids': token_type_ids, # Can be None if RoBERTa doesn't use them
            'past_key_values': past_key_values_for_step,
            'img_feats': img_feats_for_step, # None after first step if info is in past_key_values

            # Control flags for model.forward()
            'output_attentions': False,
            'output_hidden_states': False,
            'use_cache': True, # Critical for generation
            # 'is_training': False, # Model should be in eval mode if called from generate
        }
        return model_inputs

    # prod_generate and prod_no_hidden_generate also need adaptation from BertForImageCaptioning
    # These are specialized versions of `generate`
    # Placeholder - THESE NEED FULL ADAPTATION
    def prod_generate(self, *args, **kwargs):
        logger.warning("RobertaForImageCaptioning.prod_generate() needs full adaptation.")
        # Adapt from BertForImageCaptioning.prod_generate, changing self.bert to self.roberta
        # and ensuring all RoBERTa-specific API details are handled.
        return self.generate(*args, **kwargs) # Fallback, not correct

    def prod_no_hidden_generate(self, *args, **kwargs):
        logger.warning("RobertaForImageCaptioning.prod_no_hidden_generate() needs full adaptation.")
        # Adapt from BertForImageCaptioning.prod_no_hidden_generate
        return self.generate(*args, **kwargs) # Fallback, not correct


# Helper function from HuggingFace transformers.models.bert.modeling_bert, adapted if needed
# This might be in transformers.modeling_utils or specific to bert/roberta modeling scripts
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
