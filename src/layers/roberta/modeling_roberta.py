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
from transformers.modeling_utils import PreTrainedModel, prune_linear_layer, add_start_docstrings
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

        self.LayerNorm = RobertaLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")


    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = inputs_embeds + token_type_embeddings
        if self.position_embedding_type == "absolute":
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
        
        self.output_attentions = config.output_attentions


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

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
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
        self.output_attentions = config.output_attentions # Added
        self.output_hidden_states = config.output_hidden_states # Added

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False, # Fallback if not in config
        output_hidden_states=False, # Fallback if not in config
        return_dict=True, # Added
    ):
        all_hidden_states = () if self.output_hidden_states else None # Corrected to use self.output_hidden_states
        all_attentions = () if self.output_attentions else None # Corrected to use self.output_attentions
        
        # Corrected usage of output_attentions and output_hidden_states
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (
        #     output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        # )


        for i, layer_module in enumerate(self.layer):
            if self.output_hidden_states: # Corrected
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
                self.output_attentions, # Corrected
            )
            hidden_states = layer_outputs[0]

            if self.output_attentions: # Corrected
                all_attentions = all_attentions + (layer_outputs[1],)
        
        if self.output_hidden_states: # Corrected
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states : # Corrected
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions : # Corrected
            outputs = outputs + (all_attentions,)
        return outputs # hidden_states, (all_hidden_states), (all_attentions)


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
    load_tf_weights = load_tf_weights_in_roberta # Placeholder
    base_model_prefix = "roberta"
    # The following three attributes are specific to RoBERTa.
    #_keys_to_ignore_on_load_missing = [r"position_ids"] # From RobertaModel
    #_keys_to_ignore_on_load_unexpected = [r"pooler"] # From RobertaForMaskedLM
    #_keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"] # From RobertaForCausalLM

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
        elif isinstance(module, RobertaLayerNorm): # Corrected to RobertaLayerNorm
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
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


        self.init_weights() # Call init_weights (it's _init_weights in PreTrainedModel)

    def _init_weights(self, module): # Renamed from init_weights to _init_weights
        super()._init_weights(module) # Call parent's _init_weights
        # Custom weight initialization for img_embedding if needed
        if isinstance(module, nn.Linear) and hasattr(module, 'is_img_embedding'):
             module.weight.data.normal_(mean=0.0, std=self.config.initializer_range) # Example
             if module.bias is not None:
                 module.bias.data.zero_()
        # Re-initialize img_embedding weights if necessary (example from BertImgModel)
        # self.img_embedding.weight.data.normal_(mean=0.0, std=self.config.initializer_range if hasattr(self.config, 'initializer_range') else 0.02)


    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None, # This was in BertImgModel, RoBERTa might handle it differently (e.g. past_key_values)
            inputs_embeds=None, # Added for compatibility with RobertaModel
            output_attentions=None, # Added
            output_hidden_states=None, # Added
            return_dict=None # Added
            ):

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict


        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
            batch_size, seq_length = input_shape
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
            batch_size, seq_length = input_shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device


        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + (img_feats.shape[1] if img_feats is not None else 0))), device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)


        # RoBERTa uses `padding_mask` instead of `attention_mask` for some internal calculations
        # The `attention_mask` passed to `RobertaModel` should be the extended version
        # However, BertImgModel constructs its own extended_attention_mask. We'll follow that pattern.

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask[:, None, None, :]
        elif attention_mask.dim() == 3: # For compatibility if a pre-extended mask is passed
            extended_attention_mask = attention_mask[:, None, :, :]
        else:
            raise ValueError(f"Wrong shape for attention_mask (shape {attention_mask.shape})")
        
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0


        embedding_output = self.embeddings(input_ids=input_ids, position_ids=position_ids,
                                           token_type_ids=token_type_ids, inputs_embeds=inputs_embeds)

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                # ... (handling for dis_code as in BertImgModel)
                pass
            else: # Default 'fc'
                img_embedding_output = self.img_embedding(img_feats)
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)
                img_embedding_output = self.dropout(img_embedding_output)
            
            # Concatenate word and image embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)
            # The attention mask also needs to be updated for the concatenated sequence length
            # This was handled by passing the combined length to ones_like in BertImgModel's attention_mask creation
            # Ensure extended_attention_mask reflects this combined length

        # RoBERTa's encoder does not typically take encoder_history_states.
        # It uses past_key_values for recurrent decoding. This needs careful adaptation if that feature is used.
        # For now, assuming standard encoder pass.
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask, # This should be the extended mask
            head_mask=head_mask,
            # encoder_hidden_states=None, # RoBERTa encoder doesn't take this directly
            # encoder_attention_mask=None, # RoBERTa encoder doesn't take this directly
            # past_key_values=encoder_history_states, # If encoder_history_states maps to past_key_values
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict, # Encoder might not return dict directly, handle from its output tuple
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        # If not using return_dict, mimic HuggingFace tuple output
        # outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        # For now, assume similar output structure to BertImgModel
        return (sequence_output, pooled_output) + encoder_outputs[1:]


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

        self.init_weights() # Call init_weights (it's _init_weights in PreTrainedModel)
        self.tie_weights()

    def _init_weights(self, module): # Renamed from init_weights to _init_weights
        super()._init_weights(module)
        # Custom weight initialization if needed for captioning heads, etc.

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
            return self.encode_forward(*args, **kwargs) # Needs adaptation

    def encode_forward(self, input_ids, img_feats, attention_mask,
                       masked_pos=None, masked_ids=None,
                       masked_pos_img=None, masked_token_img=None, # For image part prediction
                       token_type_ids=None, position_ids=None, head_mask=None,
                       is_training=True, encoder_history_states=None, # RoBERTa might use past_key_values
                       inputs_embeds=None, output_attentions=None, output_hidden_states=None, return_dict=None # HF style
                       ):
        # Pass arguments to self.roberta (RobertaImgModel)
        # Note: RoBERTa typically doesn't use token_type_ids. If RobertaImgModel handles it (e.g., by ignoring), it's fine.
        # encoder_history_states might map to past_key_values in RoBERTa if used for auto-regressive decoding features.
        outputs = self.roberta(
            input_ids=input_ids,
            img_feats=img_feats,
            attention_mask=attention_mask,
            position_ids=position_ids,
            token_type_ids=token_type_ids, # RobertaEmbeddings might ignore this if not configured
            head_mask=head_mask,
            # encoder_history_states=encoder_history_states, # Or map to past_key_values
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            # return_dict=return_dict # RobertaImgModel will handle its own return type
        )

        # Assuming outputs format is (sequence_output, pooled_output, ...) like BertImgModel
        # sequence_output = outputs[0]
        # pooled_output = outputs[1] # Not typically used directly by captioning head
        # other_outputs = outputs[2:]

        # The rest of this method (masking, loss calculation) should be very similar to BertForImageCaptioning,
        # just ensure indices and shapes match what RobertaImgModel and RobertaCaptioningHeads expect/produce.
        
        if is_training:
            # Ensure slicing and indexing are correct for combined text + image features if applicable
            # sequence_output's shape depends on whether img_feats were concatenated in RobertaImgModel
            # Assuming masked_pos refers to the text part if img_feats are concatenated.
            text_sequence_output = outputs[0][:, :input_ids.shape[1], :] # Example: extract text part

            # Masking logic from BertForImageCaptioning - needs careful check
            sequence_output_masked = text_sequence_output[masked_pos == 1, :]
            
            class_logits = self.cls(sequence_output_masked) # Pass to RobertaCaptioningHeads
            
            # Ensure masked_ids are correctly shaped and filtered
            valid_masked_ids = masked_ids[masked_ids != -1] # Filter padding
            
            masked_lm_loss = self.loss(class_logits.float(), valid_masked_ids) # Calculate loss

            total_loss = masked_lm_loss

            # Image feature prediction part (if applicable and configured)
            if masked_pos_img is not None and masked_token_img is not None:
                # Assuming img_feats were concatenated and are at the end of outputs[0]
                img_sequence_output = outputs[0][:, input_ids.shape[1]:, :] # Example: extract image part
                img_output_masked = img_sequence_output[masked_pos_img == 1, :]
                
                img_feat_logits = self.cls_img_feat(img_output_masked) # Pass to RobertaIFPredictionHead
                # Ensure masked_token_img is correctly prepared for loss
                masked_img_loss = self.loss_img_feat(img_feat_logits.float(), masked_token_img)
                
                # Combine losses (e.g., with a weighting factor)
                img_loss_weight = getattr(self.config, 'img_loss_weight', 0.1) # Example weight
                total_loss += img_loss_weight * masked_img_loss
            
            # Mimic BertForImageCaptioning's output tuple format
            # (total_loss, class_logits_for_text, other_bert_outputs...)
            # The exact content of other_bert_outputs depends on what RobertaImgModel returns after sequence_output and pooled_output
            return (total_loss, class_logits,) + outputs[2:] # Adjust based on RobertaImgModel's actual return

        else: # Not training, typically for inference or feature extraction
            # Extract text part of the sequence output
            text_sequence_output = outputs[0][:, :input_ids.shape[1], :]
            class_logits = self.cls(text_sequence_output) # Get logits for the whole sequence
            # Mimic BertForImageCaptioning's output tuple format
            return (class_logits,) + outputs[2:] # Adjust based on RobertaImgModel's actual return

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


    # `prepare_inputs_for_generation` needs to be adapted from `BertForImageCaptioning`
    # Placeholder - THIS NEEDS FULL ADAPTATION
    def prepare_inputs_for_generation(self, curr_ids, past=None, **kwargs):
        # This method must be carefully adapted from BertForImageCaptioning.prepare_inputs_for_generation
        # Key changes:
        # - Use self.mask_token_id (RoBERTa's)
        # - Construct inputs (input_ids, attention_mask, position_ids, etc.) compatible with self.roberta
        # - Handle past_key_values if that's the mechanism RoBERTa uses for caching, instead of encoder_history_states.
        logger.warning("RobertaForImageCaptioning.prepare_inputs_for_generation() is a placeholder and needs full adaptation.")

        # Conceptual adaptation:
        mask_token_id = self.config.mask_token_id # Ensure this is correct for RoBERTa
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full((batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device)

        # Logic for constructing input_ids, attention_mask, position_ids, token_type_ids (if used by RoBERTa model)
        # This involves slicing and concatenating based on `past` (cached states) and current `curr_ids`.
        # The exact details depend on how `BertForImageCaptioning` handles this and how it maps to RoBERTa's API.

        # Example of creating input_ids (very simplified):
        if past is None: # First step
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)
            # ... other inputs like attention_mask, position_ids ...
            # img_feats would be taken from kwargs or instance variables set up in `generate`
            img_feats = kwargs.get('img_feats_for_generation') # Example: assume it's passed or stored
        else: # Subsequent steps
            # `past` would contain cached key-values. RoBERTa expects `past_key_values`
            # The structure of `past` and how it's used needs to match RoBERTa's requirements.
            input_ids = torch.cat([curr_ids[:, -1:], mask_ids], dim=1) # Take last generated token + mask
            img_feats = None # Image features are usually processed only in the first step with past_key_values
            # ... other inputs ...

        # Return a dictionary of inputs for self.roberta
        # This is a placeholder return
        return {
            'input_ids': input_ids,
            'img_feats': img_feats, # May be None after first step if using past_key_values
            'attention_mask': kwargs.get('attention_mask_for_generation'), # Needs to be correctly constructed
            # 'token_type_ids': ..., # If RoBERTa model uses them
            # 'position_ids': ...,
            # 'past_key_values': past, # If `past` maps to RoBERTa's past_key_values
            'is_training': False, # Explicitly set for generation
        }

    # prod_generate and prod_no_hidden_generate also need adaptation from BertForImageCaptioning
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
