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
"""PyTorch ALBERT model. """

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

from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings, find_pruneable_heads_and_indices, apply_chunking_to_forward)

import logging
from src.utils.comm import is_main_process
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
if not is_main_process():
    logger.disabled = True

from .file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
)
from .modeling_outputs import (
    BaseModelOutput,
    BaseModelOutputWithPooling,
    MaskedLMOutput,
    MultipleChoiceModelOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
    BaseModelOutputWithPastAndCrossAttentions,
)


ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/pytorch_model.bin",
    "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/pytorch_model.bin",
    "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/pytorch_model.bin",
    "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/pytorch_model.bin",
    "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/pytorch_model.bin",
    "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/pytorch_model.bin",
    "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/pytorch_model.bin",
    "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/pytorch_model.bin",
}

ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/config.json",
    "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/config.json",
    "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/config.json",
    "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/config.json",
    "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/config.json",
    "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/config.json",
    "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/config.json",
    "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/config.json",
}


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu}

class AlbertConfig(PretrainedConfig):
    r"""
        :class:`~pytorch_transformers.BertConfig` is the configuration class to store the configuration of a
        `BertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP
    model_type = "albert"

    def __init__(
        self,
        vocab_size=30000,
        embedding_size=128,
        hidden_size=4096,
        num_hidden_layers=12,
        num_hidden_groups=1,
        num_attention_heads=64,
        intermediate_size=16384,
        inner_group_num=1,
        hidden_act="gelu_new",
        hidden_dropout_prob=0,
        attention_probs_dropout_prob=0,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        classifier_dropout_prob=0.1,
        chunk_size_feed_forward=0,
        is_decoder=False,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=3,
        **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, bos_token_id=bos_token_id, eos_token_id=eos_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_hidden_groups = num_hidden_groups
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.inner_group_num = inner_group_num
        self.hidden_act = hidden_act
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.classifier_dropout_prob = classifier_dropout_prob
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.is_decoder = is_decoder

class AlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0
    ):
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

class AlbertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0 and not hasattr(config, "embedding_size"):
            raise ValueError(
                f"The hidden size ({config.hidden_size}) is not a multiple of the number of attention "
                f"heads ({config.num_attention_heads})"
            )

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            self.max_position_embeddings = config.max_position_embeddings
            self.distance_embedding = nn.Embedding(2 * config.max_position_embeddings - 1, self.attention_head_size)

        self.is_decoder = config.is_decoder

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

        if self.is_decoder:
            # if cross_attention save Tuple(torch.Tensor, torch.Tensor) of all cross attention key/value_states.
            # Further calls to cross_attention layer can then reuse the saved tensors
            # in the cross_attention layer by passing in cross_attention_past_key_value
            past_key_value = (key_layer, value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        if self.position_embedding_type == "relative_key" or self.position_embedding_type == "relative_key_query":
            seq_length = hidden_states.size()[1]
            position_ids_l = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(-1, 1)
            position_ids_r = torch.arange(seq_length, dtype=torch.long, device=hidden_states.device).view(1, -1)
            distance = position_ids_l - position_ids_r
            positional_embedding = self.distance_embedding(distance + self.max_position_embeddings - 1)
            positional_embedding = positional_embedding.to(dtype=query_layer.dtype)  # fp16 compatibility

            if self.position_embedding_type == "relative_key":
                relative_position_scores = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores
            elif self.position_embedding_type == "relative_key_query":
                relative_position_scores_query = torch.einsum("bhld,lrd->bhlr", query_layer, positional_embedding)
                relative_position_scores_key = torch.einsum("bhrd,lrd->bhlr", key_layer, positional_embedding)
                attention_scores = attention_scores + relative_position_scores_query + relative_position_scores_key

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
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

        outputs = (context_layer, attention_probs) if output_attentions else (context_layer,)

        if self.is_decoder:
            outputs = outputs + (past_key_value,)
        return outputs


class AlbertSelfOutput(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class AlbertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = AlbertSelfAttention(config)
        self.output = AlbertSelfOutput(config)
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


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.chunk_size_feed_forward = config.chunk_size_feed_forward
        self.seq_len_dim = 1
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.full_layer_layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.activation = ACT2FN[config.hidden_act]

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
            past_key_value=self_attn_past_key_value,
            output_attentions=output_attentions,
        )
        attention_output = self_attention_outputs[0]

        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        ffn_output = self.ffn(attention_output)
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)

        hidden_states = self.full_layer_layer_norm(ffn_output + attention_output)

        outputs = (hidden_states,) + outputs

        return outputs


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        output_attentions=False,
        output_hidden_states=False,
    ):
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        for i, layer_module in enumerate(self.albert_layers):
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
                output_attentions,
            )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + (layer_outputs[2],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        return tuple(
            v
            for v in [hidden_states, all_hidden_states, all_self_attentions, all_cross_attentions]
            if v is not None
        )


class AlbertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    ):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        all_cross_attentions = () if output_attentions and self.config.add_cross_attention else None

        next_decoder_cache = () if use_cache else None
        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](
                hidden_states,
                attention_mask,
                head_mask[group_idx * layers_per_group : (group_idx + 1) * layers_per_group],
                encoder_hidden_states,
                encoder_attention_mask,
                past_key_values,
                output_attentions,
                output_hidden_states,
            )
            hidden_states = layer_group_output[0]

            if use_cache:
                next_decoder_cache += layer_group_output[-1]

            if output_attentions:
                all_self_attentions = all_self_attentions + layer_group_output[1]
                if self.config.add_cross_attention:
                    all_cross_attentions = all_cross_attentions + layer_group_output[2]

            if output_hidden_states:
                all_hidden_states = all_hidden_states + layer_group_output[1]

        if not return_dict:
            return tuple(
                v
                for v in [
                    hidden_states,
                    next_decoder_cache,
                    all_hidden_states,
                    all_self_attentions,
                    all_cross_attentions,
                ]
                if v is not None
            )
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=hidden_states,
            past_key_values=next_decoder_cache,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
            cross_attentions=all_cross_attentions,
        )


class AlbertPooler(nn.Module):
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


class AlbertPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = AlbertConfig
    base_model_prefix = "albert"
    _keys_to_ignore_on_load_missing = [r"position_ids"]

    def _init_weights(self, module):
        """Initialize the weights."""
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
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)


class AlbertModel(AlbertPreTrainedModel):
    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        if add_pooling_layer:
            self.pooler = AlbertPooler(config)
        else:
            self.pooler = None

        # Initialize weights and apply final processing
        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """
        Prunes heads of the model. heads_to_prune: dict of {layer_num: list of heads to prune in this layer} ALBERT
        has a different architecture in that its layers are shared across groups, which then has inner groups. If an
        attention head is pruned in layer 0 of group 0, it will be pruned in all layers of group 0.
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            layer_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[layer_idx].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
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
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=self.dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )

        encoder_outputs = self.encoder(
            embedding_output,
            extended_attention_mask,
            head_mask=head_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPooling(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
        )

class AlbertLMPredictionHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.activation = ACT2FN[config.hidden_act]
        self.LayerNorm = nn.LayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        prediction_scores = hidden_states + self.bias
        return prediction_scores

class AlbertImgModel(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertImgModel, self).__init__(config)

        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertEncoder(config)
        self.pooler = AlbertPooler(config)

        self.img_dim = config.img_feature_dim
        logger.info('AlbertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.hidden_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.hidden_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.hidden_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.img_layer_norm_eps)

        self.init_weights()

    def _resize_token_embeddings(self, new_num_tokens):
        old_embeddings = self.embeddings.word_embeddings
        new_embeddings = self._get_resized_embeddings(old_embeddings, new_num_tokens)
        self.embeddings.word_embeddings = new_embeddings
        return self.embeddings.word_embeddings

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):

        if attention_mask is None:
            if img_feats is not None:
                attention_mask = torch.ones((input_ids.shape[0], input_ids.shape[1] + img_feats.shape[1]), device=input_ids.device)
            else:
                attention_mask = torch.ones_like(input_ids)

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        if attention_mask.dim() == 2:
            extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        elif attention_mask.dim() == 3:
            extended_attention_mask = attention_mask.unsqueeze(1)
        else:
            raise NotImplementedError

        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids,
                token_type_ids=token_type_ids)

        if encoder_history_states is not None:
            if encoder_history_states[0].shape[1] != 0:
                assert img_feats is None or img_feats.shape[1]==0, "Cannot take image features while using encoder history states"

        if img_feats is not None:
            img_embedding_output = self.img_embedding(img_feats)
            if self.use_img_layernorm:
                img_embedding_output = self.LayerNorm(img_embedding_output)

            img_embedding_output = self.dropout(img_embedding_output)
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]
        return outputs

class AlbertCaptioningHeads(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.predictions = AlbertLMPredictionHead(config)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class AlbertCaptioningLoss(nn.Module):
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

class AlbertForImageCaptioning(AlbertPreTrainedModel):
    r"""
    Albert for Image Captioning.
    """
    def __init__(self, config):
        super(AlbertForImageCaptioning, self).__init__(config)
        self.config = config
        self.albert = AlbertImgModel(config)
        self.cls = AlbertCaptioningHeads(config)
        self.loss = AlbertCaptioningLoss(config)

        self.init_weights()
        self.tie_weights()

    def tie_weights(self):
        if hasattr(self.config, 'tie_weights') and self.config.tie_weights:
            self._tie_or_clone_weights(self.cls.predictions.decoder,
                                       self.albert.embeddings.word_embeddings)
        freeze = False
        if hasattr(self.config, 'freeze_embedding'):
            freeze = self.config.freeze_embedding
        self.albert.embeddings.word_embeddings.weight.requires_grad = not freeze

    def forward(self, *args, **kwargs):
        is_decode = kwargs.get('is_decode', False)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask,
            masked_pos=None, masked_ids=None,
            token_type_ids=None, position_ids=None, head_mask=None,
            is_training=True, encoder_history_states=None):
        outputs = self.albert(input_ids, img_feats=img_feats, attention_mask=attention_mask,
                position_ids=position_ids, token_type_ids=token_type_ids,
                head_mask=head_mask,
                encoder_history_states=encoder_history_states)
        if is_training:
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            sequence_output_masked = sequence_output[masked_pos==1, :]
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != -1]
            masked_loss = self.loss(class_logits.float(), masked_ids)
            outputs = (masked_loss, class_logits,) + outputs[2:]
        else:
            sequence_output = outputs[0][:, :input_ids.shape[-1], :]
            class_logits = self.cls(sequence_output)
            outputs = (class_logits,) + outputs[2:]
        return outputs

    def prepare_inputs_for_generation(self, curr_ids, past=None):
        # NOTE: if attention is on, it should be the token used to mask words in training
        mask_token_id = self.mask_token_id
        batch_size = curr_ids.shape[0]
        mask_ids = torch.full(
            (batch_size, 1), mask_token_id, dtype=torch.long, device=curr_ids.device
        )

        def _slice(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return t[:, start: end]

        def _remove_elements(t, start, end):
            if t is None:
                return t
            assert t.shape == (batch_size, self.max_seq_len + self.od_labels_len)
            return torch.cat([t[:, :start], t[:, end:]], dim=1)

        if past is None:
            input_ids = torch.cat([curr_ids, mask_ids], dim=1)

            curr_len = input_ids.shape[1]
            full_len = self.max_seq_len + self.od_labels_len + self.img_seq_len
            assert self.full_attention_mask.shape == (batch_size,
                    full_len, full_len)

            def _remove_rows_cols(t, row_start, row_end, col_start, col_end):
                t00 = t[:, :row_start, :col_start]
                t01 = t[:, :row_start, col_end:]
                t10 = t[:, row_end:, :col_start]
                t11 = t[:, row_end:, col_end:]
                res = torch.cat([torch.cat([t00, t01], dim=2), torch.cat([t10, t11],
                            dim=2)], dim=1)
                assert res.shape == (t.shape[0], t.shape[1]-row_end+row_start,
                        t.shape[2]-col_end+col_start)
                return res

            seq_start = curr_len
            seq_end = self.max_seq_len
            attention_mask = _remove_rows_cols(self.full_attention_mask, seq_start,
                    seq_end, seq_start, seq_end)

            masked_pos = _remove_elements(self.full_masked_pos, seq_start, seq_end)
            token_type_ids = _remove_elements(self.full_token_type_ids, seq_start, seq_end)
            position_ids = _remove_elements(self.full_position_ids, seq_start, seq_end)
            img_feats = self.img_feats

            if self.add_od_labels:
                assert self.od_label_ids.shape[1] == self.od_labels_len
                input_ids = torch.cat([input_ids, self.od_label_ids], dim=1)
        else:
            last_token = curr_ids[:, -1:]
            # The representation of last token should be re-computed, because
            # it depends on both self-attention context and input tensor
            input_ids = torch.cat([last_token, mask_ids], dim=1)
            start_pos = curr_ids.shape[1] - 1
            end_pos = start_pos + input_ids.shape[1]
            masked_pos = _slice(self.full_masked_pos, start_pos, end_pos)
            token_type_ids = _slice(self.full_token_type_ids, start_pos, end_pos)
            position_ids = _slice(self.full_position_ids, start_pos, end_pos)

            img_feats = None
            assert past[0].shape[0] == batch_size
            if self.prev_encoded_layers is None:
                assert start_pos == 1  # the first token after BOS
                assert past[0].shape[1] == 2 + self.od_labels_len + self.img_seq_len
                # reorder to [od_labels, img_feats, sentence]
                self.prev_encoded_layers = [
                        torch.cat([x[:, 2:, :], x[:, :start_pos,:]], dim=1)
                        for x in past]
                s2s = self.full_attention_mask[:, :self.max_seq_len,
                        :self.max_seq_len]
                s2i = self.full_attention_mask[:, :self.max_seq_len,
                        self.max_seq_len:]
                i2s = self.full_attention_mask[:, self.max_seq_len:,
                        :self.max_seq_len]
                i2i = self.full_attention_mask[:, self.max_seq_len:,
                        self.max_seq_len:]
                self.full_attention_mask = torch.cat(
                        [torch.cat([i2i, i2s], dim=2),
                        torch.cat([s2i, s2s], dim=2)],
                        dim=1)
            else:
                assert start_pos > 1
                assert past[0].shape[1] == 2
                self.prev_encoded_layers = [torch.cat([x, p[:, :-1, :]], dim=1)
                        for x, p in zip(self.prev_encoded_layers, past)]

            attention_mask = self.full_attention_mask[:,
                self.od_labels_len+self.img_seq_len+start_pos: self.od_labels_len+self.img_seq_len+end_pos,
                :self.od_labels_len+self.img_seq_len+end_pos]

        return {'input_ids': input_ids, 'img_feats': img_feats,
            'masked_pos': masked_pos, 'attention_mask': attention_mask,
            'token_type_ids': token_type_ids, 'position_ids': position_ids,
            'is_training': False,
            'encoder_history_states': self.prev_encoded_layers}

    def get_output_embeddings(self):
        return self.cls.predictions.decoder

    def generate(self, img_feats, attention_mask, masked_pos, token_type_ids=None,
            position_ids=None, head_mask=None, input_ids=None, max_length=None,
            do_sample=None, num_beams=None, temperature=None, top_k=None, top_p=None,
            repetition_penalty=None, bos_token_id=None, pad_token_id=None,
            eos_token_ids=None, mask_token_id=None, length_penalty=None,
            num_return_sequences=None,
            num_keep_best=1, is_decode=None,
            add_od_labels=False, od_labels_start_posid=None,
            ):
        """ Generates captions given image features
        """
        assert is_decode
        batch_size = img_feats.shape[0]
        self.img_seq_len = img_feats.shape[1]
        self.max_seq_len = max_length
        self.mask_token_id = mask_token_id
        self.prev_encoded_layers = None
        # NOTE: num_keep_best is not equavilant to num_return_sequences
        # num_keep_best is the number of hypotheses to keep in beam search
        # num_return_sequences is the repeating times of input, coupled with
        # do_sample=True can generate more than one samples per image
        self.num_keep_best = num_keep_best

        vocab_size = self.config.vocab_size

        self.add_od_labels = add_od_labels
        # avoid position_ids collision of caption and od labels
        self.od_labels_start_posid = max(od_labels_start_posid, self.max_seq_len)
        if self.add_od_labels:
            # get od labels part from input_ids
            assert input_ids.shape[0] == batch_size
            od_label_ids = input_ids[:, self.max_seq_len:]
            self.od_labels_len = input_ids.shape[1] - self.max_seq_len
            input_ids = None
        else:
            self.od_labels_len = 0
            od_label_ids = None
            assert input_ids.shape == (batch_size, self.max_seq_len)
            input_ids = None

        if input_ids is None:
            input_ids = torch.full(
                (batch_size, 1), bos_token_id, dtype=torch.long, device=next(self.parameters()).device
            )
        else:
            assert input_ids.dim() == 2, "Input prompt should be of shape (batch_size, sequence length)."
            assert input_ids.shape[0] == batch_size, "Input batch size must match image features"

        cur_len = input_ids.shape[1]
        if  num_return_sequences != 1:
            # Expand input to num return sequences
            input_ids = self._expand_for_beams(input_ids, num_return_sequences)
            effective_batch_size = batch_size * num_return_sequences
        else:
            effective_batch_size = batch_size

        if position_ids is None:
            position_ids = torch.arange(self.max_seq_len, dtype=torch.long, device=input_ids.device)
            posids_len = self.max_seq_len
            if self.add_od_labels:
                od_labels_posids = torch.arange(
                        self.od_labels_start_posid,
                        self.od_labels_start_posid + self.od_labels_len, dtype=torch.long, device=input_ids.device)
                position_ids = torch.cat([position_ids, od_labels_posids])
                posids_len += self.od_labels_len
            position_ids = position_ids.unsqueeze(0).expand([batch_size, posids_len])

        num_expand = num_beams * num_return_sequences
        self.od_label_ids = self._expand_for_beams(od_label_ids, num_expand)
        self.img_feats = self._expand_for_beams(img_feats, num_expand)
        self.full_attention_mask = self._expand_for_beams(attention_mask, num_expand)
        self.full_masked_pos = self._expand_for_beams(masked_pos, num_expand)
        self.full_token_type_ids = self._expand_for_beams(token_type_ids, num_expand)
        self.full_position_ids = self._expand_for_beams(position_ids, num_expand)
        self.full_head_mask = self._expand_for_beams(head_mask, num_expand)

        if num_beams > 1:
            output = self._generate_beam_search(
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
                effective_batch_size,
                length_penalty,
                num_beams,
                vocab_size,
            )
        else:
            output = self._generate_no_beam_search(
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
                effective_batch_size,
            )

        return output

    def _expand_for_beams(self, x, num_expand):
        if x is None or num_expand == 1:
            return x

        input_shape = list(x.shape)
        expanded_shape = input_shape[:1] + [num_expand] + input_shape[1:]
        x = x.unsqueeze(1).expand(expanded_shape)
        # (batch_size * num_expand, ...)
        x = x.contiguous().view([input_shape[0] * num_expand] + input_shape[1:])
        return x
