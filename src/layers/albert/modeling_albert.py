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
"""PyTorch ALBERT model."""

from __future__ import absolute_import, division, print_function, unicode_literals

import json
import math
import os
import sys
from io import open

import torch
from torch import nn
from torch.distributions import kl_divergence, Categorical
from torch.nn import CrossEntropyLoss, MSELoss
import torch.nn.functional as F

from torch.nn.utils.weight_norm import weight_norm

from .modeling_utils import (WEIGHTS_NAME, CONFIG_NAME, PretrainedConfig, PreTrainedModel,
                             prune_linear_layer, add_start_docstrings)

import logging
from src.utils.comm import is_main_process
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
if not is_main_process():
    logger.disabled = True

import torch.utils.checkpoint as torch_checkpoint
ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    'albert-base-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-pytorch_model.bin",
    'albert-large-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v1-pytorch_model.bin",
    'albert-xlarge-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v1-pytorch_model.bin",
    'albert-xxlarge-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v1-pytorch_model.bin",
    'albert-base-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-pytorch_model.bin",
    'albert-large-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-pytorch_model.bin",
    'albert-xlarge-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-pytorch_model.bin",
    'albert-xxlarge-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-pytorch_model.bin",
}

ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP = {
    'albert-base-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v1-config.json",
    'albert-large-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v1-config.json",
    'albert-xlarge-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v1-config.json",
    'albert-xxlarge-v1': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v1-config.json",
    'albert-base-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-base-v2-config.json",
    'albert-large-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-large-v2-config.json",
    'albert-xlarge-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xlarge-v2-config.json",
    'albert-xxlarge-v2': "https://s3.amazonaws.com/models.huggingface.co/bert/albert-xxlarge-v2-config.json",
}

def load_tf_weights_in_albert(model, config, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model."""
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        logger.error("Loading a TensorFlow model in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions.")
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    logger.info("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        logger.info("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split('/')
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m", "global_step"] for n in name):
            logger.info("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r'[A-Za-z]+_\d+', m_name):
                l = re.split(r'_(\d+)', m_name)
            else:
                l = [m_name]
            if l[0] == 'kernel' or l[0] == 'gamma':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'output_bias' or l[0] == 'beta':
                pointer = getattr(pointer, 'bias')
            elif l[0] == 'output_weights':
                pointer = getattr(pointer, 'weight')
            elif l[0] == 'squad':
                pointer = getattr(pointer, 'classifier')
            else:
                try:
                    pointer = getattr(pointer, l[0])
                except AttributeError:
                    logger.info("Skipping {}".format("/".join(name)))
                    continue
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == '_embeddings':
            pointer = getattr(pointer, 'weight')
        elif m_name == 'kernel':
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        logger.info("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def swish(x):
    return x * torch.sigmoid(x)


def gelu_new(x):
    """ Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
        Also see https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


def gelu(x):
    """ Original Implementation of the gelu activation function in Google Bert repo when initially created.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish, "gelu_new": gelu_new}


class AlbertConfig(PretrainedConfig):
    r"""
        :class:`~transformers.AlbertConfig` is the configuration class to store the configuration of a
        `AlbertModel`.


        Arguments:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `AlbertModel`.
            embedding_size: size of the embedding layer.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_hidden_groups: Number of groups for the hidden layers, parameters in the same group are shared.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            inner_group_num: The number of inner repetition of attention and ffn.
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
                `AlbertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
            layer_norm_eps: The epsilon used by LayerNorm.
    """
    pretrained_config_archive_map = ALBERT_PRETRAINED_CONFIG_ARCHIVE_MAP

    def __init__(self,
                 vocab_size_or_config_json_file=30000,
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
                 **kwargs):
        super(AlbertConfig, self).__init__(**kwargs)
        if isinstance(vocab_size_or_config_json_file, str) or (sys.version_info[0] == 2
                        and isinstance(vocab_size_or_config_json_file, unicode)):
            with open(vocab_size_or_config_json_file, "r", encoding='utf-8') as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.embedding_size = embedding_size
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_hidden_groups = num_hidden_groups
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.inner_group_num = inner_group_num
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.layer_norm_eps = layer_norm_eps
        else:
            raise ValueError("First argument must be either a vocabulary size (int)"
                             "or the path to a pretrained model config file (str)")


# Use standard PyTorch LayerNorm for better compatibility
# This avoids APEX/CUDA dependency issues in different environments
LayerNormClass = torch.nn.LayerNorm
AlbertLayerNorm = torch.nn.LayerNorm


class AlbertEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(AlbertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = AlbertLayerNorm(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertAttention(nn.Module):
    def __init__(self, config):
        super(AlbertAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.attention_dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.LayerNorm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.num_attention_heads, self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and emove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.query = prune_linear_layer(self.query, index)
        self.key = prune_linear_layer(self.key, index)
        self.value = prune_linear_layer(self.value, index)
        self.dense = prune_linear_layer(self.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.num_attention_heads = self.num_attention_heads - len(heads)
        self.all_head_size = self.attention_head_size * self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask=None, head_mask=None):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in AlbertModel forward() function)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attention_dropout(attention_probs)

        # Mask heads if we want to
        # Note: head_mask support is disabled in this codebase like BERT
        if head_mask is not None:
            logger.warning("head_mask is not supported in this ALBERT implementation")
            # Skip head_mask processing to avoid dimension issues

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        projected_context_layer = self.dense(context_layer)
        projected_context_layer_dropout = self.dropout(projected_context_layer)
        layernormed_context_layer = self.LayerNorm(input_tensor + projected_context_layer_dropout)
        outputs = (layernormed_context_layer, attention_probs) if self.output_attentions else (layernormed_context_layer,)
        return outputs


class AlbertLayer(nn.Module):
    def __init__(self, config):
        super(AlbertLayer, self).__init__()

        self.config = config
        self.full_layer_layer_norm = AlbertLayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.attention = AlbertAttention(config)
        self.ffn = nn.Linear(config.hidden_size, config.intermediate_size)
        self.ffn_output = nn.Linear(config.intermediate_size, config.hidden_size)
        self.activation = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        # Disable head_mask support like BERT does in this codebase  
        attention_outputs = self.attention(hidden_states, attention_mask, head_mask=None)
        ffn_output = self.ffn(attention_outputs[0])
        ffn_output = self.activation(ffn_output)
        ffn_output = self.ffn_output(ffn_output)
        ffn_output = self.dropout(ffn_output)
        hidden_states = self.full_layer_layer_norm(ffn_output + attention_outputs[0])

        outputs = (hidden_states,) + attention_outputs[1:]  # add attentions if we output them
        return outputs


class AlbertLayerGroup(nn.Module):
    def __init__(self, config):
        super(AlbertLayerGroup, self).__init__()

        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.albert_layers = nn.ModuleList([AlbertLayer(config) for _ in range(config.inner_group_num)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        layer_hidden_states = ()
        layer_attentions = ()

        for layer_index, albert_layer in enumerate(self.albert_layers):
            # Disable head_mask support like BERT does in this codebase
            layer_output = albert_layer(hidden_states, attention_mask, head_mask=None)
            hidden_states = layer_output[0]

            if self.output_attentions:
                layer_attentions = layer_attentions + (layer_output[1],)

            if self.output_hidden_states:
                layer_hidden_states = layer_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (layer_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (layer_attentions,)
        return outputs  # last-layer hidden state, (layer hidden states), (layer attentions)


class AlbertTransformer(nn.Module):
    def __init__(self, config):
        super(AlbertTransformer, self).__init__()

        self.config = config
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.embedding_hidden_mapping_in = nn.Linear(config.embedding_size, config.hidden_size)
        self.albert_layer_groups = nn.ModuleList([AlbertLayerGroup(config) for _ in range(config.num_hidden_groups)])

    def forward(self, hidden_states, attention_mask=None, head_mask=None):
        hidden_states = self.embedding_hidden_mapping_in(hidden_states)

        all_attentions = ()

        if self.output_hidden_states:
            all_hidden_states = (hidden_states,)

        for i in range(self.config.num_hidden_layers):
            # Number of layers in a hidden group
            layers_per_group = int(self.config.num_hidden_layers / self.config.num_hidden_groups)

            # Index of the hidden group
            group_idx = int(i / (self.config.num_hidden_layers / self.config.num_hidden_groups))

            layer_group_output = self.albert_layer_groups[group_idx](hidden_states,
                                                                     attention_mask,
                                                                     head_mask[group_idx * layers_per_group:(group_idx + 1) * layers_per_group])
            hidden_states = layer_group_output[0]

            if self.output_attentions:
                all_attentions = all_attentions + layer_group_output[-1]

            if self.output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)
        return outputs  # last-layer hidden state, (all hidden states), (all attentions)


class AlbertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = AlbertConfig
    pretrained_model_archive_map = ALBERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_albert
    base_model_prefix = "albert"

    def _init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, AlbertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()
    
    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, AlbertLayerNorm) or isinstance(module, LayerNormClass):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class AlbertModel(AlbertPreTrainedModel):
    r"""
    Outputs: `Tuple` comprising various elements depending on the configuration (config) and inputs:
        **last_hidden_state**: ``torch.FloatTensor`` of shape ``(batch_size, sequence_length, hidden_size)``
            Sequence of hidden-states at the output of the last layer of the model.
        **pooler_output**: ``torch.FloatTensor`` of shape ``(batch_size, hidden_size)``
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during Bert pretraining. This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        **hidden_states**: (`optional`, returned when ``config.output_hidden_states=True``)
            list of ``torch.FloatTensor`` (one for the output of each layer + the output of the embeddings)
            of shape ``(batch_size, sequence_length, hidden_size)``:
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        **attentions**: (`optional`, returned when ``config.output_attentions=True``)
            list of ``torch.FloatTensor`` (one for each layer) of shape ``(batch_size, num_heads, sequence_length, sequence_length)``:
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention heads.
    """
    def __init__(self, config):
        super(AlbertModel, self).__init__(config)

        self.config = config
        self.embeddings = AlbertEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.apply(self.init_weights)

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            ALBERT has a different architecture in that its layers are shared across groups, which then has inner groups.
            If an ALBERT model has 12 hidden layers and 2 hidden groups, with two inner groups, there
            is a total of 4 different layers.

            Layers are shared across groups: [0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3]
            Inner group order: [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2]
        """
        for layer, heads in heads_to_prune.items():
            group_idx = int(layer / self.config.inner_group_num)
            inner_group_idx = int(layer - group_idx * self.config.inner_group_num)
            self.encoder.albert_layer_groups[group_idx].albert_layers[inner_group_idx].attention.prune_heads(heads)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None):
        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
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

        if inputs_embeds is None:
            inputs_embeds = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)
        encoder_outputs = self.encoder(inputs_embeds,
                                       extended_attention_mask,
                                       head_mask=head_mask)
        sequence_output = encoder_outputs[0]

        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class AlbertMLMHead(nn.Module):
    def __init__(self, config, albert_model_embedding_weights):
        super(AlbertMLMHead, self).__init__()

        self.LayerNorm = AlbertLayerNorm(config.embedding_size)
        self.bias = nn.Parameter(torch.zeros(config.vocab_size))
        self.dense = nn.Linear(config.hidden_size, config.embedding_size)
        self.decoder = nn.Linear(config.embedding_size, config.vocab_size)
        self.activation = ACT2FN[config.hidden_act]

        # Need a link between the two variables so that the bias is correctly resized with `resize_token_embeddings`
        self.decoder.bias = self.bias

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)

        prediction_scores = hidden_states

        return prediction_scores


class AlbertSOPHead(nn.Module):
    def __init__(self, config):
        super(AlbertSOPHead, self).__init__()

        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, pooled_output):
        dropout_pooled_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_pooled_output)
        return logits


class AlbertForPreTraining(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForPreTraining, self).__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config, self.albert.embeddings.word_embeddings.weight)
        self.sop_classifier = AlbertSOPHead(config)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None, sentence_order_label=None):
        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)

        sequence_output, pooled_output = outputs[:2]

        prediction_scores = self.predictions(sequence_output)
        sop_scores = self.sop_classifier(pooled_output)

        outputs = (prediction_scores, sop_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        if sentence_order_label is not None:
            loss_fct = CrossEntropyLoss()
            sop_loss = loss_fct(sop_scores.view(-1, 2), sentence_order_label.view(-1))
            outputs = (sop_loss,) + outputs

        return outputs  # (sop_loss), (masked_lm_loss), prediction_scores, sop_scores, (hidden_states), (attentions)


class AlbertForMaskedLM(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForMaskedLM, self).__init__(config)

        self.albert = AlbertModel(config)
        self.predictions = AlbertMLMHead(config, self.albert.embeddings.word_embeddings.weight)

        self.apply(self.init_weights)
        self.tie_weights()

    def tie_weights(self):
        self._tie_or_clone_weights(self.predictions.decoder, self.albert.embeddings.word_embeddings)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]
        prediction_scores = self.predictions(sequence_output)

        outputs = (prediction_scores,) + outputs[2:]  # Add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            outputs = (masked_lm_loss,) + outputs

        return outputs  # (masked_lm_loss), prediction_scores, (hidden_states), (attentions)


class AlbertForSequenceClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForSequenceClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            if self.num_labels == 1:
                #  We are doing regression
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), labels.view(-1))
            else:
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForMultipleChoice(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForMultipleChoice, self).__init__(config)

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, 1)

        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        num_choices = input_ids.shape[1] if input_ids is not None else inputs_embeds.shape[1]

        flat_input_ids = input_ids.view(-1, input_ids.size(-1)) if input_ids is not None else None
        flat_attention_mask = attention_mask.view(-1, attention_mask.size(-1)) if attention_mask is not None else None
        flat_token_type_ids = token_type_ids.view(-1, token_type_ids.size(-1)) if token_type_ids is not None else None
        flat_position_ids = position_ids.view(-1, position_ids.size(-1)) if position_ids is not None else None
        flat_inputs_embeds = inputs_embeds.view(-1, inputs_embeds.size(-2), inputs_embeds.size(-1)) if inputs_embeds is not None else None

        outputs = self.albert(input_ids=flat_input_ids,
                              attention_mask=flat_attention_mask,
                              token_type_ids=flat_token_type_ids,
                              position_ids=flat_position_ids,
                              head_mask=head_mask,
                              inputs_embeds=flat_inputs_embeds)

        pooled_output = outputs[1]

        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        reshaped_logits = logits.view(-1, num_choices)

        outputs = (reshaped_logits,) + outputs[2:]  # add hidden states and attention if they are here

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(reshaped_logits, labels)
            outputs = (loss,) + outputs

        return outputs  # (loss), reshaped_logits, (hidden_states), (attentions)


class AlbertForTokenClassification(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForTokenClassification, self).__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.classifier_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        logits = self.classifier(sequence_output)

        outputs = (logits,) + outputs[2:]  # add hidden states and attention if they are here
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            # Only keep active parts of the loss
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)
                active_labels = torch.where(
                    active_loss, labels.view(-1), torch.tensor(loss_fct.ignore_index).type_as(labels)
                )
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions)


class AlbertForQuestionAnswering(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForQuestionAnswering, self).__init__(config)
        self.num_labels = config.num_labels

        self.albert = AlbertModel(config)
        self.qa_outputs = nn.Linear(config.hidden_size, config.num_labels)

        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, start_positions=None, end_positions=None):
        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        outputs = (start_logits, end_logits,) + outputs[2:]
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs  # (loss), start_logits, end_logits, (hidden_states), (attentions)


# Adapting ALBERT for Image Captioning (like BERT version)
class AlbertImgEmbeddings(nn.Module):
    """
    Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(AlbertImgEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.embedding_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.embedding_size)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNormClass(config.embedding_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class AlbertImgModel(AlbertPreTrainedModel):
    r"""
    ALBERT Model for handling both text and image features.
    Adapted from AlbertImgModel for ALBERT architecture.
    """
    def __init__(self, config):
        super(AlbertImgModel, self).__init__(config)

        self.embeddings = AlbertImgEmbeddings(config)
        self.encoder = AlbertTransformer(config)
        self.pooler = nn.Linear(config.hidden_size, config.hidden_size)
        self.pooler_activation = nn.Tanh()

        self.img_dim = config.img_feature_dim #2054 #565
        logger.info('AlbertImgModel Image Dimension: {}'.format(self.img_dim))
        self.img_feature_type = config.img_feature_type
        try:
            self.use_img_layernorm = config.use_img_layernorm
        except:
            self.use_img_layernorm = None

        if config.img_feature_type == 'dis_code':
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.embedding_size, bias=True)
        elif config.img_feature_type == 'dis_code_t': # transpose
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_size, self.config.embedding_size, bias=True)
        elif config.img_feature_type == 'dis_code_scale': # scaled
            self.input_embeddings = nn.Linear(config.code_dim, config.code_size, bias=True)
            self.code_embeddings = nn.Embedding(config.code_voc, config.code_dim, padding_idx=0)
            self.img_embedding = nn.Linear(config.code_dim, self.config.embedding_size, bias=True)
        else:
            self.img_embedding = nn.Linear(self.img_dim, self.config.embedding_size, bias=True)
            self.dropout = nn.Dropout(config.hidden_dropout_prob)
            if self.use_img_layernorm:
                self.LayerNorm = LayerNormClass(config.embedding_size, eps=config.img_layer_norm_eps)

        self.apply(self.init_weights)
        self.model_type = getattr(config, 'model_type', 'albert')

        # re-initialize img_embedding weight
        if getattr(config, 'tie_img_weights', False):
            logger.info("Tie word and img embedding weights...")
            self.img_embedding.weight = self.embeddings.word_embeddings.weight
        else:
            logger.info("Generate different weights for img embeddings...")
            # truncated normal for img_embedding initialization
            self.img_embedding.weight.data.normal_(mean=0.0, std=config.initializer_range)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None,
            position_ids=None, head_mask=None, img_feats=None,
            encoder_history_states=None):

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we're adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # head_mask has shape n_layer x batch x n_heads x N x N
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)  # We can specify head_mask for each layer
            head_mask = head_mask.to(dtype=next(self.parameters()).dtype) # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(input_ids, position_ids=position_ids, token_type_ids=token_type_ids)

        if img_feats is not None:
            if self.img_feature_type == 'dis_code':
                code_emb = self.code_embeddings(img_feats)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_t':
                code_emb = self.code_embeddings(img_feats)
                code_emb = code_emb.permute(0, 2, 1)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.img_feature_type == 'dis_code_scale':
                code_emb = self.code_embeddings(img_feats)
                code_emb = self.input_embeddings(code_emb)
                img_embedding_output = self.img_embedding(code_emb)
            elif self.model_type == 'TIMM_vit':
                img_embedding_output = img_feats
            else:
                if torch._C._get_tracing_state():
                    # Ugly workaround to make this work for ONNX.
                    #  It is also valid for PyTorch bu I keep this path separate to remove once fixed in ONNX
                    img_embedding_output = self.img_embedding(img_feats.squeeze(0)).unsqueeze(0)
                else:
                    img_embedding_output = self.img_embedding(img_feats)
                #logger.info('img_embedding_output: %s' % str(img_embedding_output.shape))
                if self.use_img_layernorm:
                    img_embedding_output = self.LayerNorm(img_embedding_output)

                # add dropout on image embedding
                img_embedding_output = self.dropout(img_embedding_output)

            # concatenate two embeddings
            embedding_output = torch.cat((embedding_output, img_embedding_output), 1)

        encoder_outputs = self.encoder(embedding_output,
                extended_attention_mask, head_mask=head_mask)
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler_activation(self.pooler(sequence_output[:, 0]))

        outputs = (sequence_output, pooled_output,) + encoder_outputs[1:]  # add hidden_states and attentions if they are here
        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)


class AlbertCaptioningHeads(nn.Module):
    def __init__(self, config):
        super(AlbertCaptioningHeads, self).__init__()
        self.predictions = AlbertMLMHead(config, None)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class AlbertCaptioningLoss(nn.Module):
    def __init__(self, config):
        super(AlbertCaptioningLoss, self).__init__()
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


class AlbertIFPredictionHead(nn.Module):
    def __init__(self, config):
        super(AlbertIFPredictionHead, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.transform_act_fn = ACT2FN[config.hidden_act]
        self.LayerNorm = LayerNormClass(config.hidden_size, eps=config.layer_norm_eps)
        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.hidden_size, config.img_feature_dim, bias=False)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class AlbertImgFeatureLoss(nn.Module):
    def __init__(self, config):
        super(AlbertImgFeatureLoss, self).__init__()
        self.drop_worst_ratio = getattr(config, 'drop_worst_ratio', 0)
        self.drop_worst_after = getattr(config, 'drop_worst_after', 0)
        self.iter = 0

    def forward(self, logits, target):
        self.iter += 1
        loss = F.mse_loss(logits, target, reduction='none')
        loss = loss.mean(axis=1)

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
        # cclin
        self.cls_img_feat = AlbertIFPredictionHead(config)
        self.loss_img_feat = AlbertImgFeatureLoss(config)

        self.apply(self.init_weights)
        self.tie_weights()

        self.model_type = getattr(config, 'model_type', 'albert')

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
        inference_mode = kwargs.get('inference_mode', '')
        if inference_mode:
            kwargs.pop('inference_mode')
            if inference_mode == 'prod':
                return self.prod_generate(*args, **kwargs)
            if inference_mode == 'prod_no_hidden':
                return self.prod_no_hidden_generate(*args, **kwargs)
            assert False, 'unknown inference_mode: {}'.format(inference_mode)
        if is_decode:
            return self.generate(*args, **kwargs)
        else:
            return self.encode_forward(*args, **kwargs)

    def encode_forward(self, input_ids, img_feats, attention_mask,
            masked_pos=None, masked_ids=None,
            masked_pos_img=None, masked_token_img=None,
            token_type_ids=None, position_ids=None, head_mask=None,
            is_training=True, encoder_history_states=None):
        
        outputs = self.albert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                position_ids=position_ids, head_mask=head_mask, img_feats=img_feats,
                encoder_history_states=encoder_history_states)
        if is_training:
            sequence_output = outputs[0][:, :masked_pos.shape[-1], :]
            sequence_output_masked = sequence_output[masked_pos==1, :]
            class_logits = self.cls(sequence_output_masked)
            masked_ids = masked_ids[masked_ids != -1]
            masked_loss = self.loss(class_logits.float(), masked_ids)
            outputs = (masked_loss, class_logits, ) + outputs
        else:
            outputs = outputs
        return outputs

    def prepare_inputs_for_generation(self, input_ids, past=None, **kwargs):
        mask_token_id = kwargs.get('mask_token_id', None)
        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        effective_batch_size = input_ids.shape[0]
        #  add a dummy token
        if kwargs.get('attn_mask', None) is None:
            attention_mask = torch.ones(effective_batch_size, input_ids.shape[1] + self.max_img_seq_length, dtype=torch.long, device=input_ids.device)
        else:
            attention_mask = kwargs['attn_mask']

        # NOTE: make sure attention_mask is a expanded tensor and not a nested list, so that it can be put to cuda device.
        # import pdb; pdb.set_trace()
        if kwargs.get('do_sample', None):
            img_feats = kwargs['img_feats'].repeat_interleave(kwargs['num_beams'], dim=0)
        else:
            img_feats = kwargs['img_feats']

        return {
            "input_ids": input_ids,
            "img_feats": img_feats,
            "past_key_values": past,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
            "is_decode": True,
        }

    # Dummy implementations for compatibility
    def generate(self, *args, **kwargs):
        # Implementation needed for generation
        pass

    def prod_generate(self, *args, **kwargs):
        # Implementation needed for production generation
        pass

    def prod_no_hidden_generate(self, *args, **kwargs):
        # Implementation needed for production generation without hidden states
        pass


# Additional ALBERT classes for completeness
class AlbertForNextSentencePrediction(AlbertPreTrainedModel):
    def __init__(self, config):
        super(AlbertForNextSentencePrediction, self).__init__(config)
        self.albert = AlbertModel(config)
        self.sop_classifier = AlbertSOPHead(config)
        self.apply(self.init_weights)

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, position_ids=None, head_mask=None,
                inputs_embeds=None, labels=None):
        outputs = self.albert(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)
        pooled_output = outputs[1]
        seq_relationship_scores = self.sop_classifier(pooled_output)
        outputs = (seq_relationship_scores,) + outputs[2:]
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            next_sentence_loss = loss_fct(seq_relationship_scores.view(-1, 2), labels.view(-1))
            outputs = (next_sentence_loss,) + outputs
        return outputs


# Placeholders for other ALBERT classes
AlbertImgForPreTraining = AlbertForPreTraining
AlbertForVLGrounding = AlbertForSequenceClassification  
AlbertImgForGroundedPreTraining = AlbertForPreTraining
