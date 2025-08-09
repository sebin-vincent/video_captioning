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

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import copy
import json
import os
from io import open

import six
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from torch.nn import functional as F

from .file_utils import cached_path

import logging
from src.utils.comm import is_main_process
logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
if not is_main_process():
    logger.disabled = True

CONFIG_NAME = "config.json"
WEIGHTS_NAME = "pytorch_model.bin"
TF_WEIGHTS_NAME = 'model.ckpt'


try:
    from torch.nn import Identity
except ImportError:
    # Older PyTorch compatibility
    class Identity(nn.Module):
        r"""A placeholder identity operator that is argument-insensitive.
        """
        def __init__(self, *args, **kwargs):
            super(Identity, self).__init__()

        def forward(self, input):
            return input


if not six.PY2:
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            fn.__doc__ = ''.join(docstr) + fn.__doc__
            return fn
        return docstring_decorator
else:
    # Not possible to update class docstrings on python2
    def add_start_docstrings(*docstr):
        def docstring_decorator(fn):
            return fn
        return docstring_decorator


class PretrainedConfig(object):
    """ Base class for all configuration classes.
        Handle a few common parameters and methods for loading/downloading/saving configurations.
    """
    pretrained_config_archive_map = {}

    def __init__(self, **kwargs):
        self.finetuning_task = kwargs.pop('finetuning_task', None)
        self.num_labels = kwargs.pop('num_labels', 2)
        self.output_attentions = kwargs.pop('output_attentions', False)
        self.output_hidden_states = kwargs.pop('output_hidden_states', False)
        self.torchscript = kwargs.pop('torchscript', False)

    def save_pretrained(self, save_directory):
        """ Save a configuration object to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # If we save using the predefined names, we can load using `from_pretrained`
        output_config_file = os.path.join(save_directory, CONFIG_NAME)

        self.to_json_file(output_config_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        r""" Instantiate a PretrainedConfig from a pre-trained model configuration.

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model configuration to load from cache
                    or download and cache if not already stored in cache (e.g. 'albert-base-v1').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a saved configuration `file`.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **return_unused_kwargs**: (`optional`) bool:
                - If False, then this function returns just the final configuration object.
                - If True, then this functions returns a tuple `(config, unused_kwargs)` where `unused_kwargs`
                is a dictionary consisting of the key/value pairs whose keys are not configuration attributes:
                ie the part of kwargs which has not been used to update `config` and is otherwise ignored.
            **kwargs**: (`optional`) dict:
                Dictionary of key/value pairs with which to update the configuration object after loading.
                - The values in kwargs of any keys which are configuration attributes will be used
                to override the loaded values.
                - Behavior concerning key/value pairs whose keys are *not* configuration attributes is controlled
                by the `return_unused_kwargs` keyword parameter.

        Examples::

            >>> config = AlbertConfig.from_pretrained('albert-base-v1')    # Download configuration from S3 and cache.
            >>> config = AlbertConfig.from_pretrained('./test/saved_model/')  # E.g. config (or model) was saved using `save_pretrained('./test/saved_model/')`
            >>> config = AlbertConfig.from_pretrained('./test/saved_model/my_configuration.json')
            >>> config = AlbertConfig.from_pretrained('albert-base-v1', output_attention=True, foo=False)
            >>> assert config.output_attention == True
            >>> config, unused_kwargs = AlbertConfig.from_pretrained('albert-base-v1', output_attention=True,
            >>>                                                    foo=False, return_unused_kwargs=True)
            >>> assert config.output_attention == True
            >>> assert unused_kwargs == {'foo': False}

        """
        cache_dir = kwargs.pop('cache_dir', None)
        return_unused_kwargs = kwargs.pop('return_unused_kwargs', False)

        if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
            config_file = cls.pretrained_config_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            config_file = os.path.join(pretrained_model_name_or_path, CONFIG_NAME)
        else:
            config_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_config_file = cached_path(config_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_config_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained model configuration file.".format(
                        config_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_config_archive_map.keys()),
                        config_file))
            return None
        if resolved_config_file == config_file:
            logger.info("loading configuration file {}".format(config_file))
        else:
            logger.info("loading configuration file {} from cache at {}".format(
                config_file, resolved_config_file))

        # Load config
        config = cls.from_json_file(resolved_config_file)

        # Update config with kwargs if needed
        to_remove = []
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                to_remove.append(key)
        # add img_layer_norm_eps, use_img_layernorm
        if "img_layer_norm_eps" in kwargs:
            setattr(config, "img_layer_norm_eps", kwargs["img_layer_norm_eps"])
            to_remove.append("img_layer_norm_eps")
        if "use_img_layernorm" in kwargs:
            setattr(config, "use_img_layernorm", kwargs["use_img_layernorm"])
            to_remove.append("use_img_layernorm")
        for key in to_remove:
            kwargs.pop(key, None)

        logger.info("Model config %s", config)
        if return_unused_kwargs:
            return config, kwargs
        else:
            return config

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `Config` from a Python dictionary of parameters."""
        config = cls(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `AlbertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"

    def to_json_file(self, json_file_path):
        """ Save this instance to a json file."""
        with open(json_file_path, "w", encoding='utf-8') as writer:
            writer.write(self.to_json_string())


class PreTrainedModel(nn.Module):
    """ Base class for all models. Handle loading/storing model config and
        a simple interface for dowloading and loading pretrained models.
    """
    config_class = PretrainedConfig
    pretrained_model_archive_map = {}
    load_tf_weights = lambda model, config, path: None
    base_model_prefix = ""
    input_embeddings = None

    def __init__(self, config, *inputs, **kwargs):
        super(PreTrainedModel, self).__init__()
        if not isinstance(config, PretrainedConfig):
            raise ValueError(
                "Parameter config in `{}(config)` should be an instance of class `PretrainedConfig`. "
                "To create a model from a pretrained model use "
                "`model = {}.from_pretrained(PRETRAINED_MODEL_NAME)`".format(
                    self.__class__.__name__, self.__class__.__name__
                ))
        # Save config in model
        self.config = config

    def _get_resized_embeddings(self, old_embeddings, new_num_tokens=None):
        """ Build a resized Embedding Module from a provided token Embedding Module.
            Increasing the size will add newly initialized vectors at the end
            Reducing the size will remove vectors from the end

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: return the provided token Embedding Module.
        Return: ``torch.nn.Embeddings``
            Pointer to the resized Embedding Module or the old Embedding Module if new_num_tokens is None
        """
        if new_num_tokens is None:
            return old_embeddings

        old_num_tokens, old_embedding_dim = old_embeddings.weight.size()
        if old_num_tokens == new_num_tokens:
            return old_embeddings

        # Build new embeddings
        new_embeddings = nn.Embedding(new_num_tokens, old_embedding_dim)
        new_embeddings.to(old_embeddings.weight.device)

        # initialize all new embeddings (in particular added tokens)
        self.init_weights(new_embeddings)

        # Copy word embeddings from the previous weights
        num_tokens_to_copy = min(old_num_tokens, new_num_tokens)
        new_embeddings.weight.data[:num_tokens_to_copy, :] = old_embeddings.weight.data[:num_tokens_to_copy, :]

        return new_embeddings

    def _tie_or_clone_weights(self, first_module, second_module):
        """ Tie or clone module weights depending of weither we are using TorchScript or not
        """
        if self.config.torchscript:
            first_module.weight = nn.Parameter(second_module.weight.clone())
        else:
            first_module.weight = second_module.weight

    def resize_token_embeddings(self, new_num_tokens=None):
        """ Resize input token embeddings matrix of the model if new_num_tokens != config.vocab_size.
            Take care of tying weights embeddings afterwards if the model class has a `tie_weights()` method.

        Args:
            new_num_tokens: (`optional`) int
                New number of tokens in the embedding matrix.
                Increasing the size will add newly initialized vectors at the end
                Reducing the size will remove vectors from the end
                If not provided or None: does nothing and just returns a pointer to the input tokens Embedding Module of the model.

        Return: ``torch.nn.Embeddings``
            Pointer to the input tokens Embedding Module of the model
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        model_embeds = base_model._resize_token_embeddings(new_num_tokens)
        if new_num_tokens is None:
            return model_embeds

        # Update base model and current model config
        self.config.vocab_size = new_num_tokens
        base_model.vocab_size = new_num_tokens

        # Tie weights again if needed
        if hasattr(self, 'tie_weights'):
            self.tie_weights()

        return model_embeds

    def prune_heads(self, heads_to_prune):
        """ Prunes heads of the base model.
            Args:
                heads_to_prune: dict of {layer_num (int): list of heads to prune in this layer (list of int)}
        """
        base_model = getattr(self, self.base_model_prefix, self)  # get the base model if needed
        base_model._prune_heads(heads_to_prune)

    def save_pretrained(self, save_directory):
        """ Save a model with its configuration file to a directory, so that it
            can be re-loaded using the `from_pretrained(save_directory)` class method.
        """
        assert os.path.isdir(save_directory), "Saving path should be a directory where the model and configuration can be saved"

        # Only save the model it-self if we are using distributed training
        model_to_save = self.module if hasattr(self, 'module') else self

        # Save configuration file
        model_to_save.config.save_pretrained(save_directory)

        # If we save using the predefined names, we can load using `from_pretrained`
        output_model_file = os.path.join(save_directory, WEIGHTS_NAME)

        torch.save(model_to_save.state_dict(), output_model_file)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        r"""Instantiate a pretrained pytorch model from a pre-trained model configuration.

            The model is set in evaluation mode by default using `model.eval()` (Dropout modules are desactivated)
            To train the model, you should first set it back in training mode with `model.train()`

        Params:
            **pretrained_model_name_or_path**: either:
                - a string with the `shortcut name` of a pre-trained model to load from cache
                    or download and cache if not already stored in cache (e.g. 'albert-base-v1').
                - a path to a `directory` containing a configuration file saved
                    using the `save_pretrained(save_directory)` method.
                - a path or url to a tensorflow index checkpoint `file` (e.g. `./tf_model/model.ckpt.index`).
                    In this case, ``from_tf`` should be set to True and a configuration object should be
                    provided as `config` argument. This loading option is slower than converting the TensorFlow
                    checkpoint in a PyTorch model using the provided conversion scripts and loading
                    the PyTorch model afterwards.
            **model_args**: (`optional`) Sequence:
                All remaning positional arguments will be passed to the underlying model's __init__ function
            **config**: an optional configuration for the model to use instead of an automatically loaded configuation.
                Configuration can be automatically loaded when:
                - the model is a model provided by the library (loaded with a `shortcut name` of a pre-trained model), or
                - the model was saved using the `save_pretrained(save_directory)` (loaded by suppling the save directory).
            **state_dict**: an optional state dictionnary for the model to use instead of a state dictionary loaded
                from saved weights file.
                This option can be used if you want to create a model from a pretrained configuraton but load your own weights.
                In this case though, you should check if using `save_pretrained(dir)` and `from_pretrained(save_directory)` is not
                a simpler option.
            **cache_dir**: (`optional`) string:
                Path to a directory in which a downloaded pre-trained model
                configuration should be cached if the standard cache should not be used.
            **output_loading_info**: (`optional`) boolean:
                Set to ``True`` to also return a dictionnary containing missing keys, unexpected keys and error messages.
            **kwargs**: (`optional`) dict:
                Dictionary of key, values to update the configuration object after loading.
                Can be used to override selected configuration parameters. E.g. ``output_attention=True``.

               - If a configuration is provided with `config`, **kwargs will be directly passed
                 to the underlying model's __init__ method.
               - If a configuration is not provided, **kwargs will be first passed to the pretrained
                 model configuration class loading function (`PretrainedConfig.from_pretrained`).
                 Each key of **kwargs that corresponds to a configuration attribute
                 will be used to override said attribute with the supplied **kwargs value.
                 Remaining keys that do not correspond to any configuration attribute will
                 be passed to the underlying model's __init__ function.

        Examples::

            >>> model = AlbertModel.from_pretrained('albert-base-v1')    # Download model and configuration from S3 and cache.
            >>> model = AlbertModel.from_pretrained('./test/saved_model/')  # E.g. model was saved using `save_pretrained('./test/saved_model/')`
            >>> model = AlbertModel.from_pretrained('albert-base-v1', output_attention=True)  # Update configuration during loading
            >>> assert model.config.output_attention == True
            >>> # Loading from a TF checkpoint file instead of a PyTorch model (slower)
            >>> config = AlbertConfig.from_json_file('./tf_model/my_tf_model_config.json')
            >>> model = AlbertModel.from_pretrained('./tf_model/my_tf_checkpoint.ckpt.index', from_tf=True, config=config)

        """
        config = kwargs.pop('config', None)
        state_dict = kwargs.pop('state_dict', None)
        cache_dir = kwargs.pop('cache_dir', None)
        from_tf = kwargs.pop('from_tf', False)
        output_loading_info = kwargs.pop('output_loading_info', False)

        # Load config
        if config is None:
            config, model_kwargs = cls.config_class.from_pretrained(
                pretrained_model_name_or_path, *model_args,
                cache_dir=cache_dir, return_unused_kwargs=True,
                **kwargs
            )
        else:
            model_kwargs = kwargs

        # Load model
        if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
            archive_file = cls.pretrained_model_archive_map[pretrained_model_name_or_path]
        elif os.path.isdir(pretrained_model_name_or_path):
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = os.path.join(pretrained_model_name_or_path, TF_WEIGHTS_NAME + ".index")
            else:
                archive_file = os.path.join(pretrained_model_name_or_path, WEIGHTS_NAME)
        else:
            if from_tf:
                # Directly load from a TensorFlow checkpoint
                archive_file = pretrained_model_name_or_path + ".index"
            else:
                archive_file = pretrained_model_name_or_path
        # redirect to the cache, if necessary
        try:
            resolved_archive_file = cached_path(archive_file, cache_dir=cache_dir)
        except EnvironmentError:
            if pretrained_model_name_or_path in cls.pretrained_model_archive_map:
                logger.error(
                    "Couldn't reach server at '{}' to download pretrained weights.".format(
                        archive_file))
            else:
                logger.error(
                    "Model name '{}' was not found in model name list ({}). "
                    "We assumed '{}' was a path or url but couldn't find any file "
                    "associated to this path or url.".format(
                        pretrained_model_name_or_path,
                        ', '.join(cls.pretrained_model_archive_map.keys()),
                        archive_file))
            return None
        if resolved_archive_file == archive_file:
            logger.info("loading weights file {}".format(archive_file))
        else:
            logger.info("loading weights file {} from cache at {}".format(
                archive_file, resolved_archive_file))

        # Instantiate model.
        model = cls(config, *model_args, **model_kwargs)

        if state_dict is None and not from_tf:
            state_dict = torch.load(resolved_archive_file, map_location='cpu')

        if from_tf:
            # Directly load from a TensorFlow checkpoint
            return cls.load_tf_weights(model, config, resolved_archive_file[:-6])  # Remove the '.index'

        # Convert old format to new format if needed from a PyTorch state_dict
        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if 'gamma' in key:
                new_key = key.replace('gamma', 'weight')
            if 'beta' in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        # Load from a PyTorch state_dict
        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        # copy state_dict so _load_from_state_dict can modify it
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(
                state_dict, prefix, local_metadata, True, missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')

        # Make sure we are able to load base models as well as derived models (with heads)
        start_prefix = ''
        model_to_load = model
        if not hasattr(model, cls.base_model_prefix) and any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            start_prefix = cls.base_model_prefix + '.'
        if hasattr(model, cls.base_model_prefix) and not any(s.startswith(cls.base_model_prefix) for s in state_dict.keys()):
            model_to_load = getattr(model, cls.base_model_prefix)

        load(model_to_load, prefix=start_prefix)
        if len(missing_keys) > 0:
            logger.info("Weights of {} not initialized from pretrained model: {}".format(
                model.__class__.__name__, missing_keys))
        if len(unexpected_keys) > 0:
            logger.info("Weights from pretrained model not used in {}: {}".format(
                model.__class__.__name__, unexpected_keys))
        if len(error_msgs) == 2 and "size mismatch for cls.seq_relationship.weight" in error_msgs[0]:
            logger.info('Error(s) in loading state_dict for {}:\n\t{}'.format(
                model.__class__.__name__, "\n\t".join(error_msgs)))
        elif len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))

        if hasattr(model, 'tie_weights'):
            model.tie_weights()  # make sure word embedding weights are still tied

        # Set model in evaluation mode to desactivate DropOut modules by default
        model.eval()

        if output_loading_info:
            loading_info = {"missing_keys": missing_keys, "unexpected_keys": unexpected_keys, "error_msgs": error_msgs}
            return model, loading_info

        return model

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids}

    def _do_output_past(self, outputs):
        has_output_past = hasattr(self.config, "output_past") and self.config.output_past
        has_mem_len = hasattr(self.config, "mem_len") and self.config.mem_len

        if has_output_past and not has_mem_len and len(outputs) > 1:
            return True
        elif has_mem_len and self.config.mem_len > 0 and len(outputs) > 1:
            return True

        return False


class Conv1D(nn.Module):
    def __init__(self, nf, nx):
        """ Conv1D layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2)
            Basically works like a Linear layer but the weights are transposed
        """
        super(Conv1D, self).__init__()
        self.nf = nf
        w = torch.empty(nx, nf)
        nn.init.normal_(w, std=0.02)
        self.weight = nn.Parameter(w)
        self.bias = nn.Parameter(torch.zeros(nf))

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(*size_out)
        return x


def prune_linear_layer(layer, index, dim=0):
    """ Prune a linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if layer.bias is not None:
        if dim == 1:
            b = layer.bias.clone().detach()
        else:
            b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = nn.Linear(new_size[1], new_size[0], bias=layer.bias is not None).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    if layer.bias is not None:
        new_layer.bias.requires_grad = False
        new_layer.bias.copy_(b.contiguous())
        new_layer.bias.requires_grad = True
    return new_layer


def prune_conv1d_layer(layer, index, dim=1):
    """ Prune a Conv1D layer (a model parameters) to keep only entries in index.
        A Conv1D work as a Linear layer (see e.g. BERT) but the weights are transposed.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    index = index.to(layer.weight.device)
    W = layer.weight.index_select(dim, index).clone().detach()
    if dim == 0:
        b = layer.bias.clone().detach()
    else:
        b = layer.bias[index].clone().detach()
    new_size = list(layer.weight.size())
    new_size[dim] = len(index)
    new_layer = Conv1D(new_size[1], new_size[0]).to(layer.weight.device)
    new_layer.weight.requires_grad = False
    new_layer.weight.copy_(W.contiguous())
    new_layer.weight.requires_grad = True
    new_layer.bias.requires_grad = False
    new_layer.bias.copy_(b.contiguous())
    new_layer.bias.requires_grad = True
    return new_layer


def prune_layer(layer, index, dim=None):
    """ Prune a Conv1D or nn.Linear layer (a model parameters) to keep only entries in index.
        Return the pruned layer as a new layer with requires_grad=True.
        Used to remove heads.
    """
    if isinstance(layer, nn.Linear):
        return prune_linear_layer(layer, index, dim=0 if dim is None else dim)
    elif isinstance(layer, Conv1D):
        return prune_conv1d_layer(layer, index, dim=1 if dim is None else dim)
    else:
        raise ValueError("Can't prune layer of class {}".format(layer.__class__))
