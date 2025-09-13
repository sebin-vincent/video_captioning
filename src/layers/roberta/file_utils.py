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
"""PyTorch RoBERTa model."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import os
import json
import logging
from io import open

from .tokenization_utils import PreTrainedTokenizer

logger = logging.getLogger(__name__)

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'roberta-base': 512,
    'roberta-large': 512,
    'roberta-large-mnli': 512,
    'distilroberta-base': 512,
    'roberta-base-openai-detector': 512,
    'roberta-large-openai-detector': 512,
}

PYTORCH_PRETRAINED_ROBERTA_CACHE = os.path.join(os.path.expanduser("~"), '.pytorch_pretrained_roberta')

def cached_path(url_or_filename, cache_dir=None):
    """
    Given something that might be a URL (or might be a local path),
    determine which. If it's a URL, download the file and cache it, and
    return the path to the cached file. If it's already a local path,
    make sure the file exists and then return the path.
    """
    if cache_dir is None:
        cache_dir = PYTORCH_PRETRAINED_ROBERTA_CACHE
    if os.path.exists(cache_dir) and os.path.isdir(cache_dir):
        return url_or_filename
    else:
        raise ValueError("cache_dir should be a directory")
