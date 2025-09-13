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

logger = logging.getLogger(__name__)

def clean_up_tokenization(out_string):
    """Clean up a list of simple English tokenization artifacts like spaces before punctuations and abbreviated forms.
    """
    out_string = (out_string.replace(' .', '.').replace(' ?', '?').replace(' !', '!')
                .replace(' ,', ',').replace(" ' ", "'").replace(" n't", "n't")
                .replace(" 'm", "'m").replace(" 's", "'s").replace(" 've", "'ve")
                .replace(" 're", "'re"))
    return out_string

class PreTrainedTokenizer(object):
    """ An abstract class to handle weights initialization and
        a simple interface for downloading and loading pretrained models.
    """
    def __init__(self, **kwargs):
        pass
