# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
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
"""Tokenization classes for ALBERT."""

import os
import unicodedata
from shutil import copyfile

import sentencepiece as spm

from .tokenization_utils import PreTrainedTokenizer
from src.utils.comm import is_main_process
import logging

logger = logging.getLogger(__name__)
if not is_main_process():
    logger.disabled = True

VOCAB_FILES_NAMES = {"vocab_file": "spiece.model", "tokenizer_config_file": "tokenizer_config.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/spiece.model",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/spiece.model",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/spiece.model",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/spiece.model",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/spiece.model",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/spiece.model",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/spiece.model",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/spiece.model",
    },
    "tokenizer_config_file": {
        "albert-base-v1": "https://huggingface.co/albert-base-v1/resolve/main/tokenizer_config.json",
        "albert-large-v1": "https://huggingface.co/albert-large-v1/resolve/main/tokenizer_config.json",
        "albert-xlarge-v1": "https://huggingface.co/albert-xlarge-v1/resolve/main/tokenizer_config.json",
        "albert-xxlarge-v1": "https://huggingface.co/albert-xxlarge-v1/resolve/main/tokenizer_config.json",
        "albert-base-v2": "https://huggingface.co/albert-base-v2/resolve/main/tokenizer_config.json",
        "albert-large-v2": "https://huggingface.co/albert-large-v2/resolve/main/tokenizer_config.json",
        "albert-xlarge-v2": "https://huggingface.co/albert-xlarge-v2/resolve/main/tokenizer_config.json",
        "albert-xxlarge-v2": "https://huggingface.co/albert-xxlarge-v2/resolve/main/tokenizer_config.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "albert-base-v1": 512,
    "albert-large-v1": 512,
    "albert-xlarge-v1": 512,
    "albert-xxlarge-v1": 512,
    "albert-base-v2": 512,
    "albert-large-v2": 512,
    "albert-xlarge-v2": 512,
    "albert-xxlarge-v2": 512,
}


class AlbertTokenizer(PreTrainedTokenizer):
    """
    Constructs an ALBERT tokenizer.

    This tokenizer inherits from [`PreTrainedTokenizer`] which contains most of the main methods. Users should refer to
    this superclass for more information regarding those methods.

    Args:
        vocab_file (`str`):
            Path to the vocabulary file.
        do_lower_case (`bool`, *optional*, defaults to `True`):
            Whether or not to lowercase the input when tokenizing.
        remove_accent (`bool`, *optional*):
            Whether or not to remove accents when tokenizing.
        keep_accents (`bool`, *optional*):
            Whether or not to keep accents when tokenizing.
        unk_token (`str`, *optional*, defaults to `"<unk>"`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        sep_token (`str`, *optional*, defaults to `"[SEP]"`):
            The separator token, which is used when building text pairs or assembling sequences with special tokens as
            in `_build_inputs_with_special_tokens`.
        pad_token (`str`, *optional*, defaults to `"<pad>"`):
            The padding token, which is used to pad sequences up to the model's maximum input length.
        cls_token (`str`, *optional*, defaults to `"[CLS]"`):
            The classifier token which is used when doing sequence classification (classification of the whole sequence
            instead of per-token classification). It is the first token of the sequence when built with special
            tokens.
        mask_token (`str`, *optional*, defaults to `"[MASK]"`):
            The token used to mask a token during training.
        sp_model_kwargs (`dict`, *optional*):
            Will be passed to the `SentencePieceProcessor.__init__()` method. The [Python wrapper for
            SentencePiece](https://github.com/google/sentencepiece/tree/master/python) can be used, among other
            things, to set:

            - `enable_sampling`: Enable subword regularization.
            - `nbest_size`: Sampling parameters for unigram. Invalid for BPE-Dropout.

              - `nbest_size = {0,1}`: No sampling is performed.
              - `nbest_size > 1`: samples from the nbest_size results.
              - `nbest_size < 0`: assuming that nbest_size is stored in meta proto, samples from the nbest_size results
                in the proto.

            - `alpha`: Smoothing parameter for unigram sampling, and dropout probability of merging operations for
              BPE-dropout.

    Attributes:
        sp_model (`SentencePieceProcessor`):
            The *SentencePiece* processor.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "token_type_ids", "attention_mask"]

    def __init__(
        self,
        vocab_file,
        do_lower_case=True,
        remove_accent=True,
        keep_accents=False,
        unk_token="<unk>",
        sep_token="[SEP]",
        pad_token="<pad>",
        cls_token="[CLS]",
        mask_token="[MASK]",
        sp_model_kwargs=None,
        **kwargs
    ):
        # Masking is done by the model, not the tokenizer.
        super().__init__(
            do_lower_case=do_lower_case,
            remove_accent=remove_accent,
            keep_accents=keep_accents,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            sp_model_kwargs=sp_model_kwargs,
            **kwargs,
        )
        self.vocab_file = vocab_file
        self.do_lower_case = do_lower_case
        self.remove_accent = remove_accent
        self.keep_accents = keep_accents
        self.sp_model_kwargs = {} if sp_model_kwargs is None else sp_model_kwargs

        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(vocab_file)

    @property
    def vocab_size(self):
        return len(self.sp_model)

    def get_vocab(self):
        vocab = {self.convert_ids_to_tokens(i): i for i in range(self.vocab_size)}
        vocab.update(self.added_tokens_encoder)
        return vocab

    def __getstate__(self):
        state = self.__dict__.copy()
        state["sp_model"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.sp_model = spm.SentencePieceProcessor(**self.sp_model_kwargs)
        self.sp_model.Load(self.vocab_file)

    def _tokenize(self, text):
        """Take as input a string and return a list of strings (tokens) for words."""
        text = self._normalize(text)
        return self.sp_model.encode_as_pieces(text)

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.sp_model.piece_to_id(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.sp_model.id_to_piece(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (string) in a single string."""
        return self.sp_model.decode_pieces(tokens)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks by concatenating and
        adding special tokens. An ALBERT sequence has the following format:

        - a single sequence: `[CLS] X [SEP]`
        - a pair of sequences: `[CLS] A [SEP] B [SEP]`

        Args:
            token_ids_0 (`List[int]`):
                List of IDs to which the special tokens will be added.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [input IDs](../glossary#input-ids) with the appropriate special tokens.
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]
        if token_ids_1 is None:
            return cls + token_ids_0 + sep
        return cls + token_ids_0 + sep + token_ids_1 + sep

    def get_special_tokens_mask(
        self, token_ids_0, token_ids_1=None, already_has_special_tokens=False
    ):
        """
        Retrieve sequence ids from a token list that has no special tokens added. This method is called when adding
        special tokens using the tokenizer `prepare_for_model` method.

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.
            already_has_special_tokens (`bool`, *optional*, defaults to `False`):
                Whether or not the token list is already formatted with special tokens for the model.

        Returns:
            `List[int]`: A list of integers in the range [0, 1]: 1 for a special token, 0 for a sequence token.
        """
        if already_has_special_tokens:
            if token_ids_1 is not None:
                raise ValueError(
                    "You should not supply a second sequence if the provided sequence of "
                    "ids is already formatted with special tokens."
                )
            return super().get_special_tokens_mask(
                token_ids_0=token_ids_0, token_ids_1=token_ids_1, already_has_special_tokens=True
            )

        if token_ids_1 is None:
            return [1] + ([0] * len(token_ids_0)) + [1]
        return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]

    def create_token_type_ids_from_sequences(
        self, token_ids_0, token_ids_1=None
    ):
        """
        Create a mask from the two sequences passed to be used in a sequence-pair classification task. An ALBERT
        sequence pair mask has the following format:

        ```
        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | C |           A           | S |           B           | S |
        | L |                       | E |                       | E |
        | S |                       | P |                       | P |
        ```

        If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

        Args:
            token_ids_0 (`List[int]`):
                List of IDs.
            token_ids_1 (`List[int]`, *optional*):
                Optional second list of IDs for sequence pairs.

        Returns:
            `List[int]`: List of [token type IDs](../glossary#token-type-ids) according to the given sequence(s).
        """
        sep = [self.sep_token_id]
        cls = [self.cls_token_id]

        if token_ids_1 is None:
            return len(cls + token_ids_0 + sep) * [0]
        return len(cls + token_ids_0 + sep) * [0] + len(token_ids_1 + sep) * [1]

    def _normalize(self, text):
        """Normalize a string."""
        if self.do_lower_case:
            text = text.lower()
        if self.remove_accent:
            text = self._run_strip_accents(text)
        return text

    def _run_strip_accents(self, text):
        """Strips accents from a piece of text."""
        text = unicodedata.normalize("NFD", text)
        output = []
        for char in text:
            cat = unicodedata.category(char)
            if cat == "Mn":
                continue
            output.append(char)
        return "".join(output)

    def save_vocabulary(self, save_directory, filename_prefix=None):
        """
        Save the vocabulary and special tokens file to a directory.

        Args:
            save_directory (`str`):
                The directory in which to save the vocabulary.

        Returns:
            `Tuple(str)`: Paths to the saved files.
        """
        if not os.path.isdir(save_directory):
            logger.error(f"Vocabulary path ({save_directory}) should be a directory")
            return
        out_vocab_file = os.path.join(
            save_directory, (filename_prefix + "-" if filename_prefix else "") + VOCAB_FILES_NAMES["vocab_file"]
        )

        if os.path.abspath(self.vocab_file) != os.path.abspath(out_vocab_file):
            copyfile(self.vocab_file, out_vocab_file)

        return (out_vocab_file,)
