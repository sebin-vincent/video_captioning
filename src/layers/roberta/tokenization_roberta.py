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
"""Tokenization classes for RoBERTa, similar to BERT."""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import collections
import os
import unicodedata
from io import open

from .tokenization_utils import PreTrainedTokenizer, clean_up_tokenization

PRETRAINED_VOCAB_FILES_MAP = {
    'roberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-vocab.json",
    'roberta-large': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-vocab.json",
    'roberta-large-mnli': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-vocab.json",
    'distilroberta-base': "https://s3.amazonaws.com/models.huggingface.co/bert/distilroberta-base-vocab.json",
    'roberta-base-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-openai-detector-vocab.json",
    'roberta-large-openai-detector': "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-openai-detector-vocab.json",
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    'roberta-base': 512,
    'roberta-large': 512,
    'roberta-large-mnli': 512,
    'distilroberta-base': 512,
    'roberta-base-openai-detector': 512,
    'roberta-large-openai-detector': 512,
}

def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

def whitespace_tokenize(text):
    """Runs basic whitespace cleaning and splitting on a piece of text."""
    text = text.strip()
    if not text:
        return []
    tokens = text.split()
    return tokens

class BasicTokenizer(object):
    """Runs basic tokenization (punctuation splitting, lower casing, etc.)."""

    def __init__(self, do_lower_case=True, never_split=None):
        """Constructs a BasicTokenizer.
        Args:
            do_lower_case: Whether to lower case the input.
            never_split: (`optional`) list of str
                Kept as-is and never split.
        """
        if never_split is None:
            never_split = []
        self.do_lower_case = do_lower_case
        self.never_split = set(never_split)

    def tokenize(self, text):
        """Tokenizes a piece of text."""
        text = self._clean_text(text)
        text = self._tokenize_chinese_chars(text)
        orig_tokens = whitespace_tokenize(text)
        split_tokens = []
        for token in orig_tokens:
            if self.do_lower_case and token not in self.never_split:
                token = token.lower()
                token = self._run_strip_accents(token)
            split_tokens.extend(self._run_split_on_punc(token))

        output_tokens = whitespace_tokenize(" ".join(split_tokens))
        return output_tokens

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

    def _run_split_on_punc(self, text):
        """Splits punctuation on a piece of text."""
        if text in self.never_split:
            return [text]
        chars = list(text)
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if _is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word:
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1

        return ["".join(x) for x in output]

    def _tokenize_chinese_chars(self, text):
        """Adds whitespace around any CJK character."""
        output = []
        for char in text:
            cp = ord(char)
            if self._is_chinese_char(cp):
                output.extend([" ", char, " "])
            else:
                output.append(char)
        return "".join(output)

    def _is_chinese_char(self, cp):
        """Checks whether CP is the codepoint of a CJK character."""
        if ((cp >= 0x4E00 and cp <= 0x9FFF) or  #
                (cp >= 0x3400 and cp <= 0x4DBF) or  #
                (cp >= 0x20000 and cp <= 0x2A6DF) or  #
                (cp >= 0x2A700 and cp <= 0x2B73F) or  #
                (cp >= 0x2B740 and cp <= 0x2B81F) or  #
                (cp >= 0x2B820 and cp <= 0x2CEAF) or
                (cp >= 0xF900 and cp <= 0xFAFF) or  #
                (cp >= 0x2F800 and cp <= 0x2FA1F)):  #
            return True

        return False

    def _clean_text(self, text):
        """Performs invalid character removal and whitespace cleanup on text."""
        output = []
        for char in text:
            cp = ord(char)
            if cp == 0 or cp == 0xfffd or _is_control(char):
                continue
            if _is_whitespace(char):
                output.append(" ")
            else:
                output.append(char)
        return "".join(output)

class WordpieceTokenizer(object):
    """Runs WordPiece tokenization."""

    def __init__(self, vocab, unk_token="<unk>", max_input_chars_per_word=200):
        self.vocab = vocab
        self.unk_token = unk_token
        self.max_input_chars_per_word = max_input_chars_per_word

    def tokenize(self, text):
        """Tokenizes a piece of text into its word pieces.
        This uses a greedy longest-match-first algorithm to perform tokenization
        using the given vocabulary.
        For example:
            input = "unaffable"
            output = ["un", "##aff", "##able"]
        Args:
            text: A single token or whitespace separated tokens. This should have
                already been passed through `BasicTokenizer.
        Returns:
            A list of wordpiece tokens.
        """

        output_tokens = []
        for token in whitespace_tokenize(text):
            chars = list(token)
            if len(chars) > self.max_input_chars_per_word:
                output_tokens.append(self.unk_token)
                continue

            is_bad = False
            start = 0
            sub_tokens = []

            while start < len(chars):
                end = len(chars)
                cur_subtoken = None
                while start < end:
                    substr = "".join(chars[start:end])
                    if start > 0:
                        substr = "##" + substr
                    if substr in self.vocab:
                        cur_subtoken = substr
                        break
                    end -= 1
                if cur_subtoken is None:
                    is_bad = True
                    break

                sub_tokens.append(cur_subtoken)
                start = end

            if is_bad:
                output_tokens.append(self.unk_token)
            else:
                output_tokens.extend(sub_tokens)
        return output_tokens

def _is_whitespace(char):
    """Checks whether `char` is a whitespace character."""
    if char == " " or char == "\t" or char == "\n" or char == "\r":
        return True
    cat = unicodedata.category(char)
    if cat == "Zs":
        return True
    return False

def _is_control(char):
    """Checks whether `char` is a control character."""
    if char == "\t" or char == "\n" or char == "\r":
        return False
    cat = unicodedata.category(char)
    if cat.startswith("C"):
        return True
    return False

def _is_punctuation(char):
    """Checks whether `char` is a punctuation character."""
    cp = ord(char)
    if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
            (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
        return True
    cat = unicodedata.category(char)
    if cat.startswith("P"):
        return True
    return False

class RobertaTokenizer(PreTrainedTokenizer):
    """RoBERTa tokenizer, similar to BERT but with some differences."""

    vocab_files_names = {'vocab_file': 'vocab.json'}
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES

    def __init__(self, vocab_file, do_lower_case=True, do_basic_tokenize=True,
                 never_split=None, unk_token="<unk>", sep_token="</s>",
                 pad_token="<pad>", cls_token="<s>", mask_token="<mask>",
                 tokenize_chinese_chars=True, strip_accents=None, **kwargs):
        """Constructs a RobertaTokenizer.
        Args:
            vocab_file: Path to a vocabulary file.
            do_lower_case: Whether to lower case the input when tokenizing.
            do_basic_tokenize: Whether to do basic tokenization before wordpiece.
            never_split: List of tokens which will never be split during tokenization.
            unk_token: The unknown token. A string that is not in the vocabulary.
            sep_token: The separator token, which is used when building a sequence
                from multiple sequences, e.g. two sequences for sequence classification.
            pad_token: The token used for padding, for example when batching sequences of different lengths.
            cls_token: The classifier token which is used when doing sequence classification.
            mask_token: The token used for masking values.
            tokenize_chinese_chars: Whether to tokenize Chinese characters.
            strip_accents: Whether to strip all accents.
        """
        super(RobertaTokenizer, self).__init__(**kwargs)

        if not os.path.isfile(vocab_file):
            raise ValueError(
                f"Can't find a vocabulary file at path '{vocab_file}'. To load the vocabulary from a pretrained model use `tokenizer = RobertaTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)`"
            )
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.do_basic_tokenize = do_basic_tokenize
        if do_basic_tokenize:
            self.basic_tokenizer = BasicTokenizer(
                do_lower_case=do_lower_case,
                never_split=never_split,
                tokenize_chinese_chars=tokenize_chinese_chars,
                strip_accents=strip_accents,
            )
        self.wordpiece_tokenizer = WordpieceTokenizer(vocab=self.vocab, unk_token=self.unk_token)

    @property
    def unk_token(self):
        return "<unk>"

    @property
    def sep_token(self):
        return "</s>"

    @property
    def pad_token(self):
        return "<pad>"

    @property
    def cls_token(self):
        return "<s>"

    @property
    def mask_token(self):
        return "<mask>"

    def tokenize(self, text):
        split_tokens = []
        if self.do_basic_tokenize:
            for token in self.basic_tokenizer.tokenize(text):
                for sub_token in self.wordpiece_tokenizer.tokenize(token):
                    split_tokens.append(sub_token)
        else:
            split_tokens = self.wordpiece_tokenizer.tokenize(text)
        return split_tokens

    def convert_tokens_to_ids(self, tokens):
        """Converts a sequence of tokens into ids using the vocab."""
        ids = []
        for token in tokens:
            ids.append(self.vocab.get(token, self.vocab.get(self.unk_token)))
        return ids

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        """Converts a sequence of ids in wordpiece tokens using the vocab."""
        tokens = []
        for id in ids:
            if skip_special_tokens and id in [self.vocab[self.pad_token], self.vocab[self.sep_token], 
                                            self.vocab[self.cls_token], self.vocab[self.mask_token]]:
                continue
            tokens.append(self.ids_to_tokens.get(id, self.unk_token))
        return tokens

    def encode(self, text, add_special_tokens=True, max_length=None, truncation=False, padding=False):
        """Converts a string to a sequence of ids (adds special tokens)."""
        tokens = self.tokenize(text)
        if add_special_tokens:
            tokens = [self.cls_token] + tokens + [self.sep_token]
        
        ids = self.convert_tokens_to_ids(tokens)
        
        if max_length is not None and truncation:
            ids = ids[:max_length]
        
        if padding and len(ids) < max_length:
            ids.extend([self.vocab[self.pad_token]] * (max_length - len(ids)))
        
        return ids

    def decode(self, token_ids, skip_special_tokens=True):
        """Converts a sequence of ids back to a string."""
        tokens = self.convert_ids_to_tokens(token_ids, skip_special_tokens=skip_special_tokens)
        return clean_up_tokenization(" ".join(tokens))

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        """Instantiate a pretrained RoBERTa tokenizer."""
        if pretrained_model_name_or_path in PRETRAINED_VOCAB_FILES_MAP:
            vocab_file = PRETRAINED_VOCAB_FILES_MAP[pretrained_model_name_or_path]
        else:
            vocab_file = os.path.join(pretrained_model_name_or_path, 'vocab.json')
        
        return cls(vocab_file, *args, **kwargs)
