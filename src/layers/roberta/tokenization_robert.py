import random
from transformers import RobertaTokenizer, RobertaTokenizerFast   # Import RobertaTokenizer
from transformers import PreTrainedTokenizer
from typing import List, Optional, Tuple
import os
import collections

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}  # RoBERTa specific
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    },
    "merges_file": {~
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/merges.txt",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/merges.txt",
    },
}
PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "roberta-base": 512,
    "roberta-large": 512,
}  # RoBERTa specific
class RobertaTokenizerModified(PreTrainedTokenizer):  # Rename the class
    """
    Constructs a RobertaTokenizer.  This class is adapted from BertTokenizer.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]  # RoBERTa specific

    def __init__(
        self,
        vocab_file,
        merges_file,
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        add_prefix_space=False,  # RoBERTa specific
        **kwargs,
    ):
        # Step 1: Initialize the underlying RobertaTokenizerFast.
        # This tokenizer will correctly load vocab.json and merges.txt.
        self.roberta_tokenizer = RobertaTokenizerFast(
            vocab_file=vocab_file,
            merges_file=merges_file,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space,
            **kwargs # Pass kwargs, RobertaTokenizerFast will use what it needs
        )

        # Step 2: Set self.vocab using the vocab loaded by RobertaTokenizerFast.
        # This is crucial because super().__init__() will call self.get_vocab(),
        # which in this class returns self.vocab.
        self.vocab = self.roberta_tokenizer.get_vocab()  # get_vocab() returns a dict copy
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        # The 'add_prefix_space' attribute is primarily for the roberta_tokenizer's behavior.
        # Storing it on self directly might be for consistency or if any local methods use it.
        self.add_prefix_space = add_prefix_space

        # Step 3: Now call the superclass __init__.
        # It handles setting up special token attributes on the tokenizer instance.
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            add_prefix_space=add_prefix_space, # Pass this, PreTrainedTokenizer may use it via kwargs
            **kwargs # Pass remaining kwargs
        )

    def get_random_token(self) -> str:
        """
        Gets a random token from the vocabulary, excluding special tokens.
        """
        # Use the vocabulary from the underlying RobertaTokenizerFast instance
        vocab_items = list(self.roberta_tokenizer.get_vocab().keys())

        # Identify special tokens to exclude.
        # self.all_special_tokens is available from PreTrainedTokenizer.
        special_tokens_to_exclude = set(self.all_special_tokens)

        # Filter out special tokens
        valid_tokens = [token for token in vocab_items if token not in special_tokens_to_exclude]

        if not valid_tokens:
            # Fallback or error if no valid tokens are found (should not happen with a proper vocab)
            # You could return a common token or raise an error.
            # For simplicity, let's pick from all tokens if filtering somehow fails,
            # though this is less ideal for MLM.
            # A better fallback might be to return self.unk_token, but that's not random.
            # Or, ensure vocab_items itself is non-empty and pick from it if valid_tokens is empty.
            if not vocab_items:
                raise ValueError("Vocabulary is empty, cannot select a random token.")
            return random.choice(vocab_items)

        return random.choice(valid_tokens)

    def build_inputs_with_special_tokens(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None
    ) -> List[int]:
        """
        Build model inputs from a sequence or a pair of sequence for sequence classification tasks
        by concatenating and adding special tokens.
        A RoBERTa sequence has the following format:
            - single sequence:         `[CLS] X [SEP]`
            - pair of sequences:    `[CLS] A [SEP] B [SEP]`
        """
        if token_ids_1 is None:
            return [self.cls_token_id] + token_ids_0 + [self.sep_token_id]
        cls_token = [self.cls_token_id]
        sep_token = [self.sep_token_id]
        return cls_token + token_ids_0 + sep_token + token_ids_1 + sep_token
    def get_special_tokens_mask(
        self, token_ids_0: List[int], token_ids_1: Optional[List[int]] = None,
        with_added_special_tokens: bool = True
    ) -> List[int]:
        """
        Retrieves sequence id that mask the special tokens from the rest.
        """
        if with_added_special_tokens:
            if token_ids_1 is None:
                return [1] + ([0] * len(token_ids_0)) + [1]
            return [1] + ([0] * len(token_ids_0)) + [1] + ([0] * len(token_ids_1)) + [1]
        elif token_ids_1 is None:
            return [0] * len(token_ids_0)
        else:
            return [0] * len(token_ids_0) + [0] * len(token_ids_1)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def _tokenize(self, text):
        """Tokenize a string."""
        if self.add_prefix_space:
            text = " " + text
        return self.roberta_tokenizer.tokenize(
            text
        )  # Use the RobertaTokenizer's tokenize method

    def _convert_token_to_id(self, token):
        """Converts a token (str) in an id using the vocab."""
        return self.roberta_tokenizer.convert_tokens_to_ids(token)

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (str) using the vocab."""
        return self.roberta_tokenizer.convert_ids_to_tokens(index)

    def convert_tokens_to_string(self, tokens):
        """Converts a sequence of tokens (str) in a single string."""

        return self.roberta_tokenizer.convert_tokens_to_string(tokens)

    def save_vocabulary(self, save_directory: str, filename_prefix: Optional[str] = None) -> Tuple[str]:
        """
        Save the tokenizer vocabulary and merge files to a directory.
        The `filename_prefix` argument is compatible with the parent Hugging Face
        tokenizer's `save_vocabulary` method.
        """
        # This delegates to RobertaTokenizerFast's save_vocabulary,
        # which handles vocab.json and merges.txt.
        return self.roberta_tokenizer.save_vocabulary(
            save_directory=save_directory,
            filename_prefix=filename_prefix
        ) # Use RobertaTokenizer's method

    def get_vocab(self):
        """
        Returns the vocabulary as a dictionary of token to ids.
        """
        return self.vocab

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs):
        """Instantiate a RobertaTokenizerModified from a pre-trained tokenizer.
        """
        return super().from_pretrained(pretrained_model_name_or_path, *inputs, **kwargs)

    def load_vocab(self, vocab_file, merges_file):
        """Loads the vocabulary from file."""
        vocab = {}
        with open(vocab_file, "r", encoding="utf-8") as f:
            for index, line in enumerate(f):
                word = line.strip()
                vocab[word] = index
        with open(merges_file, "r", encoding="utf-8") as f:
            merges = f.read().split("\n")[1:-1]
        vocab = {v: k for k, v in enumerate(
            ["!", "#", "$", "%", "&", "'", "(", ")", "*", "+", ",", "-", ".", "/", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";", "<", "=", ">", "?", "@", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "[", "\\", "]", "^", "_", "`", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "{", "|", "}", "~", "Ġ", "…", "\n", ""] + merges
        )}
        return vocab

    def load_merges(self, merges_file):
        """Loads the merges file."""
        with open(merges_file, "r", encoding="utf-8") as f:
            merges = f.read().split("\n")[1:-1]
        return merges