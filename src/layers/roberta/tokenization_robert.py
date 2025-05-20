from transformers import RobertaTokenizer  # Import RobertaTokenizer
from transformers import PreTrainedTokenizer
import os
import collections

VOCAB_FILES_NAMES = {"vocab_file": "vocab.json", "merges_file": "merges.txt"}  # RoBERTa specific
PRETRAINED_VOCAB_FILES_MAP = {
    "vocab_file": {
        "roberta-base": "https://huggingface.co/roberta-base/resolve/main/vocab.json",
        "roberta-large": "https://huggingface.co/roberta-large/resolve/main/vocab.json",
    },
    "merges_file": {
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
        merges_file,  # RoBERTa specific
        unk_token="[UNK]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        cls_token="[CLS]",
        mask_token="[MASK]",
        add_prefix_space=False,  # RoBERTa specific
        **kwargs,
    ):
        super().__init__(
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
            **kwargs,
        )
        if not os.path.isfile(vocab_file):
            raise ValueError(
                "Can't find a vocabulary file at path '{}'.".format(vocab_file)
            )
        if not os.path.isfile(merges_file):
            raise ValueError(
                "Can't find a merges file at path '{}'.".format(merges_file)
            )

        self.vocab = self.load_vocab(vocab_file, merges_file)  # Use custom load_vocab
        self.ids_to_tokens = collections.OrderedDict(
            [(ids, tok) for tok, ids in self.vocab.items()]
        )
        self.merges = self.load_merges(merges_file)
        self.add_prefix_space = add_prefix_space  # Store the add_prefix_space attribute
        self.roberta_tokenizer = RobertaTokenizerFast(  # Use the fast tokenizer
            vocab_file=vocab_file,
            merges_file=merges_file,
            add_prefix_space=add_prefix_space,
            unk_token=unk_token,
            sep_token=sep_token,
            pad_token=pad_token,
            cls_token=cls_token,
            mask_token=mask_token,
        )

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

    def save_vocabulary(self, vocab_path):
        """Save the tokenizer vocabulary and configuration files (if any) to a directory.
        """
        return self.roberta_tokenizer.save_vocabulary(
            vocab_path
        )  # Use RobertaTokenizer's method

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