# coding=utf-8
""" Yasa Tokenizer."""

from typing import Any, Optional, Tuple, Union, List

import inseption
from transformers import (
    PreTrainedTokenizer,
)

class YasaTiktokenizer(PreTrainedTokenizer):
    """Tokenizer."""

    def __init__(self, tiktoken_special_tokens=None, **kwargs):
        self.tokenizer = inseption.Tokenizer()
        super().__init__(
            unk_token="<|endoftext|>",
            bos_token="<|endoftext|>",
            eos_token="<|endoftext|>",
            add_prefix_space=False,
            **kwargs,
        )
        self.clean_up_tokenization_spaces = False

    @property
    def vocab_size(self) -> int:
        """
        `int`: Size of the base vocabulary (without the added tokens).
        """
        return self.tokenizer.max_token_value + 1

    def get_vocab(self):
        ret = {}
        for i in range(self.tokenizer.max_token_value+1):
            try:
                ret[self.tokenizer.decode_single_token_bytes(i)] = i
            except:
                pass
        return ret

    def _tokenize(self, text, **kwargs):
        """
        Converts a string in a sequence of tokens (bytes), using the tokenizer.

        Do NOT take care of added tokens.
        """
        output = [
            self._convert_id_to_token(t)
            for t in self.tokenizer.encode(
                text
            )
        ]
        return output

    def _convert_token_to_id(self, token):
        return self.tokenizer.encode_single_token(token)

    def _convert_id_to_token(self, index: int) -> bytes:
        return self.tokenizer.decode_single_token_bytes(index)

    def convert_tokens_to_string(self, tokens: List[bytes]) -> str:
        return b"".join(tokens).decode("utf8", errors=self.utf8_decoding_strategy)
