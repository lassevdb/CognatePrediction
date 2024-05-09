"""
Decoder-only Cognate Prediction
Lasse van den Berg, Adnan Bseisu
CPSC 477, Spring 2024

This file contains our custom CharacterTokenizer class.
It contains several helpful functions to tokenize, detokenize and process
vocabulary.

This class was made from scratch to ensure that it fits our specific needs for
decoding.
"""

import torch
from torch import Tensor
from collections import Counter
from typing import Optional, Tuple, Dict, List, Union


class CharacterTokenizer:
    def __init__(self,
                 corpus: Union[List[str], str],
                 max_length: int,
                 unk_token: Optional[str] = "*",
                 pad_token: Optional[str] = "#",
                 bos_token: Optional[str] = "<",
                 sep_token: Optional[str] = "|",
                 eos_token: Optional[str] = ">"):
        
        self.max_length = max_length
        self.corpus = corpus
        
        self.unk_token = unk_token
        self.pad_token = pad_token
        self.bos_token = bos_token
        self.sep_token = sep_token
        self.eos_token = eos_token
        
        self.special_tokens = [
            self.unk_token,
            self.pad_token,
            self.bos_token,
            self.sep_token,
            self.eos_token
        ]
        
        counts = Counter("".join(corpus))
        counts = sorted(counts, key=lambda x: counts[x], reverse=True)
        
        self.vocab = self.special_tokens + counts          
        self.vocab_size = len(self.vocab)

        self.token_to_idx = {token: idx for idx, token in enumerate(self.vocab)}
        self.idx_to_token = {idx: token for token, idx in self.token_to_idx.items()}

        
    def __call__(self,
                 inputs: Union[List[str], str],
                 as_tensor: Optional[bool] = True,
                 padding: Optional[bool] = True,) -> Union[torch.Tensor, List[list], list]:
        
        return self.tokenize(inputs, as_tensor=as_tensor, padding=padding)

    
    def _tokenize_string(self,
                         string: str,
                         as_tensor: Optional[bool] = True,
                         padding: Optional[bool] = True) -> Union[torch.Tensor, list]:
        
        assert type(string) == str
        tokens = [self.token_to_idx[char] if char in self.vocab \
                  else self.token_to_idx[self.unk_token] \
                  for char in list(string)]        
        if padding:
            padded_list = [self.token_to_idx[self.pad_token]] * self.max_length
            for i, t in enumerate(tokens):
                padded_list[i] = t
                tokens = padded_list      
        if as_tensor: return torch.Tensor(tokens).to(torch.int)
        else: return tokens

        
    def _tokenize_list(self,
                       str_list: List[str],
                       as_tensor: Optional[bool] = True,
                       padding: Optional[bool] = True) -> Union[torch.Tensor, List[list]]:
        
        assert type(str_list) == list
        
        encodings = [self._tokenize_string(instance, as_tensor, padding) for instance in str_list]
        
        if as_tensor: return torch.stack(encodings)
        else: return encodings

        
    def tokenize(self,
                 inputs: Union[List[str], str],
                 as_tensor: Optional[bool] = True,
                 padding: Optional[bool] = True,) -> Union[torch.Tensor, List[list], list]:
        
        assert not (as_tensor and not padding)
        
        if type(inputs) == list:
            return self._tokenize_list(inputs, as_tensor=as_tensor, padding=padding)
        elif type(inputs) == str:
            return self._tokenize_string(inputs, as_tensor=as_tensor, padding=padding)
        else:
            raise 

            
    def _decode(self,
                encoded_indeces: torch.Tensor,
                display_padding: Optional[bool] = False) -> str:
        
        assert len(list(encoded_indeces.shape)) == 1
        decoding = [self.idx_to_token[idx] for idx in encoded_indeces.tolist()]
        if not display_padding: decoding = [char for char in decoding if char != self.pad_token]
        return "".join(decoding)
    
    
    def _decode_batch(self,
                     encoded_batch: torch.Tensor,
                     display_padding: Optional[bool] = False) -> List[str]:
        
        assert len(list(encoded_batch.shape)) == 2
        decodings = [self._decode(instance, display_padding=display_padding) for instance in encoded_batch]
        return decodings
    
    
    def decode(self,
              encodings: torch.Tensor,
              display_padding: Optional[bool] = False) -> Union[List[str], str]:
        
        if len(list(encodings.shape)) == 1:
            return self._decode(encodings, display_padding=display_padding)
        elif len(list(encodings.shape)) == 2:
            return self._decode_batch(encodings, display_padding=display_padding)
        else:
            raise
