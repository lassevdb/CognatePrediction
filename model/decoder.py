"""
Decoder-only Cognate Prediction
Lasse van den Berg, Adnan Bseisu
CPSC 477, Spring 2024

This file contains our decoder-only transformer class.
It is based on the PSET from class 
"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import List, Dict, Optional

class Embedding(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int):
        super(Embedding, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        
    def forward(self, x: Tensor) -> Tensor:
        return self.emb(x)

    
class PositionalEmbeddings(nn.Module):
    def __init__(self, d_model: int, max_len: int): 
        super(PositionalEmbeddings, self).__init__()
        self.positional_embeddings = nn.Embedding(max_len, d_model)
        
    def forward(self, pos: Tensor) -> Tensor:
        return self.positional_embeddings(pos)


class TokenEmbedder(nn.Module):
    
    def __init__(self, vocab_size: int, d_model: int, max_len: int):
        super(TokenEmbedder, self).__init__()
        self.token_embedding = Embedding(vocab_size, d_model)
        self.positional_embedding = PositionalEmbeddings(d_model, max_len)

    def forward(self, x: Tensor) -> Tensor:
        pos = torch.arange(0, x.shape[1], dtype=torch.long).to(x.device) # shape: [sequence length]
        token_embeddings = self.token_embedding(x)
        return token_embeddings + self.positional_embedding(pos)

    
class Projection(nn.Module):   
    def __init__(self, d_model, vocab_size):
        super(Projection, self).__init__()
        self.ff = nn.Linear(d_model, vocab_size)
        
    def forward(self, x):
        return self.ff(x)
    

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, attn_pdrop: float):
        super(MultiHeadAttention, self).__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(attn_pdrop)

    def forward(self, x: Tensor, causal_mask: bool, print_mask=False) -> Tensor:
        batch_size, seq_len, d_model = x.size()
        
        query_bsd = self.q_proj(x)  # b:batch_size, s:seq_len, k: d_k
        key_bsd = self.k_proj(x)
        value_bsd = self.v_proj(x)

        query_bhsk = query_bsd.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        key_bhsk = key_bsd.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value_bhsk = value_bsd.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        raw_scores_bhst = torch.einsum('bhqd,bhkd->bhqk',query_bhsk,key_bhsk)

        if causal_mask != None:
            def fill_tensor(tensor, indices):
                batch_size = tensor.size(0)
                for i in range(batch_size):
                    n = indices[i]
                    tensor[i, :, :, :n] = 1
                return tensor
            indeces = causal_mask
            causal_mask = torch.tril(torch.ones(seq_len, seq_len)).type_as(raw_scores_bhst)
            prefix_mask = fill_tensor(torch.zeros(batch_size, self.num_heads, seq_len, seq_len), indeces)
            mask = prefix_mask.masked_fill(causal_mask==1, 1) 
            mask = mask.permute(0, 1, 3, 2) 
            raw_scores_bhst = raw_scores_bhst.masked_fill(mask==0, float('-inf'))
            
            # attention_weights = mask[0, 0][:20,:20].detach().numpy() 
            # sns.set(font_scale=0.7)
            # plt.figure(figsize=(8, 6))
            # sns.heatmap(attention_weights, annot=False, cmap='Blues', fmt='.2f', square=True,)
            # plt.title('Attention Pattern')
            # plt.xlabel('Input Tokens')
            # plt.ylabel('Output Tokens')
            # plt.show()
            # print(raw_scores_bhst)
            # raw_scores_bhst = raw_scores_bhst.masked_fill(mask_bhst==0, float('-inf'))

        attn_probs_bhst = F.softmax(raw_scores_bhst, dim=-1)

        if print_mask:
            for i in range(1):
                attention_weights = attn_probs_bhst[0, i][:20,:20].detach().numpy()  # Replace with your attention weights

                sns.set(font_scale=0.7)
                plt.figure(figsize=(8, 6))
                sns.heatmap(attention_weights, annot=False, cmap='Blues', fmt='.2f', square=True,)
                plt.title('Attention Pattern')
                plt.xlabel('Input Tokens')
                plt.ylabel('Output Tokens')
                plt.show()

        attn_probs_bhst = self.attn_dropout(attn_probs_bhst)
        out_bsd = torch.einsum('bhql,bhld->bqhd', attn_probs_bhst, value_bhsk).transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        final_out_bsd = self.out_proj(out_bsd)

        return final_out_bsd
    


    
class FeedForward(nn.Module):
    
    def __init__(self, d_model, d_ff, dropout):
        super(FeedForward, self).__init__()
        self.ln1 = nn.Linear(d_model, d_ff)
        self.ln2 = nn.Linear(d_ff, d_model)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)


    def forward(self, x):
        x = self.ln1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.ln2(x)
        return x
    

class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps: float):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(hidden_size))
        self.beta = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps

    def forward(self, x):
        mean = torch.mean(x, dim=-1, keepdim=True)
        std = torch.std(x, dim=-1, keepdim=True, unbiased=False)
        normalized = (x - mean) / (std + self.eps)
        return self.gamma * normalized + self.beta


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, attn_pdrop, dropout, d_ff, eps):
        super(TransformerBlock, self).__init__()
        self.attn = MultiHeadAttention(d_model, num_heads, attn_pdrop)
        self.ln1 = LayerNorm(d_model, eps)
        self.ff = FeedForward(d_model, d_ff, dropout)
        self.ln2 = LayerNorm(d_model, eps)

    def forward(self, x, causal_mask, print_mask=False):
        x_attn = self.attn(x, causal_mask, print_mask)
        x = self.ln1(x + x_attn)
        x_ff = self.ff(x)
        x = self.ln2(x + x_ff)
        return x

    
class TransformerStack(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, attn_pdrop, dropout, d_ff, eps):
        super(TransformerStack, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.extend([TransformerBlock(d_model=d_model,
                                        num_heads=num_heads,
                                        attn_pdrop=attn_pdrop,
                                        dropout=dropout,
                                        d_ff=d_ff,
                                        eps=eps) for i in range(num_layers)])
        
        self.dims = [500, 400, 300, 200, 100]
        dims = self.dims
        self.squeeze = nn.Linear(d_model * 24, dims[0])

        self.l1 = nn.Linear(dims[0], dims[1])
        self.l2 = nn.Linear(dims[1], dims[2])
        self.l3 = nn.Linear(dims[2], dims[3])
        self.l4 = nn.Linear(dims[3], dims[4])

        self.tanh = nn.Tanh()
        self.tangle = nn.Linear(100, 100)

        self.l4_ = nn.Linear(dims[4], dims[3])
        self.l3_ = nn.Linear(dims[3], dims[2])
        self.l2_ = nn.Linear(dims[2], dims[1])
        self.l1_ = nn.Linear(dims[1], dims[0])
        self.unsqueeze = nn.Linear(dims[0],d_model * 24)

        self.a = nn.ModuleList()
        self.a.extend([TransformerBlock(d_model=d_model,
                                        num_heads=num_heads,
                                        attn_pdrop=attn_pdrop,
                                        dropout=dropout,
                                        d_ff=d_ff,
                                        eps=eps) for i in range(num_layers)])

    def forward(self, x, causal_mask, print_mask=False):
        for layer in self.layers:
            x = layer(x, causal_mask, print_mask)
        return x


class TransformerModel(nn.Module):
    def __init__(self, tokenizer, vocab_size, d_model, num_heads, attn_pdrop, dropout, d_ff, max_len, num_layers, eps):
        super(TransformerModel, self).__init__()
        self.tokenizer = tokenizer
        self.token_embedder = TokenEmbedder(vocab_size=vocab_size, d_model=d_model, max_len=max_len)
        self.transformer_stack = TransformerStack(num_layers=num_layers,
                                                 d_model=d_model,
                                                 num_heads=num_heads,
                                                 attn_pdrop=attn_pdrop,
                                                 dropout=dropout,
                                                 d_ff=d_ff,
                                                 eps=eps)
        self.projection = Projection(d_model=d_model, vocab_size=vocab_size)

    def forward(self, x, causal_mask=None, print_mask=False):
        embeddings = self.token_embedder(x)
        logits = self.transformer_stack(embeddings, causal_mask, print_mask)
        return self.projection(logits)
    
    
    def generate(self, inputs: List[str]):

        tokenizer = self.tokenizer

        generated_sequence = inputs[0] #string
        tokens = tokenizer(inputs) #[1, 50]
        idx_to_predict = len(inputs[0])

        with torch.no_grad():
            
            for i in range(idx_to_predict, tokenizer.max_length - 1):
                
                logits = self.forward(tokens, torch.tensor([i]))
                
                next_token = torch.argmax(logits, dim=-1)[0][i].unsqueeze(0)
                previous_tokens = tokens[0][:i]

                tokens = torch.concat((previous_tokens, next_token), dim=0)
                tokens = tokenizer(tokenizer.decode(tokens)).unsqueeze(0)
                
                generated_sequence += tokenizer.decode(next_token, True)
                if tokenizer.decode(next_token) == ">":
                    print("End")
                    break
        print("Full")
        return generated_sequence
    

def NewInstance(config = {
                "tokenizer": None,
                "vocab_size": 50,
                "max_len": 50,
                "d_model": 128,
                "num_heads": 8,
                "attn_pdrop": 0.2,
                "dropout": 0.2,
                "d_ff": 128,
                "num_layers":2,
                "eps": 1e-6}):
    
    tokenizer = config["tokenizer"]
    vocab_size = config["vocab_size"]
    d_model = config["d_model"]
    num_heads = config["num_heads"]
    attn_pdrop = config["attn_pdrop"]
    dropout = config["dropout"]
    d_ff = config["d_ff"]
    max_len = config["max_len"]
    num_layers = config["num_layers"]
    eps = config["eps"]
    
    model = TransformerModel(tokenizer=tokenizer,
                             vocab_size=vocab_size,
                             d_model=d_model,
                             num_heads=num_heads,
                             attn_pdrop=attn_pdrop,
                             dropout=dropout,
                             d_ff=d_ff,
                             max_len=max_len,
                             num_layers=num_layers,
                             eps=eps)
    return model

