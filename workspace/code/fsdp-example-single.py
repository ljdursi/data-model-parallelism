"""
FSDP example, single-GPU version (no FSDP)

Slightly tweaked from - https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html
"""
import argparse
from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

# import fully_shard, FSDPModule, and FullyShardedDataParallel from torch.distributed.fsdp
# import FullStateDictConfig & StateDictType from torch.distributed.fsdp and 
#   get_model_state_dict, StateDictOptions from torch.distributed.checkpoint.state_dict 

@dataclass
class ModelArgs:
    n_layers: int = 2
    vocab_size: int = 8
    max_seq_len: int = 16
    dim: int = 16
    n_heads: int = 4
    dropout_p: float = 0.1

class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.dim % args.n_heads == 0
        self.head_dim = args.dim // args.n_heads
        self.n_heads = args.n_heads
        self.dropout_p = args.dropout_p
        self.resid_dropout = nn.Dropout(args.dropout_p)

        self.wq = nn.Linear(args.dim, args.dim, bias=False)
        self.wk = nn.Linear(args.dim, args.dim, bias=False)
        self.wv = nn.Linear(args.dim, args.dim, bias=False)
        self.wo = nn.Linear(args.dim, args.dim, bias=False)

    def forward(self, x):
        bsz, seq_len, _ = x.size()
        queries, keys, values = self.wq(x), self.wk(x), self.wv(x)
        queries = queries.view(bsz, seq_len, self.n_heads, self.head_dim)
        keys = keys.view(bsz, seq_len, self.n_heads, self.head_dim)
        values = values.view(bsz, seq_len, self.n_heads, self.head_dim)

        queries = queries.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        keys = keys.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)
        values = values.transpose(1, 2)  # (bsz, n_heads, seq_len, head_dim)

        output = F.scaled_dot_product_attention(
            queries,
            keys,
            values,
            None,
            self.dropout_p if self.training else 0,
        )
        output = output.transpose(1, 2).contiguous().view(bsz, seq_len, -1)
        return self.resid_dropout(self.wo(output))


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout_p):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim)
        self.gelu = nn.GELU()
        self.w2 = nn.Linear(hidden_dim, dim)
        self.resid_dropout = nn.Dropout(dropout_p)

    def forward(self, x):
        return self.resid_dropout(self.w2(self.gelu(self.w1(x))))

class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.attention_norm = nn.LayerNorm(args.dim)
        self.attention = Attention(args)
        self.ffn_norm = nn.LayerNorm(args.dim)
        self.feed_forward = FeedForward(
            args.dim, hidden_dim=4 * args.dim, dropout_p=args.dropout_p
        )

    def forward(self, x):
        h = x + self.attention(self.attention_norm(x))
        out = h + self.feed_forward(self.ffn_norm(h))
        return out


# A toy transformer model, partly inspired by the nanoGPT model:
# https://github.com/karpathy/nanoGPT.
class Transformer(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        assert args.vocab_size is not None
        assert args.max_seq_len is not None
        self.model_args = args
        self.max_seq_len = args.max_seq_len
        self.tok_embeddings = nn.Embedding(args.vocab_size, args.dim)
        self.pos_embeddings = nn.Embedding(args.max_seq_len, args.dim)
        self.dropout = nn.Dropout(args.dropout_p)
        self.layers = nn.ModuleList()
        for _ in range(args.n_layers):
            self.layers.append(TransformerBlock(args))
        self.norm = nn.LayerNorm(args.dim)
        self.output = nn.Linear(args.dim, args.vocab_size, bias=False)

    def forward(self, tokens):
        _bsz, seq_len = tokens.size()
        assert seq_len <= self.max_seq_len
        h = self.tok_embeddings(tokens)
        pos = torch.arange(0, seq_len, device=tokens.device)
        p = self.pos_embeddings(pos)  # positional embeddings of shape (seq_len, dim)
        h = h + p
        h = self.dropout(h)
        for layer in self.layers:
            h = layer(h)
        h = self.norm(h)
        output = self.output(h).float()
        return output


def main(args):
    # usual init process group stuff here
    rank = 0
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    torch.manual_seed(0)
    
    vocab_size = 1024
    batch_size = 32
    seq_len = 64
    model_args = ModelArgs(
        n_layers=10,
        n_heads=4,
        vocab_size=vocab_size,
        max_seq_len=seq_len,
        dropout_p=0,
    )
    
    model = Transformer(model_args).to(device)

    # fully shard model with default args
    print(model)

    optim = torch.optim.Adam(model.parameters(), lr=1e-2)

    for _ in range(10):
        optim.step()
        x = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        loss = model(x).sum()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optim.zero_grad()

    # get model state on CPU and only have rank 0 save
    torch.save(model.state_dict(), "fsdp-example-singlegpu.pt")

    # destroy process group

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch FSDP2 example")
    args = parser.parse_args()
    main(args)