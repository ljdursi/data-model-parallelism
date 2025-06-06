"""
Pipelineing example, taken from https://docs.pytorch.org/tutorials//intermediate/pipelining_tutorial.html
with some modifications for consistentcy with the rest of the material.

Multi-gpu example
"""
import torch
import torch.nn as nn
from dataclasses import dataclass

import os
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe

import torch.optim as optim

vocab_size = 10000

class Transformer(nn.Module):
   def __init__(self, dim:int = 512, n_layers:int = 8, n_heads:int = 8, vocab_size:int = 10000):
      super().__init__()

      self.tok_embeddings = nn.Embedding(vocab_size, dim)

      self.layers = torch.nn.ModuleDict()
      for layer_id in range(n_layers):
            self.layers[str(layer_id)] = nn.TransformerDecoderLayer(dim, n_heads)

      self.norm = nn.LayerNorm(dim)
      self.output = nn.Linear(dim, vocab_size)

   def forward(self, tokens: torch.Tensor):
      h = self.tok_embeddings(tokens) if self.tok_embeddings else tokens

      for layer in self.layers.values():
            h = layer(h, h)

      h = self.norm(h) if self.norm else h
      output = self.output(h).clone() if self.output else h
      return output

def init_distributed():
   local_rank = int(os.environ.get("LOCAL_RANK", 0))
   rank = int(os.environ.get("RANK", 0))
   world_size = int(os.environ.get("WORLD_SIZE", 1))

   device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
   dist.init_process_group()

   return local_rank, rank, world_size, device

def cleanup():
    dist.destroy_process_group()

if __name__ == "__main__":
    local_rank, rank, world_size, device = init_distributed()
    num_microbatches = 4
    model = Transformer(vocab_size=vocab_size)

    # Dummy data to prime the pipeline, to create the job graph
    x = torch.ones(32, 500, dtype=torch.long)
    y = torch.randint(0, vocab_size, (32, 500), dtype=torch.long)
    example_input_microbatch = x.chunk(num_microbatches)[0]

    # manually split the graph 
    split_spec={"layers.4": SplitPoint.BEGINNING,}
    pipe = pipeline(model, mb_args=(example_input_microbatch,), split_spec=split_spec)
    stage = pipe.build_stage(rank, device, dist.group.WORLD)

    model.to(device)
    x = x.to(device)
    y = y.to(device)

    def tokenwise_loss_fn(outputs, targets):
       loss_fn = nn.CrossEntropyLoss()
       outputs = outputs.reshape(-1, vocab_size)
       targets = targets.reshape(-1)
       return loss_fn(outputs, targets)

    lr = 0.1
    momentum = 0.8

    schedule = ScheduleGPipe(stage, n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)
    optimizer = optim.SGD(stage.submod.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(5):
        optimizer.zero_grad()
        if rank == 0:
           schedule.step(x)
        elif rank == 1:
           losses = []
           output = schedule.step(target=y, losses=losses)
           print(f"epoch: {epoch} losses: {sum(losses)}")

        optimizer.step()
       
    cleanup()
