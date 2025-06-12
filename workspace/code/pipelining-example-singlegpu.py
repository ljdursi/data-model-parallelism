"""
Pipelineing example, taken from https://docs.pytorch.org/tutorials//intermediate/pipelining_tutorial.html
with some modifications for consistentcy with the rest of the material.

Single-gpu example
"""
import torch
import torch.nn as nn

# TODO - import os for os.environ

# TODO - import torch.distributed,
# and pipeline, SplitPoint, and ScheduleGPipe from
# torch.distributed.pipelining
# TODO - import checkpoint from torch.distributed

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

# TODO - get local_rank, rank, world_size,
# and init the process group
# return all + device
def init_device():
   device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
   return device

# TODO - destroy process group
def cleanup():
    pass

if __name__ == "__main__":
    device = init_device()
    num_microbatches = 4
    batch_size = 32
    seq_len = 500
    model = Transformer(vocab_size=vocab_size)

    # dummy data
    x = torch.ones(32, 500, dtype=torch.long)
    y = torch.randint(0, vocab_size, (32, 500), dtype=torch.long)

    # TODO-example data (microbatch-sized) to prime the pipeline, to create the job graph

    # TODO - manually split the graph 
    # use pipeline and split_spec, along with example data

    # TODO - don't need this if we've split the model up
    model.to(device)

    # TODO - only move data to the device if it's used on that device
    # (e.g. inputs on rank 0, outputs on rank 1)
    x = x.to(device)
    y = y.to(device)

    def tokenwise_loss_fn(outputs, targets):
       loss_fn = nn.CrossEntropyLoss()
       outputs = outputs.reshape(-1, vocab_size)
       targets = targets.reshape(-1)
       return loss_fn(outputs, targets)

    lr = 0.1
    momentum = 0.8

    # TODO - add a ScheduleGPipe scheduler

    # TODO - optimizer only applies to stage.submod parameters
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    
    for epoch in range(2):
        optimizer.zero_grad()
                
        # TODO - replace this with a schedule.step() 
        # the call will depend on the rank
        outputs = model(y)
        loss = tokenwise_loss_fn(outputs, y)
        loss.backward()

        # TODO - only last rank knows the losses
        print(f"epoch: {epoch} losses: {torch.mean(loss)}")

        optimizer.step()
       
    # TODO - save just the stage
    chkpt_dir='pipeline'
    torch.save(model.state_dict(), chkpt_dir+"/example-single-model.pt") 

    cleanup()