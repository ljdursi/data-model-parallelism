#!/usr/bin/env python3
"""
Example: EuroSAT step 0 for FSDP lab

Launch with:
  ./code/run_w_torchrun.sh 2 ./code/eurosat_fsdp_step0.py
"""

import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

import torchvision
import torchvision.transforms.v2 as transforms

from torch.profiler import profile, ProfilerActivity

import torch.distributed as dist
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy

# define our model
class Net(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(
                in_channels=3, out_channels=32, kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64 -> 32x32
        )

        self.conv_block2 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, out_channels=64, kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
        )

        self.conv_block3 = nn.Sequential(
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=3, padding=1, stride=1
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
        )

        # Global Average Pooling and Fully Connected Layers
        self.global_avg_pool = nn.AdaptiveAvgPool2d(
            (1, 1)
        )  # Reduces each 128-channel map to 1x1

        self.classifier = nn.Sequential(
            nn.Flatten(),  # Input will be (batch_size, 128)
            nn.Linear(in_features=128, out_features=64),
            nn.ReLU(),
            nn.Dropout(0.5),  # Standard dropout for FC layers
            nn.Linear(in_features=64, out_features=num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)

        x = self.global_avg_pool(x)
        x = self.classifier(x)

        return x

@torch.no_grad()
def test(model, loader, loss_fn, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        labels = labels.to(device)

        # forward pass
        out = model(imgs)

        # extracting predicted labels
        loss_sum += loss_fn(out, labels).item()
        correct += (out.argmax(1) == labels).sum()
        total += labels.numel()
    return correct / total, loss_sum / len(loader)


def main(args):
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo",
                            world_size=world_size,
                            rank=rank,
                            device_id=device)

    transform = transforms.Compose([
        transforms.ToImage(), 
        transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomVerticalFlip(), 
        transforms.RandomHorizontalFlip(),
        transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
    ])

    if local_rank == 0:
        dataset = torchvision.datasets.EuroSAT(root="./data", download=True, transform=tfm)
    dist.barrier()
    if local_rank != 0:
        dataset = torchvision.datasets.EuroSAT(root="./data", download=False, transform=tfm)

    n = len(dataset); n_train, n_val = int(0.6*n), int(0.2*n)
    train_set, val_set, _ = tud.random_split(dataset, [n_train, n_val, n-n_train-n_val])

    # Distributed data loading
    train_sampler = tud.distributed.DistributedSampler(train_set)
    train_loader  = tud.DataLoader(train_set, batch_size=args.batch_size,
                                   sampler=train_sampler, num_workers=8, pin_memory=True)
    val_loader    = tud.DataLoader(val_set, batch_size=args.batch_size,
                                   shuffle=False, num_workers=8, pin_memory=True)

    model = Net(num_classes=10).to(device)

    # **Minimal FSDP‑v2** : shard the *entire* model
    for layer in model.layers:
        fully_shard(layer)
    fully_shard(model)                                       # :contentReference[oaicite:0]{index=0}

    # --- OPTIONAL ADVANCED TOGGLES (uncomment to demo) --------------------------
    # (A) Mixed‑precision (bf16 parameters & reduce‑scatter in fp32)
    # mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
    #                           reduce_dtype=torch.float32,
    #                           output_dtype=torch.float32)
    # model = fully_shard(model, mp_policy=mp)

    # (B) CPU offload of parameters/optimizer state
    # offload = OffloadPolicy()        # default: offload everything to CPU
    # model = fully_shard(model, offload_policy=offload)

    # (C) Gradient‑only sharding (keep params gathered for forward/backward)
    # model = fully_shard(model, reshard_after_forward=False)

    # ---------------------------------------------------------------------------

    loss_fn   = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('/workspace/logs/eurosat_fsdp'),
                 profile_memory=True) as prof:
        for epoch in range(args.epochs):
            model.train()
            train_sampler.set_epoch(epoch)       # shuffle per epoch
            t0, running = time.time(), 0.0
            for i, (imgs, labels) in enumerate(train_loader):
                imgs, labels = imgs.to(device), labels.to(device)
                optimizer.zero_grad(set_to_none=True)
                out = model(imgs)
                loss = loss_fn(out, labels)
                loss.backward()
                optimizer.step()
                running += loss.item()

            # simple metrics (rank‑0)
            dist.barrier()
            images_per_sec = torch.tensor(len(train_loader) * args.batch_size / epoch_time).to(device)
            acc, vloss = test(model, val_loader, loss_fn, device)
            dist.all_reduce(acc, op=dist.ReduceOp.AVG)
            dist.all_reduce(vloss, op=dist.ReduceOp.AVG)
            dist.all_reduce(images_per_sec, op=dist.ReduceOp.SUM)

            if rank == 0:
                imgs_sec = (len(train_loader.dataset) / (time.time() - t0))
                print(f"[epoch {epoch:2d}] loss {running/len(train_loader):.4f}  "
                    f"val‑loss {vloss:.4f}  val‑acc {acc:.4f}  imgs/s {imgs_sec:.1f}")

            prof.step()

        if rank == 0:
            torch.save(model.state_dict(), "eurosat_fsdp2.pth")
            print("Checkpoint saved to eurosat_fsdp2.pth")

    dist.barrier()            # wait so rank‑0 doesn’t exit early
    dist.destroy_process_group()


if __name__ == "__main__":
    p = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--epochs",     type=int, default=10)
    p.add_argument("--lr",         type=float, default=0.01)
    p.add_argument("--momentum",   type=float, default=0.9)
    args = p.parse_args()
    main(args)
