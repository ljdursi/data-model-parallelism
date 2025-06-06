#!/usr/bin/env python3
"""
Example: EuroSAT on multiple GPUs with PyTorch FSDP‑v2 (`fully_shard`).

Launch with:
  torchrun --standalone --nproc_per_node=<GPUS> fsdp2_train.py [...]
"""

import argparse, time, os
import torch, torch.nn as nn, torch.optim as optim, torch.utils.data as tud
import torchvision, torchvision.transforms.v2 as transforms

from torch.profiler import profile, record_function, ProfilerActivity

import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy, OffloadPolicy

class Net(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 1, 1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.Conv2d(128, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.conv_block1(x); x = self.conv_block2(x); x = self.conv_block3(x)
        x = self.global_avg_pool(x); x = self.classifier(x)
        return x


@torch.no_grad()
def test(model, loader, loss_fn, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
        out = model(imgs)
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

    tfm = transforms.Compose([
        transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True),
        transforms.RandomVerticalFlip(), transforms.RandomHorizontalFlip(),
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

    # Build a 1‑D device mesh (world‑size GPUs)
    mesh = init_device_mesh("cuda", (world_size,))

    # **Minimal FSDP‑v2** : shard the *entire* model
    model = fully_shard(model, mesh=mesh)                                       # :contentReference[oaicite:0]{index=0}

    # --- OPTIONAL ADVANCED TOGGLES (uncomment to demo) --------------------------
    # (A) Mixed‑precision (bf16 parameters & reduce‑scatter in fp32)
    # mp = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
    #                           reduce_dtype=torch.float32,
    #                           output_dtype=torch.float32)
    # model = fully_shard(model, mesh=mesh, mp_policy=mp)

    # (B) CPU offload of parameters/optimizer state
    # offload = OffloadPolicy()        # default: offload everything to CPU
    # model = fully_shard(model, mesh=mesh, offload_policy=offload)

    # (C) Gradient‑only sharding (keep params gathered for forward/backward)
    # model = fully_shard(model, mesh=mesh, reshard_after_forward=False)

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
            if rank == 0:
                acc, vloss = test(model, val_loader, loss_fn, device)
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
