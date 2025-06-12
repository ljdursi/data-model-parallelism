#!/usr/bin/env python3

import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms.v2 as transforms

from torch.profiler import profile, record_function, ProfilerActivity

# import torch.utils.data, torch.distributed, 
# and pipeline, SplitPoint, ScheduleGPipe
# from torch.distributed.pipelining
import torch.utils.data as tud
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe
# import checkpoint from torch.distributed
import torch.distributed.checkpoint as dcp

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

def main(args):
    # Initialize distributed environment
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    # define torch device
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size, device_id=device)

    # define the dataset and the transforms we'll apply
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.RandomVerticalFlip(),
            transforms.RandomHorizontalFlip(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # only have rank 0 download the data
    # use barrier to wait for download then other ranks get metadata
    if rank == 0:
        print("Downloading data if needed")
        dataset = torchvision.datasets.EuroSAT(root="./data", download=True, transform=transform)
    dist.barrier()
    if rank != 0:
        dataset = torchvision.datasets.EuroSAT(root="./data", download=False, transform=transform)

    classes = ["AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", "Industrial",
               "Pasture", "PermanentCrop", "Residential", "River", "SeaLake"]
    num_classes = len(classes)

    # dataset split
    total_count = len(dataset)
    train_count = int(0.6 * total_count)
    valid_count = int(0.2 * total_count)
    test_count = total_count - train_count - valid_count

    train_dataset, valid_dataset, test_dataset = tud.random_split(dataset, (train_count, valid_count, test_count))

    # make sure drop_last is true in the data loader, so that there's no partial
    # microbatches; this would mess up the schedule
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8
    )

    # Instantiate the model
    net = Net(num_classes).to(device)

    # only have the 0th global rank print this out
    if rank == 0:
        print(f"{rank}: creating pipeline")

    # TODO - Define the pipeline parallelism
    # First, define example input for pipeline graph analysis
    # use torch.randn
    # Assuming input images are 2x64x64,
    # and there's the microbatch size is args.batch_size // args.chunks

    # TODO - Now, create the split spec to split the model into three stages:
    # Stage 0: conv_block1
    # Stage 1: conv_block2
    # Stage 2: conv_block3 -> global_avg_pool -> classifier
    # use the the SplitPoint functionality to split 
    # automatically.  If you use SplitPoint.END, the final
    # split point can be implicit

    # TODO - Now, create the pipeline with the model,
    # mb_args=(example_input,) and split_spec

    # TODO - now build the pipeline stage using the .build_stage method
    # on the pipe object above

    # define loss
    criterion = nn.CrossEntropyLoss()

    # define optimizer
    # TODO - optimzier should only optimize parameters on the current stage
    # e.g. stage.submod.parameters(), not net.parameters()
    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=args.momentum)

    # create a schedule using ScheduleGPipe on this stage, defining the number of microbatches
    # per iteration and the loss function

    if rank == 0:
        print(f"Beginning training: {args.epochs} epochs")
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('/workspace/logs/eurosat_pipeline'),
                 profile_memory=True) as prof:
        # training loop
        total_time = 0
        nlosses = 0
        for epoch in range(args.epochs):
            running_loss = 0.0
            t0 = time.time()
            optimizer.zero_grad()

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                running_loss += sum(loss).item()

                optimizer.step()
                nlosses += 1

            # timing
            # add a barrier so all ranks are done
            dist.barrier()
            epoch_time = time.time() - t0
            total_time += epoch_time

            # TODO - Step 1 - only have _last_ rank calculate these and print them out
            # output metrics at the end of each epoch
            if rank == world_size - 1:
                images_per_sec = torch.tensor(len(trainloader) * args.batch_size / epoch_time).to(device)
                train_loss = running_loss/nlosses

                print(
                    f"Epoch = {epoch:2d}: Cumulative Time = {total_time:5.3f}, Epoch Time = {epoch_time:5.3f}, Images/sec = {images_per_sec:5.3f}, Train Loss = {train_loss:5.3f}"
                )

            prof.step()

    if rank == 0:
        print("Finished Training")
    
    # replace with torch.distributed.checkpoint
    save_path = "./eurosat_net.pth"
    torch.save(net.state_dict(), save_path)

    #  destroy process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EuroSAT training example',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--chunks', type=int, default=4, help='number of micro-batches (chunks) per mini-batch')
    args = parser.parse_args()
    main(args)
