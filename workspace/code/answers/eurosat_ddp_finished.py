#!/usr/bin/env python3

import argparse
import time

# add os import so we can get at environment variables w/ os.environ.get()
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud

import torchvision
import torchvision.transforms.v2 as transforms

from torch.profiler import profile, record_function, ProfilerActivity

# add torch DDP import - import as DDP
from torch.nn.parallel import DistributedDataParallel as DDP

# import torch distributed as dist for barrier, process_group operations, reductions
import torch.distributed as dist


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

# apply the model on validation data for accuracy metrics
def test(model, test_loader, loss_fn, device):
    total_labels = 0
    correct_labels = 0
    loss_total = 0
    model.eval()
    with torch.no_grad():
        for images, labels in test_loader:
            # Transfering images and labels to GPU if available
            labels = labels.to(device)
            images = images.to(device)

            # Forward pass
            outputs = model(images)
            loss = loss_fn(outputs, labels)

            # Extracting predicted label, and computing validation loss and validation accuracy
            predictions = torch.max(outputs, 1)[1]
            total_labels += len(labels)
            correct_labels += (predictions == labels).sum()
            loss_total += loss

    v_accuracy = correct_labels / total_labels
    v_loss = loss_total / len(test_loader)

    return v_accuracy, v_loss


def main(args):
    # define torch device
    # get LOCAL_RANK with os.environ.get to set the GPU appropriately
    local_rank = int(os.environ.get("LOCAL_RANK", default="0"))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    # set up process group
    # you'll need RANK (as the global rank) and WORLD_SIZE to pass init_process_group
    # and local_rank for device_id
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    global_rank = int(os.environ.get('RANK', '0'))

    dist.init_process_group(backend="nccl" if torch.cuda.is_available() else "mpi",
                            world_size=world_size,
                            rank=global_rank,
                            device_id=device)

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

    # only need to download data once per node but everyone needs to know information about the dataset
    if local_rank == 0:
        print("Downloading data if needed")
        dataset = torchvision.datasets.EuroSAT(root="./data", download=True, transform=transform)
    # wait for everyone
    dist.barrier()
    if local_rank != 0:
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

    # define train_sampler and test_sampler from tud.distributed.DistributedSampler
    # remember to specifiy num_replicas (from WORLD_SIZE) and rank
    train_sampler = tud.distributed.DistributedSampler(train_dataset, num_replicas=world_size, rank=global_rank)
    test_sampler = tud.distributed.DistributedSampler(test_dataset, num_replicas=world_size, rank=global_rank)

    # define loaders
    # add a parameter sampler=train_sampler or sampler=test_sampler in place of shuffle=True
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8, drop_last=True, sampler=train_sampler
    )
    
    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=8, drop_last=True, sampler=test_sampler
    )

    # instantiate model
    net = Net(num_classes).to(device)

    # optional wrap model with nn.SyncBatchNorm.convert_sync_batchnorm to sync the batch norms across microbatches
    net = nn.SyncBatchNorm.convert_sync_batchnorm(net)
    # wrap model with nn.parallel.DistributedDataParallel.   You'll have to provide device_ids=[local_rank])
    net = DDP(net, device_ids=[local_rank])
    
    # define loss, optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=args.base_lr, momentum=args.momentum)

    # only need to see this once - just have global rank == 0 print this
    if global_rank == 0:
        print(f"Beginning training: {args.epochs} epochs")

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 on_trace_ready=torch.profiler.tensorboard_trace_handler('/workspace/logs/eurosat_ddp_multigpu'),
                 profile_memory=True) as prof:
        # training loop
        total_time = 0
        for epoch in range(args.epochs):
            running_loss = 0.0
            t0 = time.time()

            # set the epoch in train_sampler so that the seed is different each time
            train_sampler.set_epoch(epoch)

            for i, data in enumerate(trainloader, 0):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data[0].to(device), data[1].to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward + backward + optimize
                outputs = net(inputs)

                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            # synchronize with barrier at the end of each epoch, or else 
            # stats will be wrong 
            dist.barrier()

            # timing
            epoch_time = time.time() - t0
            total_time += epoch_time

            # output metrics at the end of each epoch
            images_per_sec = torch.tensor(len(trainloader) * args.batch_size / epoch_time).to(device)
            v_accuracy, v_loss = test(net, testloader, criterion, device)

            # the stats we just calculated were per process; we should combine these into one
            # for v_accuracy, f_loss, these are already tensors, and we can use
            # dist.distributed.all_reduce(..., op=dist.ReduceOp.AVG) to average these
            # for images_per_sec we can just add them (op=dist.ReduceOp.SUM)
            dist.all_reduce(v_accuracy, op=dist.ReduceOp.AVG)
            dist.all_reduce(v_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(images_per_sec, op=dist.ReduceOp.SUM)
    
            # We don't need to see this line for each process (especially now that we've combined the results)
            # Just have rank 0 print this out
            if global_rank == 0:
                print(
                    f"Epoch = {epoch:2d}: Cumulative Time = {total_time:5.3f}, Epoch Time = {epoch_time:5.3f}, Images/sec = {images_per_sec:5.3f}, Validation Loss = {v_loss:5.3f}, Validation Accuracy = {v_accuracy:5.3f}"
                )
            prof.step()

    if global_rank == 0:
        # we don't need to have all ranks print finished training
        print("Finished Training")
    
        # we should only have one rank save the model
        save_path = "./eurosat_net.pth"
        torch.save(net.state_dict(), save_path)

    # get rid of the process group
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='EuroSAT training example',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--batch-size', type=int, default=32, help='input batch size for training')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--base-lr', type=float, default=0.01, help='learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    args = parser.parse_args()

    main(args)
