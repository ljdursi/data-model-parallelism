import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tud
import torch.distributed as dist
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleGPipe, PipelineStage

import torchvision
import torchvision.transforms.v2 as transforms

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
def test(model, test_loader, device, rank, world_size, chunks, test_schedule, test_stage):
    correct_labels, total_labels, loss_total = 0, 0, 0.0

    # Set the test stage to evaluation mode
    test_stage.submod.eval()
    
    # Create criterion for validation loss
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data[0], data[1]
            
            if rank == 0:
                inputs = inputs.to(device)      
            if rank == world_size - 1:
                labels = labels.to(device)

            outputs = None
            
            if rank == 0:
                test_schedule.step(inputs)
            elif rank == world_size - 1:
                outputs = test_schedule.step()
                predictions = torch.max(outputs, 1)[1]
                total_labels += len(labels)
                correct_labels += (predictions == labels).sum().item()
                
                # Calculate validation loss
                loss_total += criterion(outputs, labels).item()
            else:
                test_schedule.step()

            # In validation, add this on the last rank
            if rank == world_size - 1 and i == 0:
                print(f"Model outputs shape: {outputs.shape}")
                print(f"Model outputs range: [{outputs.min():.3f}, {outputs.max():.3f}]")
                print(f"Sample outputs: {outputs[0][:5]}")  # First 5 logits of first sample
                

            dist.barrier()
    
    test_stage.submod.train()
            
    v_accuracy, v_loss = 0.0, 0.0
    if rank == world_size - 1:
        if total_labels > 0:
            v_accuracy = correct_labels / total_labels
        if len(test_loader) > 0:
            v_loss = loss_total / len(test_loader)

    return v_accuracy, v_loss


def main(args):
    # Initialize distributed environment
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # define torch device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")

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

    # define loaders
    trainloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    testloader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    # Instantiate the model
    net = Net(num_classes).to(device)

    # Define split points for pipeline parallelism
    # We will split the model into three stages:
    # Stage 0: conv_block1
    # Stage 1: conv_block2
    # Stage 2: conv_block3 -> global_avg_pool -> classifier
    split_spec = {
        "conv_block1": SplitPoint.END,
        "conv_block2": SplitPoint.END,
        # The third split point is implicit after conv_block3,
        # meaning conv_block3, global_avg_pool, and classifier
        # will be in the last stage.
    }

    print(f"{rank}: creating pipeline")

    # Example input for pipeline graph analysis
    # Assuming input images are 3x64x64
    example_input = torch.randn(args.batch_size // args.chunks, 3, 64, 64, device=device)

    # Create the pipeline
    pipe = pipeline(net, mb_args=(example_input,), split_spec=split_spec)

    # Build the pipeline stage for the current rank
    stage = pipe.build_stage(rank, device, dist.group.WORLD)
    test_stage = pipe.build_stage(rank, device, dist.group.WORLD)

    # define loss, optimizer
    criterion = nn.CrossEntropyLoss()

    # Attach to a schedule
    train_schedule = ScheduleGPipe(stage, n_microbatches=args.chunks, loss_fn=criterion)
    test_schedule = ScheduleGPipe(test_stage, n_microbatches=args.chunks)

    with torch.no_grad():
        dummy_input = torch.randn(args.batch_size, 3, 64, 64, device=device)
        if rank == 0:
            test_schedule.step(dummy_input)
        else:
            test_schedule.step()
        dist.barrier()

    # The optimizer should only optimize parameters on the current stage
    optimizer = optim.SGD(stage.submod.parameters(), lr=args.base_lr, momentum=args.momentum)

    if rank == 0:
        print(f"Beginning training: {args.epochs} epochs")

    # training loop
    total_time = 0
    local_loss = 0
    nlosses = 0
    for epoch in range(args.epochs):
        running_loss = 0.0
        t0 = time.time()
        optimizer.zero_grad()

        for i, data in enumerate(trainloader, 0):
            inputs, labels = data[0], data[1]
            if rank == 0:
                inputs = inputs.to(device)
            elif rank == world_size - 1:
                labels = labels.to(device)

            # Forward + backward + optimize using the pipeline schedule
            # Only rank 0 provides the input data
            if rank == 0:
                train_schedule.step(inputs)
            elif rank == world_size - 1:
                losses = []
                train_schedule.step(target=labels, losses=losses)
                local_loss=sum(losses).item()
                running_loss += local_loss
                nlosses += 1
            else:
                train_schedule.step()

                        
        # Zero the parameter gradients for the current stage's optimizer
        optimizer.step()

        # Timing
        dist.barrier()
        epoch_time = time.time() - t0
        total_time += epoch_time

        v_accuracy, v_loss = test(net, testloader, device, rank, world_size, args.chunks, test_schedule, test_stage)
        if rank == world_size - 1:
            images_per_sec = len(trainloader) * args.batch_size / epoch_time
            train_loss = running_loss/nlosses
            print(
                f"Epoch = {epoch:2d}: Cumulative Time = {total_time:5.3f}, Epoch Time = {epoch_time:5.3f}, Images/sec = {images_per_sec:5.3f}, Train loss = {train_loss:5.3f} Validation Loss = {v_loss:5.3f}, Validation Accuracy = {v_accuracy:5.3f}"
            )

    if rank == 0:
        print("Finished Training")

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
