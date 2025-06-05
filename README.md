# Data and Model Parallelism

This set of notebooks gives an overview of how to use several data and model parallelism approaches with PyTorch. We cover:

* Distributed data parallelism (DDP), using the torch.nn.parallel.DistributedDataParallel module.
* Pipeline parallelism, using the torch.distributed.pipeline module.
* Fully Sharded Data Parallel (FSDP), using the torch.distributed.fsdp module.
* and to support that, we also discuss torchrun in some detail.

The notebooks are designed to be run in an environment where multiple GPUs are available. The distributed examples make by default make use of 3 or 4 GPUs.

You can get started by running `docker compose up`.

The notebooks are structured as follows:

* [0_Start_Here](workspace/0_Start_Here.ipynb): You are here, the table of contents
* [1_EuroSAT_single_gpu](workspace/1_EuroSAT_single_gpu.ipynb): A single GPU interactive example of training a model on the EuroSAT dataset.
* [2_Torchrun_and_distributed](workspace/2_Torchrun_and_distributed.ipynb): An introduction to the `torchrun` command we'll be using to run distributed training jobs; it works for single-node jobs (which is what we'll be doing) but also for multi-node jobs.
* [3_Data_and_model_parallelism](workspace/3_Data_and_model_parallelism.ipynb): An introduction to data and model parallelism.  We'll talk about the difference between data and model parallelism, and give very brief introductinos to the idea behind each.
* [4_Distributed_data_parallel](workspace/4_Distributed_data_parallel.ipynb): An introduction to distributed data parallelism, using the `torch.nn.parallel.DistributedDataParallel` module.   We'll walk through the steps of taking our single-GPU EuroSAT example and converting it to use DDP.  
* [5_Pipelining](workspace/5_Pipelining.ipynb): An introduction to pipeline parallelism, using the `torch.distributed.pipeline` module.  We'll walk through the steps of taking our single-GPU EuroSAT example and converting it to use pipelining.
* [6_Fully_sharded_data_parallel](workspace/6_Fully_sharded_data_parallel.ipynb): An introduction to fully sharded data parallelism, using the `torch.distributed.fsdp` module.  We'll walk through the steps of taking our single-GPU EuroSAT example and converting it to use FSDP.
* [7_Next_steps_and_other_resources](workspace/7_Next_steps_and_other_resources.ipynb): A summary of what we've learned, and pointers to other resources for learning more about distributed training in PyTorch.
