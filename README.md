# ENAS
ENAS_tutorial run
ENAS is an advanced version of the NAS algorithm which was developed primarily to improve the efficiency of NAS by forcing 
all child models to share weights to bypass the training of each child model from scratch to convergence. 
The objective of ENAS is to find a neural network by searching for a subgraph within a larger computational graph. 
This automated model design method shares the parameters (weights and biases) among all the child nodes.


How to run ??
To run the code, the input files need to be changed in the labels_data variable defined with the path to fetch the input file. 
The number of tasks, number of class and classes per tasks are defined as 10,100 and 10 respectively and need not be modified. 
The number of training epochs for the training defined as epochs can be altered to get an improved accuracy.

Hyperparameters used for the experimentation:
    num_tasks = 10
    num_class = 100
    class_per_task = 10
    M = 12
    rigidness_coff = 2.5
    dataset = "CIFAR"
    epochs = 1
    L = 9
    N = 4
    lr = 0.001
    train_batch = 16
    test_batch = 16
    workers = 16
    resume = False
    arch = "res-18"
    start_epoch = 0
    evaluate = False
    schedule = [20, 30, 40]
    gamma = 0.5
