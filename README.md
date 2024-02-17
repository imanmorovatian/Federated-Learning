# An Introduction to Federated Learning (FL)

This repository contains the code of the project of **Machine Learning and Deep Learning (MLDL)** course at **Politecnico di Torino** (Second semester 2022/2023).

The project investigates the preliminaries of FL and introduces some challenges, such as statistical heterogeneity, the availability of clients for participation, and the generalization ability to unknown target data (e.g., a new client). Subsequently, proposed solutions for these challenges are implemented. Specifically, the **power-of-choice** strategy is implemented to address the availability of clients, while the **FedSR** and **FedDG** methods are implemented to tackle the generalization ability challenge.

## Setup Environment
Install environment with conda (preferred): 
```bash 
conda env create -f mldl23fl.yml
```

## How to run
The ```main.py``` orchestrates training. All arguments need to be specified through the ```args``` parameter (options can be found in ```utils/args.py```). For example,

```bash
python main.py --dataset femnist --model resnet18 --num_rounds 1000 --num_epochs 5 --clients_per_round 10 
```

Also, ```centralized_training.py``` and ```centralized_training_hyperparameter_tunning.py``` are provided to find the best hyperparameters of the model that will be used in clients.
