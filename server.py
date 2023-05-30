import copy
from collections import OrderedDict

import numpy as np
import torch


class Server:

    def __init__(self, args, train_clients, test_clients, model, metrics, wandb):
        self.args = args
        self.train_clients = train_clients
        self.test_clients = test_clients
        self.model = model
        self.metrics = metrics
        self.model_params_dict = copy.deepcopy(self.model.state_dict())
        self.wandb = wandb

        wandb.init(
        project = 'narenji', 
        name = 'First Phase Attempt: 1', # set the name of experiment
        config={
            'architecture': self.args.model,
            'dataset': self.args.dataset,
            'epochs': self.args.num_epochs,
            'batch_size': self.args.bs,
            'learning_rate': self.args.lr,
            'momentum': self.args.m,
            'weight_decay': self.args.wd
            }
        )

    def select_clients(self):
        num_clients = min(self.args.clients_per_round, len(self.train_clients))
        return np.random.choice(self.train_clients, num_clients, replace=False)

    def train_round(self, clients):
        """
            This method trains the model with the dataset of the clients. It handles the training at single round level
            :param clients: list of all the clients to train
            :return: model updates gathered from the clients, to be aggregated
        """
        updates = []
        for i, c in enumerate(clients):
            # TODO: missing code here!
            (len, parm) = c.train()
            updates.append((len, parm))
        return updates

    def aggregate(self, updates):
        """
        This method handles the FedAvg aggregation
        :param updates: updates received from the clients
        :return: aggregated parameters
        """
        # TODO: missing code here!
        output = updates[0][1].copy()
        weighted_sum = 0
        weights = 0
        for key in output:
            weighted_sum = 0
            for i in updates:
                weighted_sum += i[1][key] * i[0]
                weights += i[0]
            output[key] = weighted_sum / weights
          
        return output

    def train(self):
        """
        This method orchestrates the training the evals and tests at rounds level
        """
        for r in range(self.args.num_rounds):
            # TODO: missing code here!
            clients_temp = self.select_clients()
            updates_temp = self.train_round(clients_temp)
            self.aggregate(updates_temp) # update self.model

            if r % 20 == 0:
                print('Round: ', r+1)
                self.eval_train()
                self.test()

    def eval_train(self):
        """
        This method handles the evaluation on the train clients
        """
        # TODO: missing code here!
        loss = 0
        for client in self.train_clients:
            loss += client.test(self.metrics['eval_train'])

        loss = loss / len(self.train_clients)
        overal_acc = self.metrics['eval_train'].get_results()['Overall Acc']
        mean_acc = self.metrics['eval_train'].get_results()['Mean Acc']
        print('****** Train Results ******')
        print('Loss: ', round(loss, 2))
        print('Total Accuracy: ', round(overal_acc, 2))
        print('Mean Accuracy: ', round(mean_acc, 2))
        print()
        
        self.wandb.log({
            'Train Loss': loss,
            'Train Overal Accuracy': overal_acc,
            'Train Mean Accuracy': mean_acc,
        })

    def test(self):
        """
            This method handles the test on the test clients
        """
        # TODO: missing code here!
        loss = 0
        for client in self.test_clients:
            loss += client.test(self.metrics['test'])

        loss = loss / len(self.test_clients)
        overal_acc = self.metrics['test'].get_results()['Overall Acc']
        mean_acc = self.metrics['test'].get_results()['Mean Acc']
        print('****** Test Results ******')
        print('Loss: ', round(loss, 2))
        print('Total Accuracy: ', round(overal_acc, 2))
        print('Mean Accuracy: ', round(mean_acc, 2))
        print()

        self.wandb.log({
            'Test Loss': loss,
            'Test Overal Accuracy': overal_acc,
            'Test Mean Accuracy': mean_acc,
        })
