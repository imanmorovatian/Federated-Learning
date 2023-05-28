import copy
import torch

from torch import optim, nn
from collections import defaultdict
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction


class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) \
            if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

    def __str__(self):
        return self.name

    @staticmethod
    def update_metric(metric, outputs, labels):
        _, prediction = outputs.max(dim=1)
        labels = labels.cpu().numpy()
        prediction = prediction.cpu().numpy()
        metric.update(labels, prediction)

    def _get_outputs(self, images):
        if self.args.model == 'deeplabv3_mobilenetv2':
            return self.model(images)['out']
        if self.args.model == 'resnet18' or self.args.model == 'cnn':
            return self.model(images)

    def run_epoch(self, cur_epoch, optimizer):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """
        self.model.train() # set model to training mode

        for cur_step, (images, labels) in enumerate(self.train_loader):
            # TODO: missing code here!
            images = images.to('cuda')
            labels = labels.to('cuda')

            optimizer.zero_grad()
            outputs = self._get_outputs(images)
            loss = self.criterion(outputs, labels)
            loss = self.reduction(loss, labels)
            loss.backward()
            optimizer.step()

    def train(self):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """
        # TODO: missing code here! -----------> ##### DONE :)
        local_optimizer = optim.SGD(self.model.parameters(), lr=self.args.lr, momentum=self.args.m, weight_decay=self.args.wd)
        for epoch in range(self.args.num_epochs):
            # TODO: missing code here! -----------> ##### DONE :)
            self.run_epoch(epoch, local_optimizer)
        return len(self.dataset), copy.deepcopy(self.model.state_dict())

    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        # TODO: missing code here! -----------> ##### DONE :)
        samples = 0
        cumulative_loss = 0
        self.model.eval() # set model to evaluation mode
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                # TODO: missing code here! -----------> ##### DONE :)
                images = images.to('cuda')
                labels = labels.to('cuda')

                outputs = self._get_outputs(images)
                self.update_metric(metric, outputs, labels)

                # calculating loss
                loss = self.criterion(outputs, labels)
                loss = self.reduction(loss, labels)
                cumulative_loss += loss.item() * images.shape[0]
                samples += images.shape[0]

        return cumulative_loss / samples