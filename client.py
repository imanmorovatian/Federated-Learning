import copy
from collections import OrderedDict

import torch
from torch import optim, nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from utils.utils import HardNegativeMining, MeanReduction, freq_space_interpolation

class Client:

    def __init__(self, args, dataset, model, test_client=False):
        self.args = args
        self.dataset = dataset
        self.name = self.dataset.client_name
        self.model = model
        self.train_loader = DataLoader(self.dataset, batch_size=self.args.bs, shuffle=True, drop_last=True) if not test_client else None
        self.test_loader = DataLoader(self.dataset, batch_size=1, shuffle=False)
        self.criterion = nn.CrossEntropyLoss(ignore_index=255, reduction='none')
        self.reduction = HardNegativeMining() if self.args.hnm else MeanReduction()

        self.r_mu = nn.Parameter(torch.zeros(62, 1024))
        self.r_sigma = nn.Parameter(torch.ones(62, 1024))
        self.C = nn.Parameter(torch.ones([]))

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

    def run_epoch(self, optimizer, bank, train_client_names):
        """
        This method locally trains the model with the dataset of the client. It handles the training at mini-batch level
        :param cur_epoch: current epoch of training
        :param optimizer: optimizer used for the local training
        """

        self.model.train() # set model to training mode

        if self.args.dgm == 'feddg':
            data = []
            for images, labels in self.train_loader:
                batch_ctrs = []
                batch_ctrs_labels = []
                for image, label in zip(images, labels):
                    amps = bank.get_amps(self.name, label, train_client_names)
                    for amp in amps:
                        counterpart = freq_space_interpolation(image, amp, L=0.003, ratio=0.4)
                        batch_ctrs.append(counterpart)
                        batch_ctrs_labels.append(label)

                data.append((images, labels, batch_ctrs, batch_ctrs_labels))

            for images, labels, ctrs, ctrs_labels in data:
                # meta-train
                # compute the first part of the total lost and updated paramaters

                images = images.to('cuda')
                labels = labels.to('cuda')
                
                optimizer.zero_grad()

                outputs = self._get_outputs(images)
                loss_part1 = self.criterion(outputs, labels)
                loss_part1 = self.reduction(loss_part1, labels)

                grads = torch.autograd.grad(loss_part1, self.model.parameters(), retain_graph=True)

                meta_step_size = 1e-3
                clip_value = 100 
                updated_params = OrderedDict((name, param - torch.mul(meta_step_size, torch.clamp(grad, -clip_value, clip_value))) for
                                                    ((name, param), grad) in zip(self.model.named_parameters(), grads))
                
                # meta-test

                ctrs_torch_gpu = [torch.from_numpy(ctr).to('cuda') for ctr in ctrs]
                ctrs_labels_gpu = [label.to('cuda') for label in ctrs_labels]

                ctrs = torch.stack(ctrs_torch_gpu, dim=0).float()
                ctrs_labels = torch.stack(ctrs_labels_gpu, dim=0).long()

                for name, param in self.model.state_dict().items():
                    if name in updated_params.keys():
                        param.data.copy_(updated_params[name])

                outputs_with_updated_params = self.model(ctrs)
                loss_part2 = self.criterion(outputs_with_updated_params, ctrs_labels)
                loss_part2 = self.reduction(loss_part2, ctrs_labels)

                total_loss = loss_part1 + loss_part2
                
                total_loss.backward()

        elif self.args.dgm == 'fedsr':
            for cur_step, (images, labels) in enumerate(self.train_loader):
                images = images.to('cuda')
                labels = labels.to('cuda')
                
                optimizer.zero_grad()

                z, z_mu, z_sigma = self.model.featurize(images)

                outputs = self._get_outputs(images)
                loss = self.criterion(outputs, labels)
                loss = self.reduction(loss, labels)

                regL2R = torch.zeros_like(loss)
                regCMI = torch.zeros_like(loss)

                regL2R = z.norm(dim=1).mean()
                loss = loss + self.args.l2r * regL2R

                r_sigma_softplus = F.softplus(self.r_sigma)
                r_mu = self.r_mu[labels.cpu()]
                r_mu = r_mu.cuda()
                r_sigma = r_sigma_softplus[labels.cpu()]
                r_sigma = r_sigma.cuda()
                z_mu_scaled = z_mu * self.C
                z_sigma_scaled = z_sigma * self.C
                regCMI = torch.log(r_sigma) - torch.log(z_sigma_scaled) + (z_sigma_scaled**2+(z_mu_scaled-r_mu)**2)/(2*r_sigma**2) - 0.5
                regCMI = regCMI.sum(1).mean()
                loss = loss + self.args.cmi*regCMI          
                loss.backward()

        else:
            for cur_step, (images, labels) in enumerate(self.train_loader):
                images = images.to('cuda')
                labels = labels.to('cuda')

                optimizer.zero_grad()
                outputs = self._get_outputs(images)
                loss = self.criterion(outputs, labels)
                loss = self.reduction(loss, labels)
                loss.backward()

        optimizer.step()

    def train(self, bank, train_client_names):
        """
        This method locally trains the model with the dataset of the client. It handles the training at epochs level
        (by calling the run_epoch method for each local epoch of training)
        :return: length of the local dataset, copy of the model parameters
        """

        CNN_WEIGHT_DECAY = 1e-5
        FC_WEIGHT_DECAY = 1e-3
        local_optimizer = optimizer = optim.SGD([
            {'params': self.model.layer1.parameters(), 'weight_decay': CNN_WEIGHT_DECAY},
            {'params': self.model.layer2.parameters(), 'weight_decay': CNN_WEIGHT_DECAY},
            {'params': self.model.fc1.parameters(), 'weight_decay': FC_WEIGHT_DECAY},
            {'params': self.model.fc2.parameters(), 'weight_decay': FC_WEIGHT_DECAY}
            ],
            lr=self.args.lr,
            momentum=self.args.m
        )

        for epoch in range(self.args.num_epochs):
            self.run_epoch(epoch, local_optimizer, bank, train_client_names)

        return len(self.dataset), copy.deepcopy(self.model.state_dict())
    
    def test(self, metric):
        """
        This method tests the model on the local dataset of the client.
        :param metric: StreamMetric object
        """
        
        samples = 0
        cumulative_loss = 0
        self.model.eval() # set model to evaluation mode
        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):

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