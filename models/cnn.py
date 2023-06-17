import torch.nn as nn
import torch.nn.functional as F 
import torch.distributions as distributions


class CNN(nn.Module):

    def __init__(self, in_channels=1, num_classes=62):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.fc1 = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(64*7*7, 2048),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
            nn.Dropout(0.8),
            nn.Linear(2048, num_classes)
        )


    def forward(self, X):
        out = self.layer1(X)
        out = self.layer2(out)

        out = out.reshape(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        
        return out
    

    def featurize(self, X, num_samples=1):
        temp = self.layer1(X)
        temp = self.layer2(temp)

        temp = temp.reshape(temp.size(0), -1)

        z_params = self.fc1(temp)

        z_mu = z_params[:,:1024]
        z_sigma = F.softplus(z_params[:,1024:])
        z_dist = distributions.Independent(distributions.normal.Normal(z_mu,z_sigma),1)
        z = z_dist.rsample([num_samples]).view([-1,1024])
            
        return z, z_mu, z_sigma