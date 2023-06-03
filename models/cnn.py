import torch.nn as nn 


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