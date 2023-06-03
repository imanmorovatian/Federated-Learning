import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import wandb


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


def train(data_loader, net, loss_function, optimizer, device):

  samples = 0
  cumulative_loss = 0
  cumulative_accuracy = 0

  net.train()

  for images, targets in data_loader:

    images = images.to(device)
    targets = targets.to(device)

    optimizer.zero_grad()

    # forward
    outputs = net(images)
    loss = loss_function(outputs, targets)

    # backward
    loss.backward()
    optimizer.step()
    
    cumulative_loss += loss.item() * images.shape[0]

    _, predicted = outputs.max(1)
    cumulative_accuracy += predicted.eq(targets).sum().item()

    samples += images.shape[0]

  return cumulative_loss / samples, cumulative_accuracy / samples * 100


def validation_or_test(data_loader, net, loss_function, device):

  samples = 0
  cumulative_loss = 0
  cumulative_accuracy = 0

  net.eval()

  with torch.no_grad():

    for images, targets in data_loader:

      images = images.to(device)
      targets = targets.to(device)

      outputs = net(images)
      loss = loss_function(outputs, targets)
      cumulative_loss += loss.item() * images.shape[0]

      _, predicted = outputs.max(1)
      cumulative_accuracy += predicted.eq(targets).sum().item()

      samples += images.shape[0]

  return cumulative_loss / samples, cumulative_accuracy / samples * 100


def plot_accuracy(train_accuracy, val_accuracy):
  plt.plot(train_accuracy, label='Train')
  plt.plot(val_accuracies, label='Validation')
  plt.xlabel('epoch')
  plt.ylabel('accuracy')
  plt.title('Train Accuracy Vs Validation Accuracy')
  plt.legend()


def plot_loss(train_loss, val_loss):
  plt.plot(train_loss, label='Train')
  plt.plot(val_loss, label='Validation')
  plt.xlabel('epoch')
  plt.ylabel('loss')
  plt.title('Train Loss Vs Validation Loss')
  plt.legend()


wandb.login()

ATTEMPT = 8 # always increament this variable by one and set the new value for that
            # for example if the values is 1, you can set it to 2, or if it is 2
            # you should set it for 3
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # use gpu if available
EPOCHS = 100
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
MOMENTUM = 0.9
CNN_WEIGHT_DECAY = 1e-5
FC_WEIGHT_DECAY = 1e-3

train_set = datasets.EMNIST(root='data', split='byclass',train=True, download=True,
                            transform=transforms.Compose([transforms.ToTensor()])
                           )

test_set = datasets.EMNIST(root='data', split='byclass', train=False, download=True,
                           transform=transforms.Compose([transforms.ToTensor()])
                          )

entire_trainset = torch.utils.data.DataLoader(train_set, shuffle=True)

split_train_size = int(0.8*(len(entire_trainset)))  # use 80% as train set
split_valid_size = len(entire_trainset) - split_train_size  # use 20% as validation set
train_set, val_set = torch.utils.data.random_split(train_set, [split_train_size, split_valid_size]) 

train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)

net = CNN().to(DEVICE)
nn.init.xavier_uniform_(net.layer1[0].weight)
nn.init.xavier_uniform_(net.layer2[0].weight)
nn.init.xavier_uniform_(net.fc1[1].weight)
nn.init.xavier_uniform_(net.fc2[1].weight)

loss_function = torch.nn.CrossEntropyLoss()

optimizer = optim.SGD(
    [
        {'params': net.layer1.parameters(), 'weight_decay': CNN_WEIGHT_DECAY},
        {'params': net.layer2.parameters(), 'weight_decay': CNN_WEIGHT_DECAY},
        {'params': net.fc1.parameters(), 'weight_decay': FC_WEIGHT_DECAY},
        {'params': net.fc2.parameters(), 'weight_decay': FC_WEIGHT_DECAY}
    ],
    lr=LEARNING_RATE,
    momentum=MOMENTUM
    )

wandb.init(
      # Set the project where this run will be logged
      project = 'narenji', 
      # Se the run name
      name = f'Centeralized Training attempt:{ATTEMPT}',

      # Track hyperparameters and run metadata
      config={
      'architecture': "CNN",
      'dataset': 'EMNIST',
      'epochs': EPOCHS,
      'batch_size': BATCH_SIZE,
      'learning_rate': LEARNING_RATE,
      'momentum': MOMENTUM,
      'weight_decay_cnn': CNN_WEIGHT_DECAY,
      'weight_decay_fc': FC_WEIGHT_DECAY
      }

      )

train_losses = []
train_accuracies = []

val_losses = []
val_accuracies = []

for epoch in range(EPOCHS):

  train_loss, train_accuracy = train(train_loader, net, loss_function, optimizer, DEVICE)
  train_losses.append(train_loss)
  train_accuracies.append(train_accuracy)

  val_loss, val_accuracy = validation_or_test(val_loader, net, loss_function, DEVICE)
  val_losses.append(val_loss)
  val_accuracies.append(val_accuracy)

  wandb.log({
    'Train Loss': train_loss,
    'Train Accuracy': train_accuracy,
    'Validation Loss': val_loss,
    'Validation Accuracy': val_accuracy
    })
  
  print(f'Epoch: {epoch+1}/{EPOCHS}')
  print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f} *** Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}')


plot_accuracy(train_accuracies, val_accuracies)

plot_loss(train_losses, val_losses)

test_loss, test_accuracy = validation_or_test(test_loader, net, loss_function, DEVICE)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')

wandb.log({"Test Accuracy": test_accuracy, "Test Loss": test_loss})

wandb.finish()

