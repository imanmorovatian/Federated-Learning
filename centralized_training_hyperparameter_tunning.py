import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
from torchvision import datasets, transforms

import optuna

def get_data():
  '''This function download train and test datasets and return them'''

  train_set = datasets.EMNIST(
      root='data', split='byclass',train=True, download=True,
      transform=transforms.Compose([transforms.ToTensor()])
                           )

  test_set = datasets.EMNIST(
      root='data', split='byclass', train=False, download=True,
      transform=transforms.Compose([transforms.ToTensor()])
                          )
  
  return train_set, test_set

class CNN(nn.Module):
    '''This class defines the CNN described in the instructions'''

    def __init__(self, in_channels=1, num_classes=62):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2) 
        )

        self.fc1 = nn.Sequential(
            nn.Linear(64*7*7, 2048),
            nn.ReLU()
        )

        self.fc2 = nn.Sequential(
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
  '''This function perform the train process and returns accuracy and 
  loss of train'''
  
  samples = 0
  cumulative_loss = 0
  cumulative_accuracy = 0

  net.train()

  for images, targets in data_loader:

    images = images.to(device)
    targets = targets.to(device)

    # forward
    outputs = net(images)
    loss = loss_function(outputs, targets)

    # backward
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    cumulative_loss += loss.item() * images.shape[0]

    _, predicted = outputs.max(1)
    cumulative_accuracy += predicted.eq(targets).sum().item()

    samples += images.shape[0]

  return cumulative_loss / samples, cumulative_accuracy / samples * 100

def validation_or_test(data_loader, net, loss_function, device):
  '''This function perform the validation process and returns accuracy and 
  loss of validation'''

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

def training_process(optuna_trial, train_set):
    '''Thie function orchestrate the training process. It employs utitily functions
    defined above and sets different hyperparameters.'''

    net = CNN()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(device)

    training_size = int(len(train_set) * 0.8) # 80% of train set is for training and 20% for validation
    train_subset, val_subset = random_split(train_set, [training_size, len(train_set) - training_size])
    
    batch_size = optuna_trial.suggest_categorical("batch_size", [16, 32, 64, 128])
    train_loader = torch.utils.data.DataLoader(train_subset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_subset, batch_size=batch_size, shuffle=True)

    loss_function = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        net.parameters(),
        lr = optuna_trial.suggest_loguniform("lr", 1e-4, 1),
        momentum = optuna_trial.suggest_loguniform("momentum", 0.9, 0.99),
        weight_decay = optuna_trial.suggest_loguniform("weight_decay", 1e-5, 1e-1),
        nesterov = optuna_trial.suggest_categorical("nesterov", [False, True])
    )

    epochs = optuna_trial.suggest_categorical("epochs", [50, 100, 200])

    for epoch in range(epochs):
      train_loss, train_accuracy = train(train_loader, net, loss_function, optimizer, device)      
      val_loss, val_accuracy = validation_or_test(val_loader, net, loss_function, device)

    return val_accuracy # this is the accuracy of the last epoch on the validation set

def objective(trial, train_set):
  return training_process(trial, train_set)



train_set, test_set = get_data()

study = optuna.create_study(direction="maximize")
study.optimize(lambda trial: objective(trial, train_set), n_trials=50)

print('Best trial:')
best_trial = study.best_trial

print('    The Best Validation Accuracy: ', best_trial.value)

print('         Params:')
for key, value in best_trial.params.items():
  print(f'            {key} --> {value}')

final_model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
final_model.to(device)

batch_size = best_trial.params['batch_size']
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

loss_function = nn.CrossEntropyLoss()

optimizer = optim.SGD(
    final_model.parameters(),
    lr = best_trial.params['lr'],
    momentum = best_trial.params['momentum'],
    weight_decay = best_trial.params['weight_decay'],
    nesterov = best_trial.params['nesterov']
    )

epochs = best_trial.params['epochs']

for epoch in range(epochs):
  train_loss, train_accuracy = train(train_loader, final_model, loss_function, optimizer, device)

print('*** End of the training of the final model ***')

test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)
test_loss, test_accuracy = validation_or_test(test_loader, final_model, nn.CrossEntropyLoss(), device)
print(f'Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}')


path = ''
torch.save(final_model.state_dict(), path)