import torch
import torch.nn as nn
import torch.nn.functional as F
from constants import log_interval


class Net(nn.Module):

    def __init__(self, in_channels=3, out_channels=25):
        """
        Init Network with
        Two convolutional layers and two Fully Connected layers
        """
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(8704, 500)
        self.fc2 = nn.Linear(500, 250)
        self.fc3 = nn.Linear(250, 100)
        self.fc4 = nn.Linear(100, out_channels)

    def forward(self, x):
        """
        Forward operation:
        Max pool function for conv1 network
        Dropout and Max pool for conv2 network
        Apply three fully connected networks
        with dropout on first network
        Activation function - relu
        """
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # print(x.shape)
        try:
            x = x.view(-1, 8704)
        except:
            print(x.shape[1] * x.shape[2] * x.shape[3])
            raise Exception
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)
        return F.log_softmax(x)


def train(model, device, train_loader, optimizer, epoch, logger=None):
    """Training given model on some dataset"""
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            if logger:
                logger.INFO('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                           100. * batch_idx / len(train_loader), loss.item()))


def test(model_name, model, device, test_loader, logger = None, update_max = 0):
    global MAX
    """Testing and outputing the accuracy of the given mode"""
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
    test_loss /= len(test_loader.dataset)
    score = 100. * correct / len(test_loader.dataset)
    if logger:
        logger.info('\n{} accuracy results: Accuracy: ({:.2f}%)\n'.format(
            model_name,
            score))