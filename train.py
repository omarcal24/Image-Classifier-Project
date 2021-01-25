import torch
from torch import nn, optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import time
from collections import OrderedDict
import numpy as np
import PIL
from PIL import Image
import matplotlib.pyplot as plt
import shutil, argparse

# The user can choose the some options such as data directory, GPU/CPU and architecture

parser = argparse.ArgumentParser(description='Training a model which categorizes flowers')
parser.add_argument('--data_directory', type=str, default='flowers', help='The directory where images come from')
parser.add_argument('--save_directory', type=str, default='checkpoint.pth', help='The directory in which the checkpoint will be stored')
parser.add_argument('--gpu', type=bool, default=False, help='Use GPU or CPU')
parser.add_argument('--lr', type=float, default=0.001, help='Choose the learining')
parser.add_argument('--epochs', type=int, default=3, help='Choose the epochs number for the model')

# Storing inputs
args = parser.parse_args()   

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'
    
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])
    
valid_transforms = transforms.Compose([transforms.Resize(256),
                                       transforms.CenterCrop(224),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(), 
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

# TODO: Load the datasets with ImageFolder
train_datasets = datasets.ImageFolder(train_dir, transform=train_transforms)
valid_datasets = datasets.ImageFolder(valid_dir, transform=valid_transforms)
test_datasets = datasets.ImageFolder(test_dir, transform=test_transforms)

# TODO: Using the image datasets and the trainforms, define the dataloaders
trainloader = torch.utils.data.DataLoader(train_datasets, batch_size=128, shuffle=True)
validloader = torch.utils.data.DataLoader(valid_datasets, batch_size=64)
testloader = torch.utils.data.DataLoader(test_datasets, batch_size=64)

    
# Choosing the model
model = models.vgg11(pretrained=True)

# Freezing parameters
for param in model.parameters():
    param.requires_grad = False
    
    
# Building the model classifier    
model.classifier = nn.Sequential(nn.Linear(25088, 4096),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(4096, 1024),
                                 nn.ReLU(),
                                 nn.Dropout(0.2), 
                                 nn.Linear(1024, 102), 
                                 nn.LogSoftmax(dim=1))

# Defining criterion
criterion = nn.NLLLoss()

# Defining optimizer witht the learning rate the user chose
optimizer = optim.Adam(model.classifier.parameters(), lr=args.lr)

#Cuda - CPU, model to device if running on GPU
if args.gpu != 'CPU':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
if args.gpu == 'CPU':
    device = torch.device("cpu")

# Training the model
epochs = args.epochs
steps = 0
running_loss = 0
print_every = 50

for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        
        #moving inputs and label tensors to device
        
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            model.eval()
            test_loss = 0
            accuracy = 0
            
            with torch.no_grad():
                for inputs, labels in validloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    #accuracy
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every: .3f}.. "
                  f"Test loss: {test_loss/len(validloader): .3f}.. "
                  f"Test accuracy: {accuracy/len(validloader): .3f}.. ")
            
            running_loss = 0
            model.train()
            
# TODO: Do validation on the test set
correct = 0
total = 0

with torch.no_grad():
    model.eval()
    for inputs, labels in testloader:
        inputs, labels = inputs.to(device), labels.to(device)
        output = model(inputs)
        _, predicted = torch.max(output.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracyof the network: %d %%' % (100 * correct / total))
    
# TODO: Save the checkpoint 
model.class_to_idx = train_datasets.class_to_idx
structure ='vgg11'

checkpoint = {
    'structure': structure,
    'epochs': epochs,
    'classifier': model.classifier,
    'state_dict': model.state_dict(),
    'optimizer_dict': optimizer.state_dict(),
    'criterion': criterion,
    'class_to_idx': model.class_to_idx
}

torch.save(checkpoint, 'checkpoint.pth')

