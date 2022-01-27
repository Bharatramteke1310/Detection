# -*- coding: utf-8 -*-
"""
Created on Mon Jan 24 16:00:30 2022

@author: HP
"""
import pandas as pd
import matplotlib.pyplot as plt


dataset = pd.read_csv("sample-submission.csv")
dataset.head()

import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt

train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
train_dir = os.path.join('C:\\Users\\HP\\Downloads\\crack-identification-ce784a-2020-iitk (1)\\train')
test_dir = os.path.join("C:\\Users\\HP\Downloads\\crack-identification-ce784a-2020-iitk (1)\\test")

classes = ['cracked','uncracked']

data_transform = transforms.Compose([transforms.RandomResizedCrop(224),
                                      transforms.ToTensor()])

train_data = datasets.ImageFolder(train_dir, transform=data_transform)
#test_data = datasets.ImageFolder(test_dir, transform=data_transform)

# print out some data stats
print('Num training images: ', len(train_data))
#print('Num test images: ', len(test_data))

batch_size = 32
num_workers=0

# prepare data loaders
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, 
                                           num_workers=num_workers, shuffle=True)
#test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, 
                                        #  num_workers=num_workers, shuffle=True)
                                        
# Visualize some sample data

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
images = images.numpy() # convert images to numpy for display

# plot the images in the batch, along with the corresponding labels
fig = plt.figure(figsize=(25, 4))
for idx in np.arange(20):
    ax = fig.add_subplot(2, 20/2, idx+1, xticks=[], yticks=[])
    plt.imshow(np.transpose(images[idx], (1, 2, 0)))
    ax.set_title(classes[labels[idx]])
                                   
vgg16 = models.vgg16(pretrained=True)

# print out the model structure
print(vgg16)               
                                      
print(vgg16.classifier[6].in_features) 
print(vgg16.classifier[6].out_features)  
                                        
# Freeze training for all "features" layers
for param in vgg16.features.parameters():
    param.requires_grad = False 
    
import torch.nn as nn

n_inputs = vgg16.classifier[6].in_features

# add last linear layer (n_inputs -> 5 flower classes)
# new layers automatically have requires_grad = True
last_layer = nn.Linear(n_inputs, len(classes))

vgg16.classifier[6] = last_layer

# if GPU is available, move the model to GPU
if train_on_gpu:
    vgg16.cuda()

# check to see that your last layer produces the expected number of outputs
print(vgg16.classifier[6].out_features)
#print(vgg16)

import torch.optim as optim
# specify loss function (categorical cross-entropy)
criterion = nn.CrossEntropyLoss()

# specify optimizer (stochastic gradient descent) and learning rate = 0.001
optimizer = optim.SGD(vgg16.classifier.parameters(), lr=0.005)


n_epochs = 30

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    
    ###################
    # train the model #
    ###################
    # model by default is set to train
    for batch_i, (data, target) in enumerate(train_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.cuda(), target.cuda()
        # clear the gradients of all optimized variables
        optimizer.zero_grad()
        # forward pass: compute predicted outputs by passing inputs to the model
        output = vgg16(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # backward pass: compute gradient of the loss with respect to model parameters
        loss.backward()
        # perform a single optimization step (parameter update)
        optimizer.step()
        # update training loss 
        train_loss += loss.item()
        
        if batch_i %32 == 31:    # print training loss every specified number of mini-batches
            print('Epoch %d, Batch %d loss: %.16f' %
                  (epoch, batch_i + 1, train_loss / 32))
            train_loss = 0.0
            
data_dir = 'crack-detection-ce784a-iitk/'
test_dir = os.path.join(data_dir, 'test/')

                                        
                                        
                                        
                                        
                                        
                                        
                                        
