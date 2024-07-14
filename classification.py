#!/usr/bin/env python
# coding: utf-8

# In[2]:


# Importing libraries. 

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.notebook import tqdm

# To avoid non-essential warnings 
import warnings
warnings.filterwarnings('ignore')

from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Dataset path. You should change the dataset path to the location that you place the data.
data_dir = '/Users/Summerzhao/Document/animal/dataset/dataset' 
classes = os.listdir(data_dir)


# In[4]:


# Performing Image Transformations. 
##Hints: Data Augmentation can be applied here. Have a look on RandomFlip, RandomRotation...
train_transform = transforms.Compose([
            transforms.Resize(112),
            transforms.RandomHorizontalFlip(),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize((0.488), (0.2172)),
        ])


# In[5]:


# Checking the dataset training size.
dataset = ImageFolder(data_dir, transform=train_transform)
print('Size of training dataset :', len(dataset))


# In[6]:


# Viewing one of images shape.
img, label = dataset[100]
print(img.shape)


# In[7]:


# Preview one of the images..
def show_image(img, label):
    print('Label: ', dataset.classes[label], "("+str(label)+")")
    plt.imshow(img.permute(1,2,0))


# In[8]:


show_image(*dataset[200])


# In[9]:


# Setting seed so that value won't change everytime. 
# Splitting the dataset to training, validation, and testing category.
torch.manual_seed(10)
val_size = len(dataset)//20
test_size = len(dataset)//10
train_size = len(dataset) - val_size - test_size


# In[10]:


# Random Splitting. 
train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
len(train_ds), len(val_ds),len(test_ds)  


# In[11]:


batch_size = 16
train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_ds, batch_size, num_workers=2, pin_memory=True)
test_loader = DataLoader(test_ds, batch_size, num_workers=2, pin_memory=True)


# In[12]:


# Multiple images preview. 
for images, labels in train_loader:
    fig, ax = plt.subplots(figsize=(18,10))
    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(images, nrow=16).permute(1, 2, 0))
    break


# In[13]:


# Baseline model class for training and validation purpose. Evaluation metric function - Accuracy.
def accuracy(output, target, topk=(1,)):
  
   with torch.no_grad():
       maxk = 3
       batch_size = target.size(0)

       # st()
       _, pred = output.topk(maxk, 1, True, True)
       pred = pred.t()
       # st()
       # correct = pred.eq(target.view(1, -1).expand_as(pred))
       # correct = (pred == target.view(1, -1).expand_as(pred))
       correct = (pred == target.unsqueeze(dim=0)).expand_as(pred)



       correct_3 = correct[:3].reshape(-1).float().sum(0, keepdim=True)

       return correct_3.mul_(1.0 / batch_size)
#def accuracy(outputs, labels):
#   _, preds = torch.max(outputs, dim=1)
 #  return torch.tensor(torch.sum(preds == labels).item() / len(preds))

# class ImageClassificationBase(nn.Module):
#     def training_step(self, batch):
#         images, labels = batch 
#         out = self(images)                  # Generate predictions
#         loss = F.cross_entropy(out, labels) # Calculate loss, Hints: the loss function can be changed to improve the accuracy
#         return loss
   
#     def validation_step(self, batch):
#         images, labels = batch 
#         out = self(images)                    # Generate predictions
#         loss = F.cross_entropy(out, labels)   # Calculate loss
#         acc = accuracy(out, labels, (5))           # Calculate accuracy
#         return {'val_loss': loss.detach(), 'val_acc': acc}
       
#     def validation_epoch_end(self, outputs):
#         batch_losses = [x['val_loss'] for x in outputs]
#         epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
#         batch_accs = [x['val_acc'] for x in outputs]
#         epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
#         return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
   
#     def epoch_end(self, epoch, result):
#         print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
#             epoch, result['train_loss'], result['val_loss'], result['val_acc']))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ImageClassificationBase(nn.Module):
   def training_step(self, batch):
       images, labels = batch 
       out = self(images)                  # Generate predictions
       loss = F.cross_entropy(out, labels) # Calculate loss
       return loss
   
   def validation_step(self, batch):
       images, labels = batch 
       out = self(images)                    # Generate predictions
       loss = F.cross_entropy(out, labels)   # Calculate loss
       acc = self.accuracy(out, labels)      # Calculate accuracy
       return {'val_loss': loss.detach(), 'val_acc': acc}
       
   def validation_epoch_end(self, outputs):
       batch_losses = [x['val_loss'] for x in outputs]
       epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
       batch_accs = [x['val_acc'] for x in outputs]
       epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
       return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
   
   def epoch_end(self, epoch, result):
       print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
           epoch, result['train_loss'], result['val_loss'], result['val_acc']))
       
   def accuracy(self, outputs, labels):
       _, preds = torch.max(outputs, dim=1)
       return torch.tensor(torch.sum(preds == labels).item() / len(preds))


# In[14]:


# To check wether Google Colab GPU has been assigned/not. 

def get_default_device():
   """Pick GPU if available, else CPU"""
   if torch.cuda.is_available():
       return torch.device('cuda')
   else:
       return None
   
def to_device(data, device):
   """Move tensor(s) to chosen device"""
   if isinstance(data, (list,tuple)):
       return [to_device(x, device) for x in data]
   return data.to(device, non_blocking=True)

class DeviceDataLoader():
   """Wrap a dataloader to move data to a device"""
   def __init__(self, dl, device):
       self.dl = dl
       self.device = device
       
   def __iter__(self):
       """Yield a batch of data after moving it to device"""
       for b in self.dl: 
           yield to_device(b, self.device)

   def __len__(self):
       """Number of batches"""
       return len(self.dl)


# In[15]:


device = get_default_device()
device
train_loader = DeviceDataLoader(train_loader, device)
val_loader = DeviceDataLoader(val_loader, device)
test_loader = DeviceDataLoader(test_loader, device)


# In[16]:


input_size = 3*112*112
output_size = 151


# In[17]:


# Convolutional Network - Baseline
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvolutionalNetwork(ImageClassificationBase):
    def __init__(self, num_classes):
        super(ConvolutionalNetwork, self).__init__()
        self.num_classes = num_classes
        
        # Load a pre-trained MobileNetV2 model
        self.mobilenet = models.mobilenet_v2(pretrained=True)
        
        
        num_ftrs = self.mobilenet.classifier[1].in_features
        self.mobilenet.classifier[1] = nn.Linear(num_ftrs, self.num_classes)
        
    def forward(self, x):
        x = self.mobilenet(x)
        return F.log_softmax(x, dim=1)


# In[18]:


# Model print
num_classes = 151
model = ConvolutionalNetwork(num_classes)
#model.cuda()


# In[19]:


# We can check the input and the output shape
for images, labels in train_loader:
    out = model(images)
    print('images.shape:', images.shape)    
    print('out.shape:', out.shape)
    print('out[0]:', out[0])
    break


# In[20]:


train_dl = DeviceDataLoader(train_loader, device)
val_dl = DeviceDataLoader(val_loader, device)
to_device(model, device)


# In[21]:


def fit(epochs, model, train_loader, val_loader, optimizer, device):
    model.to(device)
    history = []
    for epoch in range(epochs):
        # Training Phase
        model.train()
        train_losses = []
        for batch in train_loader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
        
        # Validation Phase
        result = evaluate(model, val_loader, device)
        result['train_loss'] = torch.tensor(train_losses).mean().item()
        model.epoch_end(epoch, result)
        history.append(result)
    return history

def evaluate(model, val_loader, device):
    model.eval()
    outputs = []
    with torch.no_grad():
        for batch in val_loader:
            output = model.validation_step(batch)
            outputs.append(output)
    return model.validation_epoch_end(outputs)


# In[22]:


model = to_device(model, device)


# In[23]:


import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

num_epochs = 10
lr = 0.0005

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_classes = 151  
model = ConvolutionalNetwork(num_classes).to(device)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=lr)

# Train the model
history = fit(epochs=num_epochs, model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, device=device)

# Evaluate the model
history = [evaluate(model, val_loader, device)]
print(history)


# In[24]:


# # Hints: The following parameters can be changed to improve the accuracy
# print(test_size)
# num_epochs = 10
# opt_func = torch.optim.Adam
# lr = 0.0005


# In[39]:


# def plot_accuracies(history):
#     accuracies = [x['val_acc'] for x in history]
#     plt.plot(range(1, len(accuracies) + 1), accuracies, '-x')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.title('Accuracy vs. No. of epochs (Transfer Learning)')
#     plt.show()
    
# def plot_losses(history):
#     train_losses = [x.get('train_loss') for x in history]
#     val_losses = [x['val_loss'] for x in history]
#     plt.plot(train_losses, '-bx')
#     plt.plot(val_losses, '-rx')
#     plt.xlabel('epoch')
#     plt.ylabel('loss')
#     plt.legend(['Training', 'Validation'])
#     plt.title('Loss vs. No. of epochs')
#     plt.show()

history = [
    {'train_loss': 3.0678, 'val_loss': 1.9649, 'val_acc': 0.4792},
    {'train_loss': 1.6598, 'val_loss': 1.5899, 'val_acc': 0.5559},
    {'train_loss': 1.1747, 'val_loss': 1.4137, 'val_acc': 0.6288},
    {'train_loss': 0.9070, 'val_loss': 1.4002, 'val_acc': 0.6184},
    {'train_loss': 0.7031, 'val_loss': 1.4725, 'val_acc': 0.6271},
    {'train_loss': 0.6091, 'val_loss': 1.4805, 'val_acc': 0.6507},
    {'train_loss': 0.5680, 'val_loss': 1.3613, 'val_acc': 0.6740},
    {'train_loss': 0.4652, 'val_loss': 1.4134, 'val_acc': 0.6882},
    {'train_loss': 0.4558, 'val_loss': 1.3751, 'val_acc': 0.6795},
    {'train_loss': 0.4404, 'val_loss': 1.4283, 'val_acc': 0.7045},
    {'val_loss': 1.3702694177627563, 'val_acc': 0.7038194537162781}
]

# Re-plot the accuracy and loss graphs
plot_accuracies(history)
plot_losses(history)


# In[ ]:


evaluate(model, test_loader)


# ##FLOPs

# In[29]:


#The code from https://cloudstor.aarnet.edu.au/plus/s/PcSc67ZncTSQP0E can be used to count flops
#Download the code.
get_ipython().system('wget -c https://cloudstor.aarnet.edu.au/plus/s/hXo1dK9SZqiEVn9/download')
get_ipython().system('mv download FLOPs_counter.py')
#!rm -rf download

import torch
import warnings
warnings.filterwarnings("ignore")

def print_model_parm_flops(model, input, detail=False):
  list_conv = []

  def conv_hook(self, input, output):

      # batch_size, input_channels, input_time(ops) ,input_height, input_width = input[0].size()
      # output_channels,output_time(ops) , output_height, output_width = output[0].size()

      kernel_ops = (self.in_channels / self.groups) * 2 - 1  # add operations is one less to the mul operations
      for i in self.kernel_size:
          kernel_ops *= i
      bias_ops = 1 if self.bias is not None else 0

      params = kernel_ops + bias_ops
      flops = params * output[0].nelement()

      list_conv.append(flops)

  list_linear = []

  def linear_hook(self, input, output):
      weight_ops = (2 * self.in_features - 1) * output.nelement()
      bias_ops = self.bias.nelement()
      flops = weight_ops + bias_ops
      list_linear.append(flops)

  list_bn = []

  def bn_hook(self, input, output):
      # (x-x')/σ one sub op and one div op
      # and the shift γ and β
      list_bn.append(input[0].nelement() / input[0].size(0) * 4)

  list_relu = []

  def relu_hook(self, input, output):
      # every input's element need to cmp with 0
      list_relu.append(input[0].nelement() / input[0].size(0))

  list_pooling = []

  def max_pooling_hook(self, input, output):
      # batch_size, input_channels, input_height, input_width = input[0].size()
      # output_channels, output_height, output_width = output[0].size()

      # unlike conv ops. in pool layer ,if the kernel size is a int ,self.input will be a int,not a tuple.
      # so we need to deal with this problem
      if isinstance(self.kernel_size, tuple):
          kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
      else:
          kernel_ops = self.kernel_size * self.kernel_size
          if len(output[0].size()) > 3:  # 3D max pooling
              kernel_ops *= self.kernel_size
      flops = kernel_ops * output[0].nelement()
      list_pooling.append(flops)

  def avg_pooling_hook(self, input, output):
      # cmp to max pooling ,avg pooling has an additional sub op
      # unlike conv ops. in pool layer ,if the kernel size is a int ,self.input will be a int,not a tuple.
      # so we need to deal with this problem
      if isinstance(self.kernel_size, tuple):
          kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
      else:
          kernel_ops = self.kernel_size * self.kernel_size
          if len(output[0].size()) > 3:  # 3D  pooling
              kernel_ops *= self.kernel_size
      flops = (kernel_ops + 1) * output[0].nelement()
      list_pooling.append(flops)

  def adaavg_pooling_hook(self, input, output):
      kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
      kernel_ops = torch.prod(kernel)
      flops = (kernel_ops + 1) * output[0].nelement()
      list_pooling.append(flops)

  def adamax_pooling_hook(self, input, output):
      kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
      kernel_ops = torch.prod(kernel)
      flops = kernel_ops * output[0].nelement()
      list_pooling.append(flops)

  def foo(net):
      childrens = list(net.children())
      if not childrens:
          if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.Conv3d):
              net.register_forward_hook(conv_hook)
          if isinstance(net, torch.nn.Linear):
              net.register_forward_hook(linear_hook)
          if isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm3d):
              net.register_forward_hook(bn_hook)
          if isinstance(net, torch.nn.ReLU):
              net.register_forward_hook(relu_hook)
          if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.MaxPool3d):
              net.register_forward_hook(max_pooling_hook)
          if isinstance(net, torch.nn.AvgPool2d) or isinstance(net, torch.nn.AvgPool3d):
              net.register_forward_hook(avg_pooling_hook)
          if isinstance(net, torch.nn.AdaptiveAvgPool2d) or isinstance(net, torch.nn.AdaptiveAvgPool3d):
              net.register_forward_hook(adaavg_pooling_hook)
          if isinstance(net, torch.nn.AdaptiveMaxPool2d) or isinstance(net, torch.nn.AdaptiveMaxPool3d):
              net.register_forward_hook(adamax_pooling_hook)
          return
      for c in childrens:
          foo(c)

  foo(model)
  out = model(input)
  total_flops = sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling)
  print(' + Number of FLOPs: %.2fG' % (total_flops / 1e9))


# In[32]:


import torch
import warnings
warnings.filterwarnings("ignore")

def print_model_parm_flops(model, input, detail=False):
    list_conv = []

    def conv_hook(self, input, output):

        # batch_size, input_channels, input_time(ops) ,input_height, input_width = input[0].size()
        # output_channels,output_time(ops) , output_height, output_width = output[0].size()

        kernel_ops = (self.in_channels / self.groups) * 2 - 1  # add operations is one less to the mul operations
        for i in self.kernel_size:
            kernel_ops *= i
        bias_ops = 1 if self.bias is not None else 0

        params = kernel_ops + bias_ops
        flops = params * output[0].nelement()

        list_conv.append(flops)

    list_linear = []

    def linear_hook(self, input, output):
        weight_ops = (2 * self.in_features - 1) * output.nelement()
        bias_ops = self.bias.nelement()
        flops = weight_ops + bias_ops
        list_linear.append(flops)

    list_bn = []

    def bn_hook(self, input, output):
        # (x-x')/σ one sub op and one div op
        # and the shift γ and β
        list_bn.append(input[0].nelement() / input[0].size(0) * 4)

    list_relu = []

    def relu_hook(self, input, output):
        # every input's element need to cmp with 0
        list_relu.append(input[0].nelement() / input[0].size(0))

    list_pooling = []

    def max_pooling_hook(self, input, output):
        # batch_size, input_channels, input_height, input_width = input[0].size()
        # output_channels, output_height, output_width = output[0].size()

        # unlike conv ops. in pool layer ,if the kernel size is a int ,self.input will be a int,not a tuple.
        # so we need to deal with this problem
        if isinstance(self.kernel_size, tuple):
            kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
        else:
            kernel_ops = self.kernel_size * self.kernel_size
            if len(output[0].size()) > 3:  # 3D max pooling
                kernel_ops *= self.kernel_size
        flops = kernel_ops * output[0].nelement()
        list_pooling.append(flops)

    def avg_pooling_hook(self, input, output):
        # cmp to max pooling ,avg pooling has an additional sub op
        # unlike conv ops. in pool layer ,if the kernel size is a int ,self.input will be a int,not a tuple.
        # so we need to deal with this problem
        if isinstance(self.kernel_size, tuple):
            kernel_ops = torch.prod(torch.Tensor([self.kernel_size]))
        else:
            kernel_ops = self.kernel_size * self.kernel_size
            if len(output[0].size()) > 3:  # 3D  pooling
                kernel_ops *= self.kernel_size
        flops = (kernel_ops + 1) * output[0].nelement()
        list_pooling.append(flops)

    def adaavg_pooling_hook(self, input, output):
        kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
        kernel_ops = torch.prod(kernel)
        flops = (kernel_ops + 1) * output[0].nelement()
        list_pooling.append(flops)

    def adamax_pooling_hook(self, input, output):
        kernel = torch.Tensor([*(input[0].shape[2:])]) // torch.Tensor(list((self.output_size,))).squeeze()
        kernel_ops = torch.prod(kernel)
        flops = kernel_ops * output[0].nelement()
        list_pooling.append(flops)

    def foo(net):
        childrens = list(net.children())
        if not childrens:
            if isinstance(net, torch.nn.Conv2d) or isinstance(net, torch.nn.Conv3d):
                net.register_forward_hook(conv_hook)
            if isinstance(net, torch.nn.Linear):
                net.register_forward_hook(linear_hook)
            if isinstance(net, torch.nn.BatchNorm2d) or isinstance(net, torch.nn.BatchNorm3d):
                net.register_forward_hook(bn_hook)
            if isinstance(net, torch.nn.ReLU):
                net.register_forward_hook(relu_hook)
            if isinstance(net, torch.nn.MaxPool2d) or isinstance(net, torch.nn.MaxPool3d):
                net.register_forward_hook(max_pooling_hook)
            if isinstance(net, torch.nn.AvgPool2d) or isinstance(net, torch.nn.AvgPool3d):
                net.register_forward_hook(avg_pooling_hook)
            if isinstance(net, torch.nn.AdaptiveAvgPool2d) or isinstance(net, torch.nn.AdaptiveAvgPool3d):
                net.register_forward_hook(adaavg_pooling_hook)
            if isinstance(net, torch.nn.AdaptiveMaxPool2d) or isinstance(net, torch.nn.AdaptiveMaxPool3d):
                net.register_forward_hook(adamax_pooling_hook)
            return
        for c in childrens:
            foo(c)

    foo(model)
    out = model(input)
    total_flops = sum(list_conv) + sum(list_linear) + sum(list_bn) + sum(list_relu) + sum(list_pooling)
    print(' + Number of FLOPs: %.2fG' % (total_flops / 1e9))


# In[33]:


input = torch.randn(1, 3, 112, 112) 
#Get the network and its FLOPs
num_classes = 151
model = ConvolutionalNetwork(num_classes)
print_model_parm_flops(model, input, detail=False)


