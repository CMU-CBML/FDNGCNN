# %% [markdown]
# # Improved balance data
# 

# %%
import numpy as np
from dataset import H5Dataset
import matplotlib.pyplot as plt

dataset = H5Dataset('../dataset_for_ML/NG_single_CNN_restructured_11022022.hdf5')

x = dataset.data
y = dataset.target
print(f'data shape: {x.shape} | target shape: {y.shape}')

for i in range(1):
    id = np.random.randint(x.shape[0])
    plt.figure(figsize=(20, 4), dpi=80)
    plt.subplot(1,4,1)
    plt.imshow(x[id,0,...],cmap = "jet")
    plt.title(id)
    plt.colorbar()
    plt.subplot(1,4,2)
    plt.imshow(x[id,1,...],cmap = "jet")
    plt.colorbar()
    plt.subplot(1,4,3)
    plt.imshow(x[id,2,...],cmap = "jet")
    plt.colorbar()
    plt.subplot(1,4,4)
    plt.imshow(y[id,0,...],cmap = "jet")
    plt.colorbar()
    

# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils import data
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np
import data
from dataset import rdDataset_old
from dataset import H5Dataset
from model import rdcnn_2_larger
from math import log10

from tqdm import tqdm

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True
path = './data'

# Parameters

params = {'test_split': .25,
          'shuffle_dataset': True,
          'batchsize': 100,
          'testBatchsize': 100,
          'random_seed': 42,
          'numworkers':0,
          'pinmemory':True}
    
max_epoches = 100
learning_rate = 1e-4
drop_rate = 0.0

print('===> Loading datasets')
# Load All Dataset

# Creating data indices for training and validation splits:
training_data_loader, testing_data_loader = data.DatasetSplit(dataset, **params)

print('===> Building model')
model = rdcnn_2_larger(drop_rate).to(device)
# criterion = nn.MSELoss()
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,weight_decay=1e-5)

m = nn.Sigmoid()

def train(epoch):
    epoch_loss = 0
    for iteration, batch in enumerate(tqdm(training_data_loader), 1):
        input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)
        optimizer.zero_grad()
        loss = criterion(model(input), target)
        epoch_loss += loss.item()
        loss.backward()
        optimizer.step()
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch, epoch_loss / len(training_data_loader)
    
def test():
    avg_error = 0
    avg_loss = 0
    with torch.no_grad():
        for batch in tqdm(testing_data_loader):
            input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)

            prediction = model(input)
            tmp_error = 0
            for j in range(len(prediction)):
                tmp_error += torch.sqrt(torch.mean((m(prediction[j])-target[j])**2))/torch.max(target[j])
            avg_error += tmp_error / len(prediction)
            mse = criterion(prediction, target)
            avg_loss += mse
    print("===> Avg. Loss: {:.4f} ".format(avg_loss / len(testing_data_loader)))
    print("===> Avg. Error: {:.4f} ".format(avg_error / len(testing_data_loader)))
    return avg_loss / len(testing_data_loader),avg_error / len(testing_data_loader)

def checkpoint(epoch):
    model_out_path = "./checkpoint_3channel_singleNeuron_BCE/model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))
    
model.eval()  

# %%
L_train_loss = []
L_test_loss = []
L_test_error = []
for epoch in range(1, max_epoches + 1):
    train_loss = train(epoch)
    test_loss,test_error = test()
    checkpoint(epoch)
    L_train_loss.append(train_loss)
    L_test_loss.append(test_loss)
    L_test_error.append(test_error)
    print(f'###############################################################')

# %%
# model = torch.load('./checkpoint_3channel_singleNeuron/model_epoch_100.pth')
# model.eval()

# %%
import data
import importlib 
importlib.reload(data)
data.TestErrorPlot(model,device, testing_data_loader)

# %%
from matplotlib import pyplot as plt
prediction_L = []
input_L = []
target_L = []
i=0

with torch.no_grad():
    for batch in testing_data_loader:
        input, target = batch[0].to(device, torch.float), batch[1].to(device, torch.float)
        input_L.append(input)
        target_L.append(target)
        prediction = m(model(input))
        # prediction = model(input)
        prediction_L.append(prediction)
        i = i+1
        if i==10:
            break

# %%
for i in range(1):
    input = input_L[i].cpu().numpy()
    target = target_L[i]
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    for t in range(5):
        im = ax[t].imshow(target[t][0].cpu(),cmap = "jet")
        ax[t].axis('off')
        # ax[t].set_title("iteration = "+str(input[t,2,0,0]),size=10)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.27, 0.01, 0.47])
    fig.colorbar(im, cax=cbar_ax)
    
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    for t in range(5,10):
        im = ax[t-5].imshow(target[t][0].cpu(),cmap = "jet")
        ax[t-5].axis('off')
        # ax[t-5].set_title("iteration = "+str(input[t,2,0,0]),size=10)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.27, 0.01, 0.47])
    fig.colorbar(im, cax=cbar_ax)


plt.show()

# %%
for i in range(1):
    input = input_L[i].cpu().numpy()
    prediction = prediction_L[i]
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    for t in range(5):
        im = ax[t].imshow(prediction[t][0].cpu(),cmap = "jet")
        ax[t].axis('off')
        # ax[t].set_title("iteration = "+str(input[t,2,0,0]),size=10)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.27, 0.01, 0.47])
    fig.colorbar(im, cax=cbar_ax)
    
    fig, ax = plt.subplots(1,5, figsize=(20,5))
    for t in range(5,10):
        im = ax[t-5].imshow(prediction[t][0].cpu(),cmap = "jet")
        ax[t-5].axis('off')
        # ax[t-5].set_title("iteration = "+str(input[t,2,0,0]),size=10)
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.84, 0.27, 0.01, 0.47])
    fig.colorbar(im, cax=cbar_ax)


plt.show()

# %%
for i in range(1):
    input = input_L[i].cpu().numpy()
    target = target_L[i]
    prediction = prediction_L[i]
    for t in range(5):
        plt.figure(figsize=(20, 4), dpi=80)
        plt.subplot(1,5,1)
        plt.imshow(input[t][0],cmap = "jet")
        plt.colorbar()
        plt.title("Input Phi")  
        plt.subplot(1,5,2)
        plt.imshow(input[t][1],cmap = "jet")
        plt.colorbar()
        plt.title("Input theta")  
        plt.subplot(1,5,3)
        plt.imshow(input[t][2],cmap = "jet")
        plt.colorbar()
        plt.title(f'Input iteration {input[t,2,0,0]}')  
        plt.subplot(1,5,4)
        plt.imshow(prediction[t][0].cpu(),cmap = "jet")
        plt.colorbar()
        plt.title("Prediction")    
        plt.subplot(1,5,5)
        plt.imshow(target[t][0].cpu(),cmap = "jet")
        plt.colorbar()
        plt.title("Ground Truth Phi")    
    plt.show()

# %%
from data import ComputeTestError
dataset = H5Dataset('../dataset_for_ML/NG_single_CNN_restructured_11022022.hdf5')
print(f'dataset shape: {dataset.data.shape} | target shape: {dataset.data.shape}')

# %%
id = np.random.randint(182)*345
for j in range(182):
    id = j*345
# id = 6900
    print(f'picked rand data: {id}')
    input_1 = dataset.data[id,:,:,:]
    target_1 = dataset.target[id,0,:,:]
    # input_1[1,:,:] = np.random.rand(200,200)

    plt.figure(figsize=(30, 7), dpi=100)
    cols = 6
    plt_itvl = 6800
    for i in range(cols):

        with torch.no_grad():
            prediction_1 = m(model(torch.tensor(np.expand_dims(input_1,axis=0)).to(device, torch.float)))
            prediction_1=prediction_1.cpu()
            err = ComputeTestError(prediction_1[0,0,:,:].cpu(),torch.tensor(target_1).cpu())
            
        plt.subplot(2,cols,i+1)
        plt.imshow(prediction_1[0,0,:,:].numpy(),cmap='jet')
        plt.title(f'Prediction at iter {int(input_1[2,0,0]*35000)}')
        plt.colorbar()
        plt.subplot(2,cols,cols+i+1)
        plt.imshow(target_1,cmap='jet')
        plt.title(f'Ground truth at iter {int(input_1[2,0,0]*35000)}')
        plt.xlabel(f'MRE: {err*100:.2f}%')
        plt.colorbar()

        iterations = ((i+1)*plt_itvl)
        id += int(plt_itvl/100)

        input_1[2,:,:] = np.ones(input_1[2,:,:].shape)*iterations/35000
        target_1 = dataset.target[id,0,:,:]

    plt.savefig(f'./Figure/batchGen/results_bce_{j}.png')
    # plt.savefig('./Figure/results_bce_205.png')
    # plt.show()  

# %%



