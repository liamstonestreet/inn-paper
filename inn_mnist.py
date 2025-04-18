import torch
from functionalities import dataloader as dl
from functionalities import tracker as tk
from architecture import INN as inn
from functionalities import MMD_autoencoder_loss as mmd_loss
from functionalities import trainer as tr
from functionalities import filemanager as fm
from functionalities import plot as pl
from functionalities import gpu 

# Pretraining setup
num_epoch = 10
batch_size = 128
latent_dim_lst = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64]
number_dev = 0
lr_init = 1e-3
l2_reg  = 1e-6
milestones = [8, 10]
modelname = 'mnist_INN_glow_com_bottleneck'
get_model = inn.mnist_inn_com

device = gpu.get_device(number_dev)
print(device)

trainset, testset, classes = dl.load_mnist()
trainloader, validloader, testloader = dl.make_dataloaders(trainset, testset, batch_size)

# Training
model = tr.train_bottleneck(num_epoch, get_model, 'l1', modelname, milestones, latent_dim_lst, trainloader, None, 
                            testloader, a_distr=0, a_disen=0, lr_init=lr_init, l2_reg=l2_reg, device=device, save_model=True)

# Reconstruction
pl.plot_diff_all(get_model, modelname, num_epoch, testloader, latent_dim_lst, device='cpu', num_img=1, grid_row_size=10, figsize=(30, 30), 
              filename=None, conditional=False)

# Reconstruction loss vs Botteneck size
_, l1_rec_test, _, _, _ = fm.load_variable('bottleneck_test_loss_{}'.format(modelname), modelname)
_, l1_rec_train, _, _, _ = fm.load_variable('bottleneck_train_loss_{}'.format(modelname), modelname)

pl.plot(latent_dim_lst, [l1_rec_train, l1_rec_test], 'bottleneck size', 'loss', ['train', 'test'], 'Test Reconstruction Loss History', '{}_bottleneck_History'.format(modelname)) 