from torch import nn, optim
import sys
sys.path.append('.')
from dataset.Dataset import PairsDataset
import config
from model import RegressionModel,RegressionModel_Transformer
import matplotlib.pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import os
from utils import train_epoch, validation
from config import *

# Instantiate training, validation, and test sets
train_set = PairsDataset(set_type='train',root = './UKBiobank/')
val_set = PairsDataset(set_type='val',root = './UKBiobank/')
test_set  = PairsDataset(set_type='test',root = './UKBiobank/')


# Define dataloader
train_loader = DataLoader(train_set, batch_size=config.train_batch_size, shuffle=True)
valid_loader = DataLoader(val_set, batch_size=config.valid_batch_size)
test_loader = DataLoader(test_set, batch_size=config.valid_batch_size)

# Create a folder to save checkpoints
root = "./"
checkpoints = 'checkpoints'
path = os.path.join(root, checkpoints)
try:
    os.makedirs(path, exist_ok=True)
except OSError as error:
    print(error)
 
# Define Model, loss and optimizer 
model = RegressionModel_Transformer(input_dim=3, output_nodes=209*6, model_name='resnet', pretrain_weights='IMAGENET1K_V2')
loss_fn = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=config.wd)

best_loss = torch.inf
before_model_path = None

loss_train_hist = []
loss_valid_hist = []

acc_train_hist = []
acc_valid_hist = []

writer = SummaryWriter()

l1_lambda = 10e-4

for epoch in range(config.num_epochs):

    # Train
    model, loss_train = train_epoch(model,
                                    train_loader,
                                    loss_fn,
                                    l1_lambda,
                                    optimizer,
                                    epoch)
    # Validation
    loss_valid = validation(model,
                            valid_loader,
                            loss_fn,l1_lambda)

    loss_train_hist.append(loss_train)
    loss_valid_hist.append(loss_valid)


    if loss_valid < best_loss:
        best_loss = loss_valid
        if before_model_path is not None:
            os.remove(before_model_path)
        before_model_path = f'./checkpoints_sagittal/epoch:{epoch}-loss_valid-points:{best_loss:.3}.pt'
        torch.save(model.state_dict(), before_model_path)
        print(f'\nModel saved in epoch: {epoch}')

    writer.add_scalar('Loss/train', loss_train, epoch)
    writer.add_scalar('Loss/test', loss_valid, epoch)
    
    if epoch % 5 == 0:
        print()
        print(f'Train: loss={loss_train:.3}')
        print(f'Valid: loss={loss_valid:.3}')


writer.close()

# Plots: Plot learning curves
plt.plot(range(config.num_epochs), loss_train_hist, 'r-', label='Train')
plt.plot(range(config.num_epochs), loss_valid_hist, 'b-', label='Validation')
plt.ylim([0,1])
plt.xlabel('Epoch')
plt.ylabel('loss')
plt.grid(True)
plt.legend()
plt.savefig("./results/loss_2D_sagittal.png")

print('TRAINING DONE')
