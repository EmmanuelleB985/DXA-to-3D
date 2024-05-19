import torch.cuda

lr = 0.00001
wd = 0.0001
device = 'cuda' if torch.cuda.is_available() else 'cpu'
num_epochs = 2000
train_batch_size = 8
valid_batch_size = 8