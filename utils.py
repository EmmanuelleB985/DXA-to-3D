import torch
from tqdm import tqdm

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def train_epoch(model, train_loader, loss_fn, l1_lambda, optimizer, metric='mse', epoch=10):
    model.train()
    loss_train = AverageMeter()
    with tqdm(train_loader, unit="batch") as tepoch:
        for i in tepoch:
            
            inputs = i['dxa_img']
            targets = i['coord']

            if epoch is not None:
                tepoch.set_description(f"Epoch {epoch}")
            inputs = inputs.to(device)
            targets = targets.view(targets.size(0), -1)
            targets = targets.to(device)
            outputs = model(inputs)

            loss = loss_fn(outputs, targets.float().to(device))
            print("loss",loss)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            l1_norm = torch.tensor(0., requires_grad=True)
            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_norm = l1_norm + torch.linalg.norm(param)
                    
            loss = loss + l1_lambda * l1_norm

            loss_train.update(loss.item())
            tepoch.set_postfix(loss=loss_train.avg)
        return model, loss_train.avg


def validation(model, valid_loader, loss_fn, l1_lambda):
    model.eval()
    with torch.no_grad():
        loss_valid = AverageMeter()
        for i,line in enumerate(tqdm(valid_loader)):
            
            inputs = line['dxa_img'].to(device)
            targets = line['coord']
            targets = targets.view(targets.size(0), -1)
            targets = targets.to(device)

            outputs = model(inputs)

            loss = loss_fn(outputs, targets.float().to(device))
            
            l1_norm = torch.tensor(0., requires_grad=True)

            for name, param in model.named_parameters():
                if 'weight' in name:
                    l1_norm = l1_norm + torch.linalg.norm(param)
                    
            loss = loss + l1_lambda * l1_norm
        
            loss_valid.update(loss.item())

    return loss_valid.avg