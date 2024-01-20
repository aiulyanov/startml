import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader

def create_model():
    
    model = nn.Sequential(
        nn.Linear(100, 10),
        nn.ReLU(),
        nn.Linear(10, 1)
    )

    return model


def train(model: nn.Module, data_loader: DataLoader, optimizer: Optimizer, loss_fn):
    model.train()  
    total_loss = 0.0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        optimizer.zero_grad()  
        
        outputs = model(inputs)
        
        loss = loss_fn(outputs, targets)
        total_loss += loss.item()
        
        loss.backward()
        print(round(loss.item(), 5))
        
        optimizer.step()
        
    avg_loss = total_loss / len(data_loader)

    return avg_loss


def evaluate(model: nn.Module, data_loader: DataLoader, loss_fn):
    model.eval()  
    total_loss = 0.0

    with torch.no_grad(): 
        for inputs, targets in data_loader:
            outputs = model(inputs)
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()

    avg_loss = total_loss / len(data_loader)

    return avg_loss