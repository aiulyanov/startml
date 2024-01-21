from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils


logger = utils.setup_logger()


def train_epoch(model, loader: DataLoader, optimizer, criterion, device):

    model.train()
    running_loss = 0.0

    with tqdm(loader, desc='Training', leave=True) as tqdm_iter:
        for i, (images, labels) in enumerate(tqdm_iter):
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            if i % 10 == 0:
                tqdm_iter.set_postfix({'loss': running_loss / (i + 1)})
        
    return running_loss / len(loader)


@torch.inference_mode()
def validate_model(model, loader, criterion, device):

    model.eval()
    correct = 0
    total = 0

    running_loss = 0.0

    with tqdm(loader, desc='Validation', leave=True) as tqdm_iter:
        for i, (images, labels) in enumerate(tqdm_iter):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            _, predicted = torch.max(outputs.data, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            running_loss += loss.item()

            if i % 10 == 0:
                tqdm_iter.set_postfix({'loss': running_loss / (i + 1)})

    accuracy = correct / total
    total_loss = running_loss / len(loader)
    logger.info(f"Test accuracy: {accuracy * 100:.2f}% \n")

    return total_loss, accuracy


def fit_model(
        model, 
        num_epochs, 
        scheduler, 
        train_loader, 
        test_loader, 
        optimizer, 
        criterion,
        early_stopping, 
        device,
        model_dir,
        model_name='model'
    ):

    train_loss_hist, val_loss_hist = [], []
    val_acc_hist = []

    best_acc = 0.0

    for epoch in range(num_epochs):
        logger.info(f"Epoch {epoch + 1} / {num_epochs}")

        train_loss = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device)

        val_loss, val_acc = validate_model(
            model=model, loader=test_loader, criterion=criterion, device=device)
        
        train_loss_hist.append(train_loss)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), f'{model_dir}/{model_name}.pt')
        
        if best_acc >= early_stopping:
            logger.info("Early stopping threshold reached, training completed")
            break
    
    scheduler.step()

    return train_loss_hist, val_loss_hist, val_acc_hist
