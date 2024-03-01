from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import utils


logger = utils.setup_logger()


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device
    ) -> float:
    """
    Train the model for one epoch.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - loader (DataLoader): The DataLoader for training data.
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
    - criterion (nn.Module): The loss function.
    - device (torch.device): The device (CPU or GPU) on which to perform training.

    Returns:
    float: Average training loss for the epoch.
    """

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
def validate_model(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device
    ) -> tuple[float, float]:
    """
    Validate the model.

    Parameters:
    - model (nn.Module): The PyTorch model to be validated.
    - loader (DataLoader): The DataLoader for validation data.
    - criterion (nn.Module): The loss function.
    - device (torch.device): The device (CPU or GPU) on which to perform validation.

    Returns:
    tuple: A tuple containing total loss and accuracy.
    - float: Total validation loss.
    - float: Accuracy of the model on the validation set.
    """

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
        model: nn.Module,
        num_epochs: int,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        early_stopping: float,
        device: torch.device,
        model_dir: str,
        model_name: str = 'model'
    ) -> tuple[list[float], list[float], list[float]]:
    """
    Train and validate the model for a specified number of epochs.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - num_epochs (int): Number of training epochs.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): Learning rate scheduler.
    - train_loader (DataLoader): DataLoader for training data.
    - test_loader (DataLoader): DataLoader for validation data.
    - optimizer (torch.optim.Optimizer): Model optimizer.
    - criterion (nn.Module): Loss function.
    - early_stopping (float): Accuracy threshold for early stopping.
    - device (torch.device): Device (CPU or GPU) on which to perform training.
    - model_dir (str): Directory to save the trained model.
    - model_name (str): Name to use when saving the model (default is 'model').

    Returns:
    tuple: A tuple containing training loss history, validation loss history, and validation accuracy history.
    - list[float]: Training loss history for each epoch.
    - list[float]: Validation loss history for each epoch.
    - list[float]: Validation accuracy history for each epoch.
    """
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
