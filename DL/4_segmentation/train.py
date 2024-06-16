from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
from model import UNET

import utils


logger = utils.setup_logger()


def train_epoch(
        model: nn.Module,
        loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
    ) -> float:
    """
    Train the model for one epoch.

    Parameters:
    - model (nn.Module): The PyTorch model to be trained.
    - loader (DataLoader): The DataLoader for training data.
    - optimizer (torch.optim.Optimizer): The optimizer for updating model parameters.
    - criterion (nn.Module): The loss function.
    - device (torch.device): The device (CPU or GPU) on which to perform training.
    - scaler (torch.cuda.amp.GradScaler): Need that to use float16 training. 

    Returns:
    float: Average training loss for the epoch.
    """

    model.train()
    running_loss = 0.0

    num_correct = 0
    num_pixels = 0
    dice_score = 0

    with tqdm(loader, desc='Training', leave=True) as tqdm_iter:
        for i, (images, targets) in enumerate(tqdm_iter):
            images, targets = images.to(device), targets.float().unsqueeze(1).to(device)
            
            # forward
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, targets)

                preds = torch.sigmoid(outputs)
                preds = (preds > 0.5).float()
                num_correct += (preds == targets).sum()
                num_pixels += torch.numel(preds)

                dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)


            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            if i % 10 == 0:
                tqdm_iter.set_postfix({'loss': round(running_loss, 4) / (i + 1)})
    
    total_loss = running_loss / len(loader)
    total_acc = num_correct / num_pixels
    total_dice = dice_score / len(loader)
    logger.info(
        f"Train loss: {total_loss:.4f} \
        | Train acc: {total_acc * 100:.2f}%  \
        | Dice score: {total_dice:.4f}"
    )

    return total_loss, total_acc, total_dice


@torch.inference_mode()
def validate_model(
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
    ) -> tuple[float, float]:
    """
    Validate the model.

    Parameters:
    - model (nn.Module): The PyTorch model to be validated.
    - loader (DataLoader): The DataLoader for validation data.
    - criterion (nn.Module): The loss function.
    - device (torch.device): The device (CPU or GPU) on which to perform validation.
    - pred_save_dir (str): Path to saved predictions.

    Returns:
    tuple: A tuple containing total loss and accuracy.
    - float: Total validation loss.
    - float: Accuracy of the model on the validation set.
    """

    model.eval()
    running_loss = 0.0

    num_correct = 0
    num_pixels = 0
    dice_score = 0

    with tqdm(loader, desc='Validation', leave=True) as tqdm_iter:
        for i, (images, targets) in enumerate(tqdm_iter):
            images, targets = images.to(device), targets.float().unsqueeze(1).to(device)

            outputs = model(images)
            loss = criterion(outputs, targets)
            running_loss += loss.item()

            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()
            num_correct += (preds == targets).sum()
            num_pixels += torch.numel(preds)

            dice_score += (2 * (preds * targets).sum()) / ((preds + targets).sum() + 1e-8)

            if i % 10 == 0:
                tqdm_iter.set_postfix({'loss': running_loss / (i + 1)})
            
            

    total_loss = running_loss / len(loader)
    total_acc = num_correct / num_pixels
    total_dice = dice_score / len(loader)
    logger.info(
        f"Val loss: {total_loss:.4f} \
        | Val acc: {total_acc * 100:.2f}% \
        | Dice score: {total_dice:.4f}"
    )

    return total_loss, total_acc, total_dice


def fit_model(
        model: nn.Module,
        num_epochs: int,
        scheduler: optim.lr_scheduler._LRScheduler,
        train_loader: DataLoader,
        test_loader: DataLoader,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        device: torch.device,
        scaler: torch.cuda.amp.GradScaler,
        model_dir: str,
        model_name: str = 'model',
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
    - device (torch.device): Device (CPU or GPU) on which to perform training.
    - scaler (torch.cuda.amp.GradScaler): Need that to use float16 training. 
    - model_dir (str): Directory to save the trained model.
    - model_name (str): Name to use when saving the model (default is 'model').

    Returns:
    tuple: A tuple containing training loss history, validation loss history, and validation accuracy history.
    - list[float]: Training loss history for each epoch.
    - list[float]: Validation loss history for each epoch.
    - list[float]: Validation accuracy history for each epoch.
    """
    train_loss_hist, train_acc_hist, train_dice_hist  = [], [], []
    val_loss_hist, val_acc_hist, val_dice_hist  = [], [], []

    best_loss = float('inf')

    for epoch in range(num_epochs):
        print()
        logger.info(f"Epoch {epoch + 1} / {num_epochs}")
        current_lr = optimizer.param_groups[0]['lr']
        logger.info(f"Learning rate: {round(current_lr, 5)}")

        train_loss, train_acc, train_dice = train_epoch(
            model=model, loader=train_loader, optimizer=optimizer, criterion=criterion, device=device, scaler=scaler)

        val_loss, val_acc, val_dice = validate_model(
            model=model, loader=test_loader, criterion=criterion, device=device)
        
        train_loss_hist.append(train_loss)
        train_acc_hist.append(train_acc)
        train_dice_hist.append(train_dice)
        val_loss_hist.append(val_loss)
        val_acc_hist.append(val_acc)
        val_dice_hist.append(val_dice)

        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), f'{model_dir}/{model_name}.pt')
        
        scheduler.step()

    return train_loss_hist, train_acc_hist, train_dice_hist, val_loss_hist, val_acc_hist, val_dice_hist


@torch.inference_mode()
def save_predict(model: nn.Module, loader: DataLoader, device: torch.device, pred_save_dir):
    model.eval()
    with tqdm(loader, desc='Prediction', leave=True) as tqdm_iter:
        for i, (images, targets) in enumerate(tqdm_iter):
            images = images.to(device)

            outputs = model(images)
            preds = torch.sigmoid(outputs)
            preds = (preds > 0.5).float()

            torchvision.utils.save_image(preds, f"{pred_save_dir}/{i}_pred.png")
            torchvision.utils.save_image(targets.unsqueeze(1), f"{pred_save_dir}/{i}.png")


@torch.inference_mode()
def predict_tta(model: nn.Module, loader: DataLoader, device: torch.device, iterations: int = 2):
    model.eval()
    total_outputs = []
    total_labels = []

    for i in range(iterations):
        current_output = []

        if not total_labels:
            for _, labels in loader:
                total_labels.extend(labels.tolist())                # CIFAR10 test: 156 x [64] + 1 x [16] == [10000]
        
        with tqdm(loader, desc=f'TTA Prediction {i + 1}', leave=True) as tqdm_iter:
            for images, _ in tqdm_iter:
                images = images.to(device)
                outputs = model(images)
                current_output.append(outputs.data)
                
        current_output = torch.vstack(current_output)               # CIFAR10 test: 156 x [64, 10] + 1 x [16, 10] == [10000, 10]
        total_outputs.append(current_output)


    total_outputs = torch.stack(total_outputs, dim=2)               # CIFAR10 test: iterations x 156 x [64, 10] + 1 x [16, 10] == [10000, 10, iterations]
    total_average = torch.mean(total_outputs, dim=2)
    total_preds = torch.argmax(total_average, 1)                    # CIFAR10 test: 156 x [64, 10] + 1 x [16, 10] == [10000, 10] - average per iteration
    total_labels = torch.Tensor(total_labels).to(device)

    correct = total_preds.eq(total_labels).sum().item()
    accuracy = round((correct / total_labels.shape[0]), 4)
    logger.info(f"TTA Test accuracy: {accuracy * 100:.2f}% \n")

    return total_preds, accuracy