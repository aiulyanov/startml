import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

import data
import train
import model


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.abspath(f'{os.getcwd()}/DL/3_classification')
DATA_DIR = os.path.abspath(f'{BASE_DIR}/data')
MODEL_DIR = os.path.abspath(f'{BASE_DIR}/model')
CIFAR10_DIR = os.path.abspath(f'{DATA_DIR}/CIFAR10')
MODEL_NAME = 'resnet18'

INPUT_SIZE = (32, 32)
OUTPUT_SIZE = 10
BATCH_SIZE = 64  
LR = 0.01              
NUM_EPOCHS = 20
EARLY_STOPPING = None

transform_train = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.8, 1.2), shear=10),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

transform_val = transforms.Compose([
        transforms.Resize(INPUT_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])


def main():
    train_dataset, test_dataset = data.get_dataset(datasets.CIFAR10, transform_train, transform_val, CIFAR10_DIR)
    train_loader, test_loader = data.get_dataloaders(train_dataset, test_dataset, BATCH_SIZE)

    net = model.create_resnet18().to(DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=1, epochs=NUM_EPOCHS)

    train_loss_hist, train_acc_hist, val_loss_hist, val_acc_hist = train.fit_model(
        model=net, 
        num_epochs=NUM_EPOCHS, 
        scheduler=scheduler, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        early_stopping=EARLY_STOPPING,
        device=DEVICE,
        model_dir=MODEL_DIR,
        model_name=MODEL_NAME
    )

    train_dataset_tta, test_dataset_tta = data.get_dataset(datasets.CIFAR10, transform_train, transform_val, CIFAR10_DIR, tta=True)
    _, test_loader_tta = data.get_dataloaders(train_dataset_tta, test_dataset_tta, BATCH_SIZE)
    total_preds, accuracy = train.predict_tta(net, test_loader_tta, device=DEVICE, iterations=5)

if __name__ == '__main__':
    main()
