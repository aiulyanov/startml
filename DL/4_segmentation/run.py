import os
import torch
import torch.nn as nn
import torch.optim as optim
import albumentations as A
from albumentations.pytorch import ToTensorV2
from model import UNET


import data_utils
import dataset
import train


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BASE_DIR = os.path.abspath(f'{os.getcwd()}/DL/4_segmentation')
DATA_DIR = os.path.abspath(f'{BASE_DIR}/data')
MODEL_DIR = os.path.abspath(f'{BASE_DIR}/model')
OXFORD_DIR = os.path.abspath(f'{DATA_DIR}/OxfordIIIPets')
CARVANA_DIR = os.path.abspath(f'{DATA_DIR}/Carvana')
TRAIN_IMAGE_DIR = os.path.abspath(f'{CARVANA_DIR}/train_images')
TRAIN_MASK_DIR = os.path.abspath(f'{CARVANA_DIR}/train_masks')
VAL_IMAGE_DIR = os.path.abspath(f'{CARVANA_DIR}/val_images')
VAL_MASK_DIR = os.path.abspath(f'{CARVANA_DIR}/val_masks')
CARVANA_SAVE_DIR = os.path.abspath(f'{CARVANA_DIR}/saved_images')
MODEL_NAME = 'unet'

HEIGHT = 160
WIDTH = 240
BATCH_SIZE = 32  
LR = 0.001              
NUM_EPOCHS = 20

train_transforms = A.Compose(
        [
            A.Resize(height=HEIGHT, width=WIDTH),
            A.Rotate(limit=35, p=1.0),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.1),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ],
    )

val_transforms = A.Compose(
        [
            A.Resize(height=HEIGHT, width=WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2()
        ],
    )

def main():
    # train_dataset, test_dataset = data.get_dataset(datasets.OxfordIIITPet, transform_train, transform_val, OXFORD_DIR)
    train_dataset = dataset.CarvanaDataset(image_dir=TRAIN_IMAGE_DIR, mask_dir=TRAIN_MASK_DIR, transform=train_transforms)
    test_dataset = dataset.CarvanaDataset(image_dir=VAL_IMAGE_DIR, mask_dir=VAL_MASK_DIR, transform=val_transforms)
    train_loader, test_loader = data_utils.get_dataloaders(train_dataset, test_dataset, BATCH_SIZE)
    net = UNET(in_channels=3, out_channels=1).to(DEVICE)
    

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR, steps_per_epoch=1, epochs=NUM_EPOCHS)
    scaler = torch.cuda.amp.GradScaler()

    train_loss_hist, train_acc_hist, train_dice_hist, val_loss_hist, val_acc_hist, val_dice_hist = train.fit_model(
        model=net, 
        num_epochs=NUM_EPOCHS, 
        scheduler=scheduler, 
        train_loader=train_loader, 
        test_loader=test_loader, 
        optimizer=optimizer, 
        criterion=criterion, 
        device=DEVICE,
        scaler=scaler,
        model_dir=MODEL_DIR,
        model_name=MODEL_NAME
    )

    train.save_predict(net, test_loader, device=DEVICE, pred_save_dir=CARVANA_SAVE_DIR)

if __name__ == '__main__':
    main()