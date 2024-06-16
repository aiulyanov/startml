import os
from PIL import Image
from torch.utils.data import Dataset
import numpy as np

import albumentations as A
from albumentations.pytorch import ToTensorV2


class CarvanaDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None) -> None:
        super().__init__()
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace('.jpg', '_mask.gif'))

        image = np.array(Image.open(img_path).convert('RGB'))
        mask = np.array(Image.open(mask_path).convert('L'), dtype=np.float32)
        mask[mask == 255.0] = 1.0

        if self.transform:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
    
        return image, mask


def test_dataset():
    train_image_dir = '/home/arthur/code/startml/DL/4_segmentation/data/train_images'
    train_mask_dir = '/home/arthur/code/startml/DL/4_segmentation/data/train_masks'
    val_image_dir = '/home/arthur/code/startml/DL/4_segmentation/data/val_images'
    val_mask_dir = '/home/arthur/code/startml/DL/4_segmentation/data/val_masks'

    train_transform = A.Compose(
        [
            A.Resize(height=160, width=240),
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
                A.Resize(height=160, width=240),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2()
            ],
        )

    train_ds = CarvanaDataset(train_image_dir, train_mask_dir, transform=train_transform)
    val_ds = CarvanaDataset(val_image_dir, val_mask_dir, transform=val_transforms)

    print(len(train_ds.images))
    print(len(val_ds.images))


if __name__ == '__main__':
    test_dataset()