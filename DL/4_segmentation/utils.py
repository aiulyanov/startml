import torch
import torchvision
import logging
import sys


def setup_logger():
    logger = logging.getLogger(__name__)

    if not logger.handlers:
        logger.setLevel(logging.DEBUG)

        formatter = logging.Formatter("[%(levelname)s] [%(module)s] %(message)s")

        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)

        logger.addHandler(stream_handler)

    return logger


logger = setup_logger()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    logger.info("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    logger.info("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (images, labels) in enumerate(loader):
        images = images.to(device=device)
        with torch.no_grad():
            preds = torch.sigmoid(model(images))
            preds = (preds > 0.5).float()
        torchvision.utils.save_image(preds, f"{folder}/pred_{idx}.png")
        torchvision.utils.save_image(labels.unsqueeze(1), f"{folder}{idx}.png")

    model.train()
