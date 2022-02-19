import pytorch_lightning as pl
import torch
import wandb
import numpy as np


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""

    def __init__(self, val_samples, max_samples=32):
        super().__init__()
        self.val_imgs, _ = val_samples
        self.val_imgs = self.val_imgs[:max_samples]

    def on_validation_end(self, trainer, pl_module):
        val_imgs = self.val_imgs.to(device=pl_module.device)
        # implement your own
        prediction = pl_module.m_gen(val_imgs)
        prediction = (prediction * 127.5 + 127.5).numpy().astype(np.uint8)
        mosaics = torch.cat([prediction, val_imgs], dim=-2)
        caption = "Top: Output, Bottom: Input"
        trainer.logger.experiment.log({
            "val/examples": [wandb.Image(mosaic, caption=caption) for mosaic in mosaics],
            "global_step": trainer.global_step
        })
