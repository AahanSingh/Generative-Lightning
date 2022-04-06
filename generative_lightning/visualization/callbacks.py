import pytorch_lightning as pl
import torch
import wandb
import numpy as np


class WandbImageCallback(pl.Callback):
    """Logs the input and output images of a module.
    
    Images are stacked into a mosaic, with output on the top
    and input on the bottom."""

    def __init__(self, eval_interval):
        super().__init__()
        self.eval_interval = eval_interval

    def on_train_batch_end(outputs, batch, batch_idx, unused=0):
        # implement your own
        if batch_idx % self.eval_interval == 0:
            prediction = pl_module(val_imgs, generate_monet=True)
            prediction = (prediction * 127.5 + 127.5)
            prediction = prediction.squeeze().permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            mosaics = torch.cat([prediction, val_imgs], dim=-2)
            caption = "Top: Output, Bottom: Input"
            wandb.log({
                "val/examples": [wandb.Image(mosaic, caption=caption) for mosaic in mosaics],
                "global_step": trainer.global_step
            })
