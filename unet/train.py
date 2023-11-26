from unet_utils.dataset import Unet_DataModule
from unet_utils.model import Unet_lightning, LightCheckPoint
from unet_utils import config

import lightning as L
from lightning.pytorch.loggers import WandbLogger
import wandb

def unet_train_lightning():
    dm = Unet_DataModule()
    model = Unet_lightning()

    wandb.login(key="dec2ee769ce2e455dd463be9b11767cf8190d658")
    wandb_log = WandbLogger(project="unet", name="UNet_with_images", save_dir=config.BASE_OUTPUT)

    checkpoint = LightCheckPoint(logger=wandb_log)

    trainer = L.Trainer(
        max_epochs=config.NUM_EPOCHS,
        accelerator="auto",
        devices=1,
        logger=wandb_log,
        callbacks=[checkpoint]
    )
    trainer.fit(model=model, datamodule=dm)

    wandb.finish()

if __name__ == "__main__":
    L.seed_everything(1702)
    unet_train_lightning()