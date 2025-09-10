import hydra
import logging
import torch
import warnings
from omegaconf import DictConfig

from lightning.pytorch import Trainer
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch import LightningModule
from lightning.pytorch.loggers import WandbLogger

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="configs",
    config_name="train",
    version_base="1.3"
)
def main(config: DictConfig):
    torch.set_float32_matmul_precision('high')

    # Fix seed, if seed everything: fix seed for python, numpy and pytorch
    if config.get("seed"):
        hydra.utils.instantiate(config.seed)
    else:
        warnings.warn("No seed fixed, the results are not reproducible.")

    # Create trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer)

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Create model
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Log model definition
    logger.info(model)

    # Log configuration
    if isinstance(trainer.logger, WandbLogger):
        trainer.logger.log_hyperparams(config)
    
    # Fit the trainer
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    main()