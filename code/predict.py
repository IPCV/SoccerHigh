import os
import sys
import hydra
import logging
import torch
import warnings
from omegaconf import DictConfig

from lightning.pytorch import Trainer
from lightning.pytorch.core.datamodule import LightningDataModule
from lightning.pytorch import LightningModule

from inference.inference import keyshot_selection, format_output

logger = logging.getLogger(__name__)

@hydra.main(
    config_path="configs",
    config_name="predict",
    version_base="1.3"
)
def main(config: DictConfig):
    torch.set_float32_matmul_precision('high')

    # Fix seed, if seed everything: fix seed for python, numpy and pytorch
    if config.get("seed"):
        hydra.utils.instantiate(config.seed)
    else:
        warnings.warn("No seed fixed, the results are not reproducible.")
        
    # Assure checkpoint exists
    if not os.path.exists(config.checkpoint):
        print(f"Error: The path '{config.checkpoint}' does not exist.")
        sys.exit(1)
    else:
        checkpoint_path = config.checkpoint

    # Create trainer
    trainer: Trainer = hydra.utils.instantiate(config.trainer)

    # Create datamodule
    datamodule: LightningDataModule = hydra.utils.instantiate(config.datamodule)

    # Create model
    model: LightningModule = hydra.utils.instantiate(config.model)

    # Log model definition
    logger.info(model)

    # Predict the output
    predictions = trainer.predict(
        model=model, 
        datamodule=datamodule, 
        return_predictions=config.return_predictions,
        ckpt_path=checkpoint_path
    )

    if predictions:
        predictions = keyshot_selection(
            predictions=predictions, 
            dataset=trainer.datamodule.predict_dataset
        )

        # Format output predictions to JSON
        format_output(
            output=predictions, 
            game_list=trainer.datamodule.predict_dataset.game_list,
            save=config.save_output,
            fname=config.output_path
        )


if __name__ == "__main__":
    main()