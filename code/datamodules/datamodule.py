import torch

import lightning.pytorch as pl

from omegaconf import DictConfig
from torch.utils.data import DataLoader
from typing import Optional


class DataModule(pl.LightningDataModule):
    def __init__(
        self, 
        splits: list, 
        train: DictConfig,
        valid: DictConfig,
        test: DictConfig,
        predict: DictConfig
    ):
        super().__init__()

        setattr(self, 'splits', splits)

        if 'train' in splits:
            self.train_dataset = train.dataset
            train.pop('dataset')
            setattr(self, 'train', train)

        if 'valid' in splits:
            self.valid_dataset = valid.dataset
            valid.pop('dataset')
            setattr(self, 'valid', valid)
            
        if 'test' in splits:
            self.test_dataset = test.dataset
            test.pop('dataset')
            setattr(self, 'test', test)
            
        if 'predict' in splits:
            self.predict_dataset = predict.dataset
            predict.pop('dataset')
            setattr(self, 'predict', predict)

    def prepare_data(self):
        pass

    def setup(self, stage: Optional[str] = None):
        for split in self.splits:
            conf = getattr(self, f'{split}')
            conf.update({'collate_fn': custom_collate_fn})
            setattr(self, f'{split}', conf)
        return

    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            **self.train
        ) if hasattr(self, 'train_dataset') else None

    def val_dataloader(self):
        return DataLoader(
            dataset=self.valid_dataset,
            **self.valid
        ) if hasattr(self, 'valid_dataset') else None

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            **self.test
        ) if hasattr(self, 'test_dataset') else None
        
    def predict_dataloader(self):
        return DataLoader(
            dataset=self.predict_dataset,
            **self.predict
        ) if hasattr(self, 'predict_dataset') else None
    

def custom_collate_fn(batch):
    # Define the output dict
    batch_dict = {'x': None, 'x_shot': None, 'mask': None, 'info': {}, 'y': {}}

    # Process each key in the first item's keys, as all items have the same keys
    for key in batch[0].keys():
        # Handle dict elements
        if isinstance(batch[0][key], dict):
            stacked_values = [torch.stack([item[key][subkey] for item in batch]) for subkey in batch[0][key].keys()]
        else:
            stacked_values = torch.stack([item[key] for item in batch])

        # Organize each key
        if key == 'imgs': # Store images
            # Handle multiple imgs representation
            if isinstance(batch[0][key], dict):
                for key_idx, subkey in enumerate(batch[0][key].keys()):
                    batch_dict[subkey] = stacked_values[key_idx]  
            else:
                batch_dict['x'] = stacked_values
        elif key == 'mask':
            batch_dict['mask'] = stacked_values  # Store mask
        elif 'id' in key:
            batch_dict['info'][key] = stacked_values  # Store info-related data
        else:
            batch_dict['y'][key] = stacked_values  # Store other labels

    return batch_dict