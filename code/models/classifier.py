import torch
import hydra
import lightning.pytorch as pl
import numpy as np

from models.heads import DSNetAFHead
from evaluation.evaluate import compute_frame_metrics, average_predictions, keyshot_selection, compute_segment_metrics, get_metric_per_match, select_temporal_keyshots, compute_temporal_metrics, compute_shot_mAP, print_evaluation

# Original code https://github.com/tatp22/multidim-positional-encoding
def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(torch.nn.Module):
    def __init__(self, channels: int):
        """
        :param channels: The last dimension of the Tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.cached_penc = None

    def forward(self, tensor: torch.Tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq)
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device).type(tensor.type())
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

class TransformerEncoder(torch.nn.Module):
    def __init__(
            self,
            ndim,
            nlayers,
            nheads,
            dropout,
            batch_first,
            add_cls_token,
            src_key_padding_mask
    ):
        super().__init__()
        
        self.pos_encoder = PositionalEncoding1D(channels=ndim)
        
        self.encoder = torch.nn.TransformerEncoder(
            encoder_layer=torch.nn.TransformerEncoderLayer(
                d_model=ndim, 
                nhead=nheads, 
                dropout=dropout, 
                batch_first=batch_first
            ),
            num_layers=nlayers,
            norm=torch.nn.LayerNorm(ndim)
        )

        if add_cls_token:
            setattr(self, 'cls_token', torch.nn.Parameter(torch.rand((1, 1, ndim))))

        if src_key_padding_mask:
            setattr(self, 'prob_mask', src_key_padding_mask.prob_mask)

        
    def load_weights(self, weights):
        self.load_state_dict(weights, strict=False)
        print("Encoder weights succesfully loaded")

    def forward(self, x):
        if hasattr(self, 'cls_token'):
            x = torch.cat((self.cls_token.expand(x.shape[0], -1, -1), x), dim=1)
        
        x = x + self.pos_encoder(x)

        if hasattr(self, 'prob_mask'):
            random_mask = torch.bernoulli(torch.full((x.shape[0], x.shape[1]), 1 - self.prob_mask)).bool().to('cuda')
            x = self.encoder(x, src_key_padding_mask=~random_mask)
        else:
            x = x = self.encoder(x)

        return x

class SummaryClassifier(pl.LightningModule):
    def __init__(
        self,
        input_dim,
        ndim,
        encoder,
        head,
        loss_fn,
        optimizer,
        scheduler,
        metrics,
        mixup,
        evaluate,
        threshold
    ):
        super().__init__()
        
        if input_dim != ndim:
            self.dim_reduction = torch.nn.Linear(
                in_features = input_dim,
                out_features = ndim
            )
        
        self.encoder = hydra.utils.instantiate(encoder)
            
        self.head = hydra.utils.instantiate(head) 
        self.loss_fn = [hydra.utils.instantiate(loss) for loss in loss_fn]

        if not isinstance(self.loss_fn[0], torch.nn.BCEWithLogitsLoss):
            self.sigmoid = torch.nn.Sigmoid()

        setattr(self, 'optimizer', optimizer)
        setattr(self, 'scheduler', scheduler)
        
        self.metrics = hydra.utils.instantiate(metrics)
        
        if mixup:
            self.mixup = hydra.utils.instantiate(mixup)

        setattr(self, 'evaluate', evaluate)
        
        setattr(self, 'threshold', threshold)

    def forward(self, x):
        if hasattr(self, 'dim_reduction'):
            x = self.dim_reduction(x)
            
        x = self.encoder(x)
        
        x = self.head(x)
        
        x = [x[i] for i in range(self.head.nheads)] if self.head.nheads > 1 else [x]
                
        if hasattr(self, 'sigmoid'):
            for i, loss in enumerate(self.loss_fn):
                if isinstance(loss, torch.nn.BCELoss):
                    x[i] = self.sigmoid(x[i])

        return x


    def configure_optimizers(self):
        self.optimizer = hydra.utils.instantiate(self.optimizer, params=self.parameters())
        self.scheduler = hydra.utils.instantiate(self.scheduler, optimizer=self.optimizer)

        return {
            "optimizer": self.optimizer,
            "lr_scheduler": {
                "scheduler": self.scheduler,
                "monitor": "valid/loss",
                "frequency": 1
            },
        }
    
    def on_train_epoch_start(self):
        pass
    
    def training_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        
        if hasattr(self, 'mixup'):
            x, y = self.mixup(x, y)

        mask = batch['mask'] if 'mask' in batch.keys() else None

        y_hat = self.forward(x)

        loss = 0
        for i, (key, value) in enumerate(y.items()):
            # Avoid defining different number of heads and losses
            if i == len(self.loss_fn):
                break
            
            if key != 'boundaries':
                if key == 'actions' and mask is not None:
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float()) * mask.expand(-1, -1, value.shape[-1])
                    loss_fn = self.head.weights[i] * loss_fn.sum() / mask.expand(-1, -1, value.shape[-1]).sum()
                elif self.loss_fn[i].reduction == 'none':
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float())
                    loss_fn = self.head.weights[i] * loss_fn.sum() / loss_fn.numel()
                else:
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float()) if not isinstance(self.head, DSNetAFHead) else self.head.weights[i] * self.loss_fn[i](y_hat[i].float(), value.float())
                    
                loss += loss_fn

                if len(self.loss_fn) > 1:
                    self.log(
                        f"train/loss_{key}", 
                        loss_fn, 
                        on_step=False, 
                        on_epoch=True,
                        prog_bar=False, 
                        logger=True, 
                        sync_dist=True,
                        batch_size=x.shape[0]
                    )

            else:
                # Keep only positive examples
                positives = y['labels'].type(torch.bool)
                predictions = y_hat[i][positives] if positives.shape[-1] == value.shape[-1] else torch.stack([y_hat[i][:,:,channel].unsqueeze(-1)[positives] for channel in range(0, y_hat[i].shape[-1])])
                target = value[positives] if positives.shape[-1] == value.shape[-1] else torch.stack([value[:,:,channel].unsqueeze(-1)[positives] for channel in range(0, value.shape[-1])])

                # Handle the case where there are no positive samples
                if predictions.numel() == 0:
                    continue
                
                # Compute loss for positive examples
                loss_n = self.loss_fn[i](predictions.float(), target.float()) if not isinstance(self.head, DSNetAFHead) else self.head.weights[i] * self.loss_fn[i](predictions.float(), target.float())
                loss += loss_n

                self.log(
                    f"train/loss_{key}", 
                    loss_n, 
                    on_step=False, 
                    on_epoch=True,
                    prog_bar=False, 
                    logger=True, 
                    sync_dist=True,
                    batch_size=predictions.shape[-1]
                )
            
        self.log(
            'train/loss', 
             loss, 
             on_step=True, 
             on_epoch=True,
             prog_bar=True, 
             logger=True, 
             sync_dist=True,
             batch_size=x.shape[0]
        )
        
        return loss
    
    def on_validation_epoch_start(self):   
        if self.evaluate: 
            self.outputs, self.labels = [], []
            if isinstance(self.head, DSNetAFHead):
                self.predictions = []
                categories = ["", "_masked", "_offmatch"] if self.trainer.datamodule.valid_dataset.dataset_info['masked'] else [""]
                metrics = ["TP", "FP", "FN", "IoU", "precision", "recall", "f1"]
                self.eval_metrics = {f"{metric}{category}": 0 for metric in metrics for category in categories}

    def validation_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        
        mask = batch['mask'] if 'mask' in batch.keys() else None

        y_hat = self.forward(x)

        loss = 0
        for i, (key, value) in enumerate(y.items()):
            # Avoid defining different number of heads and losses
            if i == len(self.loss_fn):
                break

            if key != 'boundaries':
                if key == 'actions' and mask is not None:
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float()) * mask.expand(-1, -1, value.shape[-1])
                    loss_fn = self.head.weights[i] * loss_fn.sum() / mask.expand(-1, -1, value.shape[-1]).sum()
                elif self.loss_fn[i].reduction == 'none':
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float())
                    loss_fn = self.head.weights[i] * loss_fn.sum() / loss_fn.numel()
                else:
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float()) if not isinstance(self.head, DSNetAFHead) else self.head.weights[i] * self.loss_fn[i](y_hat[i].float(), value.float())
                    
                loss += loss_fn

                if len(self.loss_fn) > 1:
                    self.log(
                        f"valid/loss_{key}", 
                        loss_fn, 
                        on_step=False, 
                        on_epoch=True,
                        prog_bar=False, 
                        logger=True, 
                        sync_dist=True,
                        batch_size=x.shape[0]
                    )
            else:
                # Keep only positive examples
                positives = y['labels'].type(torch.bool)
                predictions = y_hat[i][positives] if positives.shape[-1] == value.shape[-1] else torch.stack([y_hat[i][:,:,channel].unsqueeze(-1)[positives] for channel in range(0, y_hat[i].shape[-1])])
                target = value[positives] if positives.shape[-1] == value.shape[-1] else torch.stack([value[:,:,channel].unsqueeze(-1)[positives] for channel in range(0, value.shape[-1])])

                # Handle the case where there are no positive samples
                if predictions.numel() == 0:
                    continue

                # Compute loss for positive examples
                loss_n = self.loss_fn[i](predictions.float(), target.float()) if not isinstance(self.head, DSNetAFHead) else self.head.weights[i] * self.loss_fn[i](predictions.float(), target.float())
                loss += loss_n

                self.log(
                    f"valid/loss_{key}", 
                    loss_n, 
                    on_step=False, 
                    on_epoch=True,
                    prog_bar=False, 
                    logger=True, 
                    sync_dist=True,
                    batch_size=predictions.shape[-1]
                )

        self.log(
            'valid/loss', 
             loss, 
             on_step=True, 
             on_epoch=True,
             prog_bar=True, 
             logger=True, 
             sync_dist=True,
             batch_size=x.shape[0]
        )
        
        if self.evaluate:
            if not hasattr(self, 'sigmoid'):
                y_hat[0] = y_hat[0].sigmoid()
                if self.head.nheads > 2:
                    y_hat[2] = y_hat[2].sigmoid()
                    
            if hasattr(self, 'predictions'):
                self.predictions.append([y_hat[i].cpu().detach().numpy() for i in range(self.head.nheads)])
            
            if self.trainer.datamodule.valid_dataset.dataset_info['stride'] == self.trainer.datamodule.valid_dataset.dataset_info['frames_per_window']:
                self.outputs.extend(np.concatenate(y_hat[0].cpu().detach().numpy()))
                self.labels.extend(np.concatenate(y['labels'].cpu().detach().numpy()))
            else:
                self.outputs.append(y_hat[0].cpu().detach().numpy())

        return loss
    
    def on_validation_epoch_end(self):
        if self.evaluate:
            if self.trainer.datamodule.valid_dataset.dataset_info['stride'] != self.trainer.datamodule.valid_dataset.dataset_info['frames_per_window']:
                self.outputs, self.labels = average_predictions(
                    predictions=self.outputs,
                    dataset=self.trainer.datamodule.valid_dataset,
                    batch_size=self.trainer.datamodule.valid.batch_size,
                    threshold=self.threshold if hasattr(self, 'threshold') else 0.5
                )
            # Analyse raw frame prediction
            for metric in self.metrics:
                score = compute_frame_metrics(
                    predictions=self.outputs,
                    target=self.labels,
                    metric=metric,
                    classes=1,
                    threshold=self.threshold if hasattr(self, 'threshold') else 0.5
                )
                
                self.log(
                    f"metrics/frame/{metric}", 
                    score, 
                    on_step=False, 
                    on_epoch=True,
                    prog_bar=False, 
                    logger=True, 
                    sync_dist=True
                )
            
            # Apply keyshot selection
            if hasattr(self, 'predictions'):
                keyshots, self.eval_metrics, preds = keyshot_selection(
                    predictions=self.predictions, 
                    dataset=self.trainer.datamodule.valid_dataset, 
                    threshold=self.threshold if hasattr(self, 'threshold') else 0.5,
                    metrics=self.eval_metrics
                )
                
                if keyshots:
                    for metric in ['f1']:
                        scores = get_metric_per_match(keyshots, metric)
                        if isinstance(self.logger, pl.loggers.WandbLogger):
                            for i, score in enumerate(scores):
                                self.log(
                                    f"inference/{metric}/match {i}", 
                                    score, 
                                    on_step=False, 
                                    on_epoch=True,
                                    prog_bar=False, 
                                    logger=True, 
                                    sync_dist=True
                                )

                    # Compute mAP
                    mAP = compute_shot_mAP(keyshots, self.trainer.datamodule.valid_dataset.nframes)
                    
                    self.log(
                        f"metrics/segment/mAP", 
                        mAP, 
                        on_step=False, 
                        on_epoch=True,
                        prog_bar=False, 
                        logger=True, 
                        sync_dist=True
                    )

                # Compute keyshot selection metrics
                if self.eval_metrics:
                    self.eval_metrics = compute_segment_metrics(self.eval_metrics, len(self.trainer.datamodule.valid_dataset.nframes), self.trainer.datamodule.valid_dataset.dataset_info['masked'])

                    for metric in self.eval_metrics:
                        self.log(
                            f"metrics/segment/{metric}", 
                            self.eval_metrics[metric], 
                            on_step=False, 
                            on_epoch=True,
                            prog_bar=False, 
                            logger=True, 
                            sync_dist=True
                        )

                if keyshots:
                    # Compute metrics for fitted predictions
                    temporal_keyshots = select_temporal_keyshots(keyshots=keyshots, clip_segment=False)
                    temporal_metrics = compute_temporal_metrics(keyshots=temporal_keyshots, n_frames=self.trainer.datamodule.valid_dataset.nframes, average=True, preds=preds)
                    
                    for metric in temporal_metrics:
                        self.log(
                            f"metrics/temporal/{metric}", 
                            temporal_metrics[metric], 
                            on_step=False, 
                            on_epoch=True,
                            prog_bar=False, 
                            logger=True, 
                            sync_dist=True
                        )
        return
    
    def on_test_epoch_start(self):    
        self.outputs, self.labels = [], []
        if isinstance(self.head, DSNetAFHead):
            self.predictions = []
            categories = ["", "_masked", "_offmatch"] if self.trainer.datamodule.test_dataset.dataset_info['masked'] else [""]
            metrics = ["TP", "FP", "FN", "IoU", "precision", "recall", "f1"]
            self.eval_metrics = {f"{metric}{category}": 0 for metric in metrics for category in categories}

    def test_step(self, batch, batch_idx):
        x, y = batch['x'], batch['y']
        
        mask = batch['mask'] if 'mask' in batch.keys() else None

        y_hat = self.forward(x)

        loss = 0
        for i, (key, value) in enumerate(y.items()):
            # Avoid defining different number of heads and losses
            if i == len(self.loss_fn):
                break

            if key != 'boundaries':
                if key == 'actions' and mask is not None:
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float()) * mask.expand(-1, -1, value.shape[-1])
                    loss_fn = self.head.weights[i] * loss_fn.sum() / mask.expand(-1, -1, value.shape[-1]).sum()
                elif self.loss_fn[i].reduction == 'none':
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float())
                    loss_fn = self.head.weights[i] * loss_fn.sum() / loss_fn.numel()
                else:
                    loss_fn = self.loss_fn[i](y_hat[i].float(), value.float()) if not isinstance(self.head, DSNetAFHead) else self.head.weights[i] * self.loss_fn[i](y_hat[i].float(), value.float())
                    
                loss += loss_fn
            else:
                # Keep only positive examples
                positives = y['labels'].type(torch.bool)
                predictions = y_hat[i][positives] if positives.shape[-1] == value.shape[-1] else torch.stack([y_hat[i][:,:,channel].unsqueeze(-1)[positives] for channel in range(0, y_hat[i].shape[-1])])
                target = value[positives] if positives.shape[-1] == value.shape[-1] else torch.stack([value[:,:,channel].unsqueeze(-1)[positives] for channel in range(0, value.shape[-1])])

                # Handle the case where there are no positive samples
                if predictions.numel() == 0:
                    continue

                # Compute loss for positive examples
                loss_n = self.loss_fn[i](predictions.float(), target.float()) if not isinstance(self.head, DSNetAFHead) else self.head.weights[i] * self.loss_fn[i](predictions.float(), target.float())
                loss += loss_n
        
        if not hasattr(self, 'sigmoid'):
            y_hat[0] = y_hat[0].sigmoid()
            if self.head.nheads > 2:
                y_hat[2] = y_hat[2].sigmoid()
                
        if hasattr(self, 'predictions'):
            self.predictions.append([y_hat[i].cpu().detach().numpy() for i in range(self.head.nheads)])
        
        if self.trainer.datamodule.test_dataset.dataset_info['stride'] == self.trainer.datamodule.test_dataset.dataset_info['frames_per_window']:
            self.outputs.extend(np.concatenate(y_hat[0].cpu().detach().numpy()))
            self.labels.extend(np.concatenate(y['labels'].cpu().detach().numpy()))
        else:
            self.outputs.append(y_hat[0].cpu().detach().numpy())

        return loss

    def on_test_epoch_end(self):
        if self.trainer.datamodule.test_dataset.dataset_info['stride'] != self.trainer.datamodule.test_dataset.dataset_info['frames_per_window']:
            self.outputs, self.labels = average_predictions(
                predictions=self.outputs,
                dataset=self.trainer.datamodule.test_dataset,
                batch_size=self.trainer.datamodule.test.batch_size,
                threshold=self.threshold if hasattr(self, 'threshold') else 0.5
            )
        
        # Apply keyshot selection
        if hasattr(self, 'predictions'):
            keyshots, self.eval_metrics, preds = keyshot_selection(
                predictions=self.predictions, 
                dataset=self.trainer.datamodule.test_dataset, 
                threshold=self.threshold if hasattr(self, 'threshold') else 0.5,
                metrics=self.eval_metrics
            )

            # Analyse keyshot selection
            if self.eval_metrics:
                self.eval_metrics = compute_segment_metrics(self.eval_metrics, len(self.trainer.datamodule.test_dataset.nframes), self.trainer.datamodule.test_dataset.dataset_info['masked'])

            if keyshots:
                # Compute mAP
                self.eval_metrics['mAP'] = compute_shot_mAP(keyshots, self.trainer.datamodule.test_dataset.nframes)
                # Compute metrics for fitted predictions
                temporal_keyshots = select_temporal_keyshots(keyshots=keyshots, clip_segment=False)
                temporal_metrics = compute_temporal_metrics(keyshots=temporal_keyshots, n_frames=self.trainer.datamodule.test_dataset.nframes, average=True, preds=preds)

        print_evaluation(self.eval_metrics, temporal_metrics)

        return