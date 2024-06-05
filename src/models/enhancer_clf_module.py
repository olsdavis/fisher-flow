import os
import yaml
from typing import Any

import torch
import torch.nn.functional as F
import lightning as pl
from torchmetrics import Accuracy

from src.models.net import CNNModel
from src.data.components import upgrade_state_dict

"""
To test the module, run:

python -m src.models.enhancer_clf_module
"""

class SequenceClassificationModel(pl.LightningModule):
    def __init__(
        self, 
        model: torch.nn.Module, 
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        num_cls: int,
        ckpt_path: str = None,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.model = model
        self.accuracy = Accuracy(task="multiclass", num_classes=num_cls)

    # def load_state_dict(self, ckpt_path):
    #     import pdb; pdb.set_trace()
    #     if isinstance(ckpt_path, dict):
    #         state_dict_new_name = upgrade_state_dict(
    #             ckpt_path['state_dict'],
    #             prefixes=['model.', 'net.'],
    #         )
    #         self.model.load_state_dict(state_dict_new_name)
    #     else:
    #         state_dict_new_name = upgrade_state_dict(
    #             torch.load(ckpt_path, map_location=self.device)['state_dict'], prefixes=['model.', 'net.'],
    #         )
    #         self.model.load_state_dict(state_dict_new_name)

    def forward(self, input_ids):
        return self.model(input_ids, t=None)

    def compute_metrics(self, logits, labels):
        loss = F.cross_entropy(logits, labels)
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)
        perplexity = torch.exp(loss)
        return loss, acc, perplexity

    def training_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss, acc, perplexity = self.compute_metrics(logits, labels)
        lr = self.optimizers().param_groups[0]['lr']
        self.log('train/loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/acc', acc, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train/perplexity', perplexity, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('learning_rate', lr, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss, acc, perplexity = self.compute_metrics(logits, labels)
        self.log('val/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        input_ids, labels = batch
        logits = self(input_ids)
        loss, acc, perplexity = self.compute_metrics(logits, labels)
        self.log('test/loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/acc', acc, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('test/perplexity', perplexity, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def on_train_end(self):
        # Retrieve the checkpoint callback
        checkpoint_callback = None
        for callback in self.trainer.callbacks:
            if isinstance(callback, pl.callbacks.ModelCheckpoint):
                checkpoint_callback = callback
                break
        
        if checkpoint_callback is None:
            print("Checkpoint callback not found.")
            return
        
        # Get the best checkpoint path
        best_checkpoint_path = checkpoint_callback.best_model_path
        if not best_checkpoint_path:
            print("Best checkpoint path not found.")
            return
        
        # Extract the directory of the best checkpoint
        checkpoint_dir = os.path.dirname(best_checkpoint_path)
        
        # Save hyperparameters to a YAML file in the same directory
        hparams_file = os.path.join(checkpoint_dir, 'hparams.yaml')
        with open(hparams_file, 'w') as f:
            yaml.dump(self.hparams, f)
        print(f"Hyperparameters saved to {hparams_file}")

    def configure_optimizers(self) -> dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    # model = PromoterModel(mode='dirichlet')
    # PromoterModule(model, None, None)
    net = CNNModel(dim=4, hidden=128, mode=None, num_cls=4, depth=3, dropout=0.1, prior_pseudocount=2, cls_expanded_simplex=False, clean_data=True, classifier=True, classifier_free_guidance=False)
    SequenceClassificationModel(net, None, None, 41)