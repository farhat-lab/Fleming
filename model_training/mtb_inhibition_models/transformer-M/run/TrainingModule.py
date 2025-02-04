from typing import Any, Optional
from pytorch_lightning.utilities.types import STEP_OUTPUT, LRSchedulerTypeUnion, OptimizerLRScheduler
import torch, torch.nn as nn
from pytorch_lightning import LightningModule
from model.transformer_m_test import TransformerM
from utils.metrics import binary_classification_metric_return, regression_metric_return
from argparse import Namespace
import torch.optim as optim
from utils.schedulers import Scheduler
from torch.optim.lr_scheduler import ExponentialLR
import pandas as pd


class TrainingModule(LightningModule):
    def __init__(self,
                lr: float = 1e-5,
                d_model: int = 768,
                loss_name: str = 'bce',
                accelerator: str = 'cpu',
                model_name: str = 'Transformer-M',
                model_args: Namespace = None,
                warmup_steps: int = 4000,
                scale_loss: bool = True,
                batch_size: int = 64,
                noise_scale: float = 0.01,
                channel_weight: list = [0.997,0.003],
                weight_decay: float = 0.0,
                finetune: bool = False,
                use_lr_scheduler: bool = True,
                classification: bool = True,
                checkpoint_dir: str = None,
                ):
        
        super().__init__()

        self.lr = lr
        self.use_lr_scheduler = use_lr_scheduler
        self.loss_name = loss_name
        self.accelerator = accelerator
        self.model_name = model_name
        self.warmup_steps = warmup_steps
        self.d_model = d_model
        self.automatic_optimization = False
        self.model_args = model_args
        self.finetune = finetune
        self.classification = classification
        self.checkpoint_dir = checkpoint_dir

        self.batch_size = batch_size
        self.noise_scale = noise_scale
        self.channel_weight = channel_weight
        self.weight_decay = weight_decay

        self.epoch_id = -1

        self.scale_loss = scale_loss
        if scale_loss:
            self.scaler = torch.cuda.amp.GradScaler()

        self.validation_predictions = torch.empty((self.batch_size,1))
        self.validation_labels = torch.empty((self.batch_size))
        self.validation_smiles = []

        if accelerator == 'gpu':
            self.validation_predictions = self.validation_predictions.to('cuda')
            self.validation_labels = self.validation_labels.to('cuda')

        if self.model_name == 'Transformer-M':
            print(model_args)
            self.model = TransformerM(model_args)

            print(self.model)

            if finetune:
                state_dict = torch.load('/n/scratch/users/a/aw274/Transformer-M/checkpoints/L12-old.pt')['model']

                for key in list(state_dict.keys()):
                    state_dict[key[8:]] = state_dict.pop(key)

                self.model.load_state_dict(state_dict, strict=False)

        if self.classification:
            if self.loss_name == 'bce':
                self.loss_function = nn.BCELoss()
        else:
            self.loss_function = nn.L1Loss()
        

    def forward(
            self,
            batch
    ):
        if self.accelerator == 'gpu':
            if torch.cuda.is_available():
                print("Running on gpu")
            else:
                print("No available gpu found, Running on cpu instead")
        else: 
            print("Running on cpu")
        
        ori_pos = batch['pos']
        noise = torch.randn(ori_pos.shape).to(ori_pos) * self.noise_scale
        noise_mask = (ori_pos == 0.0).all(dim=-1, keepdim=True)
        noise = noise.masked_fill_(noise_mask, 0.0)
        batch['pos'] = ori_pos + noise
        
        output, node_output, inner_states = self.model(batch)

        output = output[:,0,:]

        if self.classification:
            output = nn.Sigmoid()(output)

        if node_output is not None:
            node_mask = (node_output == 0.0).all(dim=-1).all(dim=-1)[:, None, None] + noise_mask
            node_output = node_output.masked_fill_(node_mask, 0.0)

            node_output_loss = (1.0 - nn.CosineSimilarity(dim=-1)(node_output.to(torch.float32), noise.masked_fill_(node_mask, 0.0).to(torch.float32)))
            node_output_loss = node_output_loss.masked_fill_(node_mask.squeeze(-1), 0.0).sum(dim=-1).to(torch.float16)

            tgt_count = (~node_mask).squeeze(-1).sum(dim=-1).to(node_output_loss)
            tgt_count = tgt_count.masked_fill_(tgt_count == 0.0, 1.0)
            node_output_loss = (node_output_loss / tgt_count).sum() * 1
        else:
            node_output_loss = (noise - noise).sum()

        self.node_output_loss = node_output_loss

        # print(self.node_output_loss)

        return output

    def loss(
            self,
            output,
            labels
    ):
        
        loss = self.loss_function(output.view(-1),labels)

        return self.channel_weight[0] * loss + self.channel_weight[1] * self.node_output_loss
    
    
    def training_step(self, batch, batch_idx):
        
        labels = batch['y']
       
        forward_output = self.forward(batch)

        loss = self.loss(forward_output, labels)

        if self.classification:
            logs = binary_classification_metric_return(y_pred = forward_output.view(-1),
                                                        y_true = labels,
                                                        state='train')
        else:
            logs = regression_metric_return(y_pred=forward_output.view(-1),
                                            y_true = labels,
                                            state = 'train')

        optimizer = self.optimizers()
        optimizer.zero_grad()

        if self.scale_loss:     
            self.manual_backward(self.scaler.scale(loss))

            self.scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_args.clip_norm)
    
            self.scaler.step(optimizer)
            self.scaler.update()

        else:
            self.manual_backward(loss)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.model_args.clip_norm)
            optimizer.step()

        if (self.finetune == False) and (self.use_lr_scheduler == True):
            scheduler = self.lr_schedulers()
            scheduler.step()

        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size = self.batch_size)

        print(logs)

        self.log_dict(logs, 
                      on_step=True, 
                      on_epoch=True,
                      batch_size=self.batch_size)
        
        self.log("node_ouput_loss", self.node_output_loss)
        self.log("total_loss", loss)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        
        labels = batch['y']

        forward_output = self.forward(batch)

        # logs = binary_classification_metric_return(y_pred = forward_output.view(-1),
        #                                            y_true = labels,
        #                                            state = 'validation_step')

        # print(logs)
        
        # self.log_dict(logs)
        self.validation_smiles = self.validation_smiles + list(batch['smiles'])
        self.validation_predictions = torch.cat((self.validation_predictions,forward_output), dim = 0) # self.validation_predictions.extend(torch.flatten(forward_output))
        self.validation_labels = torch.cat((self.validation_labels,labels), dim = 0)
        
        return forward_output, labels
    
    def on_validation_epoch_end(self):
        
        self.validation_predictions = self.validation_predictions[self.batch_size:,:]
        self.validation_labels = self.validation_labels[self.batch_size:]
        # self.validation_smiles = self.validation_smiles

        # print(self.validation_predictions.cpu().numpy())
        # print(self.validation_labels.cpu().numpy())
        # print(self.validation_smiles[self.batch_size:])

        val_pred_df = pd.DataFrame()
        # val_pred_df['SMILES'] = self.validation_smiles[self.batch_size:]
        val_pred_df['inhibition'] = self.validation_labels.cpu().numpy()
        val_pred_df['pred'] = self.validation_predictions.cpu().numpy()
        # print(val_pred_df) 
        
        if self.classification:
            logs = binary_classification_metric_return(y_pred = self.validation_predictions.view(-1),
                                                    y_true = self.validation_labels,
                                                    state = 'validation_epoch_')
        else:
            logs = regression_metric_return(y_pred = self.validation_predictions.view(-1),
                                            y_true = self.validation_labels,
                                            state = 'validation_epoch_')

        print(logs)
        
        self.log_dict(logs, batch_size=self.batch_size)

        self.validation_predictions = torch.empty((self.batch_size,1))
        self.validation_labels = torch.empty((self.batch_size))
        self.validation_smiles = []

        if self.accelerator == 'gpu':
            self.validation_predictions = self.validation_predictions.to('cuda')
            self.validation_labels = self.validation_labels.to('cuda')

        val_pred_df.to_csv(self.checkpoint_dir + f'val_pred_{self.epoch_id}.csv',
                           index = False)

        self.epoch_id += 1

    
    def test_step(self, batch,  batch_idx):
        
        labels = batch['y']

        forward_output = self.forward(batch)

        loss = self.loss(forward_output, labels)

        logs = binary_classification_metric_return(y_pred = forward_output,
                                                   y_true = labels,
                                                   state='test')

        print(logs)

        self.log_dict(logs)
        
        return forward_output, labels
    
    def configure_optimizers(self):
        
        optimizer = optim.Adam(self.parameters(), 
                               lr=self.lr,
                               weight_decay=self.weight_decay)
        
        if self.finetune or (self.use_lr_scheduler == False):
            return optimizer
        else:            
            scheduler = {'scheduler': # ExponentialLR(optimizer, gamma=0.9),
                        Scheduler(optimizer=optimizer,
                                        dim_embed=self.d_model,
                                                warmup_steps=self.warmup_steps),
                        'interval': 'step',
                        'frequency': 1,
                        'name': 'lr_rate_opt'
            }

            return [optimizer], [scheduler]

    

