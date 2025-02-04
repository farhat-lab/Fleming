import torch
from torch import nn, optim
from models.egnn_clean.egnn_clean import EGNN, get_edges_batch
from qm9.utils import get_adj_matrix
from torch.utils.data import DataLoader
from tb.data.dataset import EGNN_TB_Inhibition_Dataset
from pytorch_lightning import LightningModule
from models.egnn_clean.egnn_clean import EGNN
from model_training.mtb_inhibition_models.ffnn.models.naive_fnn import naive_fnn
from tb.utils.metrics import binary_classification_metric_return, regression_metric_return
import pandas as pd
from torchmetrics import HingeLoss
from tb.utils.loss_fn import TruncatedLoss, Binary_GCELoss, dirichlet_class_loss
from tb.utils.schedulers import Scheduler
import numpy as np

class TrainingModule(LightningModule):
    def __init__(self,
                 lr: float = 5e-4,
                 batch_size: int = 64,
                 n_feat: int = 8,
                 device: str = 'cpu',
                 hidden_nf: int = 32,
                 out_node_nf: int = 1,
                 in_edge_nf: int = 0,
                 n_layers: int = 4,
                 residual: bool = True,
                 attention: bool = False,
                 normalize: bool = True,
                 tanh: bool = False,
                 mol_feature_len: int = 0,
                 gradient_clip: float = 0.2,
                 classification: bool = True,
                 weight_decay: float = 0.2,
                 dropout: float = 0.0,
                 model_name: str = 'EGNN',
                 task: str = 'classification',
                 loss_function: str = 'BCE',
                 train_size: int = 49999,
                 warmup_steps: int = 4000,
                 use_lr_scheduler: bool = False,
                 lam: float = 0.2,
                 output_dir: str = None,):
        
        super().__init__()

        self.lr = lr
        self.batch_size = batch_size
        self.hidden_nf = hidden_nf
        self.out_node_nf = out_node_nf
        self.gradient_clip = gradient_clip
        self.weight_decay = weight_decay
        self.task = task
        self.warmup_steps= warmup_steps
        self.use_lr_scheduler = use_lr_scheduler

        self.loss_function = loss_function
        self.lam = lam

        self.classification = classification
        
        self.accelerator = device
        self.output_dir = output_dir

        self.automatic_optimization = False
        self.save_hyperparameters()

        self.model_name = model_name

        if self.model_name == 'EGNN':
            self.model = EGNN(in_node_nf=n_feat, 
                            hidden_nf=hidden_nf, 
                            out_node_nf=out_node_nf, 
                            in_edge_nf=in_edge_nf, 
                            device=device, 
                            act_fn=nn.SiLU(), 
                            n_layers=n_layers, 
                            residual=residual, 
                            attention=attention, 
                            normalize=normalize, 
                            tanh=tanh,
                            dropout_value=dropout,)

            self.graph_dec = nn.Sequential(nn.Linear(self.hidden_nf + mol_feature_len, self.hidden_nf),
                                        nn.Dropout(dropout),
                                        nn.SiLU(),
                                        nn.Linear(self.hidden_nf, out_node_nf))
        
        elif self.model_name  == 'naive_ffn':
            self.model = naive_fnn(in_node_nf=mol_feature_len,
                              hidden_nf=hidden_nf,
                              out_node_nf=out_node_nf,
                              n_layers=n_layers,
                              )    
        if self.classification:
            if loss_function == 'BCE':
                self.readout = nn.Sigmoid()
                self.loss_fn = nn.BCELoss()
            elif loss_function == 'GCE':
                self.readout = nn.Sigmoid()
                self.loss_fn = Binary_GCELoss()
            elif loss_function == 'TruncatedLoss':
                self.readout = nn.Sigmoid()
                self.loss_fn = TruncatedLoss(q = 0.5,
                                             k = 0.5,
                                             trainset_size=70000)
            elif loss_function == 'Hinge':
                self.readout = nn.Tanh()
                self.loss_fn = HingeLoss()     
            elif loss_function == 'dirichlet':
                self.readout = nn.Softplus()
        else:
            self.loss_fn = nn.L1Loss()

        self.validation_predictions = []
        self.validation_labels = []
        self.validation_idx = []

        self.test_predictions = []
        self.test_labels = []
        self.test_smiles = []

    def forward(self, batch):

        if self.model_name == 'EGNN':
            n_nodes = batch['num_atom']

            print(len(batch['smiles']))
            print(self.batch_size)

            # atom_pos = batch['pos'].view(self.batch_size * n_nodes, -1).to(self.accelerator)
            # atom_features = batch['atom_features'].view(self.batch_size * n_nodes, -1).to(self.accelerator)
            # edges = get_adj_matrix(n_nodes, self.batch_size, self.accelerator)
            atom_pos = batch['pos'].view(len(batch['smiles']) * n_nodes, -1).to(self.accelerator)
            atom_features = batch['atom_features'].view(len(batch['smiles']) * n_nodes, -1).to(self.accelerator)
            edges = get_adj_matrix(n_nodes, len(batch['smiles']), self.accelerator)

            pred = self.model(atom_features, 
                                    atom_pos, 
                                    edges, 
                                    edge_attr = None,
                                    n_nodes = n_nodes)

            if len(batch['features'][0]) > 0:
                pred = torch.cat((pred,batch['features'].float().to(pred)), dim = 1)

            pred = self.graph_dec(pred)

        elif self.model_name == 'naive_ffn':
            pred = self.model(batch['features'].float().to(self.accelerator))
        
        if self.classification:
            if self.loss_function == 'dirichlet':
                pred = self.readout(pred) + 1
            else:
                pred = self.readout(pred)

        return pred
    
    def training_step(self,batch,batch_idx):
        
        if self.task == 'classification':
            labels = batch['binary_label']
        elif self.task == 'regression':
            labels = batch['label']

        forward_output = self.forward(batch)

        if self.loss_function == 'TruncatedLoss':
            loss = self.loss_fn(forward_output, 
                                labels.unsqueeze(-1).float(),
                                batch['idx'])
        elif self.loss_function == 'dirichlet':            
            loss = dirichlet_class_loss(forward_output,
                                labels.unsqueeze(-1).float(),
                                self.lam)
            loss = loss.mean()

        else:
            loss = self.loss_fn(forward_output, 
                                labels.unsqueeze(-1).float())

        print(loss) 

        if self.classification:
            self.log(f'Train_{self.loss_function}_Loss',
                     loss,
                     on_step=True,
                     on_epoch=True,
                     batch_size = self.batch_size)

            if self.loss_function == 'dirichlet':
                batch_alphas = torch.reshape(
                        forward_output,
                        (forward_output.shape[0], forward_output.shape[1] // 2, 2),
                    )
                forward_output = batch_alphas[:, :, 1] / torch.sum(
                        batch_alphas, axis=2
                    )  # shape(data, tasks, 2)
                
                # alphas = np.array(batch_alphas)
                # S = torch.sum(batch_alphas, axis = 2)
                # num_classes = batch_alphas.shape[2]
                # u = num_classes / S

            logs = binary_classification_metric_return(y_pred = forward_output.float(),
                                                        y_true = labels.unsqueeze(-1).float(),
                                                        state = 'train')
        else:
            logs = regression_metric_return(y_pred = forward_output.view(-1).float(),
                                            y_true = labels.float(),
                                            state = 'train')

        print(logs)

        optimizer = self.optimizers()
        optimizer.zero_grad()

        self.manual_backward(loss)

        for name, param in self.model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    print(f"{name} nan gradient found")

        nn.utils.clip_grad_norm_(self.parameters(), self.gradient_clip)
        optimizer.step()

        if self.use_lr_scheduler:
            scheduler = self.lr_schedulers()
            scheduler.step()

        self.log_dict(logs,
                        on_step=True,
                        on_epoch=True,
                        batch_size=self.batch_size)
        
        self.log("lr", self.trainer.optimizers[0].param_groups[0]["lr"], batch_size=self.batch_size)

        return loss

    def validation_step(self, batch, batch_idx):
        
        if self.task == 'classification':
            labels = batch['binary_label']
            idx = batch['idx']
        elif self.task == 'regression':
            labels = batch['label']

        forward_output = self.forward(batch)

        self.validation_predictions.append(forward_output)
        self.validation_labels.append(labels.unsqueeze(-1))
        self.validation_idx = self.validation_idx + idx
        
        return forward_output, labels
    
    def on_validation_epoch_end(self):

        self.validation_predictions = torch.cat(self.validation_predictions)
        self.validation_labels = torch.cat(self.validation_labels)

        if self.loss_function == 'TruncatedLoss':
            loss = self.loss_fn(self.validation_predictions, 
                                self.validation_labels.float(),
                                self.validation_idx)
        elif self.loss_function == 'dirichlet':
            loss = dirichlet_class_loss(self.validation_predictions,
                                self.validation_labels.float(),
                                self.lam)
            loss = loss.mean()
        else:
            loss = self.loss_fn(self.validation_predictions, 
                                self.validation_labels.float())
        
        if self.classification:

            self.log(f'Validation_{self.loss_function}_Loss',
                     loss,
                     on_epoch=True,
                     batch_size = self.batch_size)
            
            if self.loss_function == 'dirichlet':
                batch_alphas = torch.reshape(
                        self.validation_predictions,
                        (self.validation_predictions.shape[0], self.validation_predictions.shape[1] // 2, 2),
                    )
                self.validation_predictions = batch_alphas[:, :, 1] / torch.sum(
                        batch_alphas, axis=2
                    )  # shape(data, tasks, 2)
                
            logs = binary_classification_metric_return(y_pred = self.validation_predictions,
                                                        y_true = self.validation_labels.float(),
                                                        state = 'validation_epoch')
        else: 
            logs = regression_metric_return(y_pred = self.validation_predictions,
                                            y_true = self.validation_labels.float(),
                                            state = 'validation_epoch')            

        print(logs)
        
        self.log_dict(logs, batch_size=self.batch_size)

        self.validation_predictions = []
        self.validation_labels = []
        self.validation_idx = []

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        return self(batch)

    def test_step(self, batch, batch_idx):

        labels = batch['label']
        smiles = batch['smiles']

        forward_output = self.forward(batch)

        self.test_labels.append(labels)
        self.test_predictions.append(forward_output)
        self.test_smiles.append(smiles)
        
        return forward_output, labels
    
    def on_test_epoch_end(self):
        
        test_labels = torch.flatten(torch.stack(self.test_labels, 0)).float()
        test_smiles = [element for sublist in self.test_smiles for element in sublist]

        if self.loss_function == 'dirichlet':
            test_output = torch.cat(self.test_predictions, 0)
            alphas = torch.reshape(test_output,
                                   [test_output.shape[0], test_output.shape[1] // 2, 2])
            test_predictions = alphas[:,:,1] / torch.sum(alphas, axis = 2)
            S = torch.sum(alphas, axis = 2)
            num_classes = alphas.shape[2]
            test_uncertainty = num_classes / S

            test_predictions = torch.flatten(test_predictions)
            test_uncertainty = torch.flatten(test_uncertainty).tolist()

            logs = binary_classification_metric_return(y_pred = test_predictions,
                                                        y_true = test_labels,
                                                        state = 'test')
            
            self.log_dict(logs, batch_size=self.batch_size)

            test_df = pd.DataFrame(
                {'SMILES':test_smiles,
                'predictions': test_predictions.tolist(),
                'uncertainty': test_uncertainty,
                'labels': test_labels.tolist()}
            )

        else:
            test_predictions = torch.flatten(torch.stack(self.test_predictions, 1)).tolist()

            test_df = pd.DataFrame(
                {'SMILES':test_smiles,
                'predictions': test_predictions,
                'labels': test_labels}
            )

        # if self.output_dir is not None:
        test_df.to_csv('/home/aw274/egnn/external_predictions.csv',
                       index = False)

    def configure_optimizers(self):

        if self.weight_decay is not None:
            optimizer = optim.AdamW(self.parameters(), 
                                lr=self.lr,
                                weight_decay=self.weight_decay,
                                )
        else:
            optimizer = optim.Adam(self.parameters(), 
                                lr=self.lr,
                                )
            
        if self.use_lr_scheduler:
            scheduler = {'scheduler':
                        Scheduler(optimizer=optimizer,
                                  dim_embed=self.hidden_nf,
                                  warmup_steps=self.warmup_steps),
                        'interval': 'step',
                        'frequency': 1,
                        'name': 'lr_rate_opt'
            }

            return [optimizer],[scheduler]

        return optimizer