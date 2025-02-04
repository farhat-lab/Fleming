import torch
import torch.nn as nn

class naive_fnn(nn.Module):
    # Naive Feedforward Neural Network that only intakes the extra features as the input

    def __init__(self, in_node_nf, hidden_nf, out_node_nf,  act_fn = nn.SiLU(), n_layers = 10,dropout=0.0):
        super().__init__()
        
        self.input_layer = nn.Sequential(*[nn.Linear(in_node_nf, hidden_nf),
                                           nn.Dropout(dropout),
                                           act_fn,
                                           ])
        
        hidden_module = []
        for i in range(n_layers):
            hidden_module.append(nn.Linear(hidden_nf, hidden_nf))
            hidden_module.append(nn.Dropout(dropout))
            hidden_module.append(act_fn)

        self.hidden_layer = nn.Sequential(*hidden_module)

        self.readout = nn.Linear(hidden_nf, out_node_nf)

    def forward(self, extra_features):

        output = self.input_layer(extra_features)
        output = self.hidden_layer(output)
        output = self.readout(output)

        return output

