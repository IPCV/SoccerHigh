import torch


class SimpleHead(torch.nn.Module):
    def __init__(
        self,
        ndim,
        num_classes
    ):
        super().__init__()

        self.fc = torch.nn.Linear(
            in_features = ndim,
            out_features = ndim
        )
        
        self.fc2 = torch.nn.Linear(
            in_features = ndim,
            out_features = num_classes
        )
        
        self.batch_norm =  torch.nn.BatchNorm1d(num_features=num_classes)
        
        self.nheads = 1

    def forward(self, x):
        x = self.fc(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        x = self.batch_norm(x.permute(0,2,1)).permute(0,2,1)
        return x
    
class TwoLayerLinearHead(torch.nn.Module):
    def __init__(
        self,
        ndim,
        num_classes
    ):
        super().__init__()

        self.head = torch.nn.Sequential(
            torch.nn.Linear(
                in_features = ndim,
                out_features = ndim
            ),
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5),
            torch.nn.LayerNorm(normalized_shape=ndim),
            torch.nn.Linear(
                in_features = ndim,
                out_features = num_classes
            )
        )

    def forward(self, x):
        return self.head(x)

class DSNetAFHead(torch.nn.Module):
    def __init__(
        self,
        ndim,
        nheads,
        num_classes,
        weights
    ):
        super().__init__()

        self.layer_norm = torch.nn.LayerNorm(ndim)

        self.fc = torch.nn.Sequential(
            torch.nn.Linear(ndim, ndim),
            torch.nn.ReLU(inplace=True),
            torch.nn.Dropout(0.1),
            torch.nn.LayerNorm(ndim)
        )
        
        heads = [
            {'cls': torch.nn.Linear(ndim, num_classes)}, 
            {'loc': torch.nn.Linear(ndim, 2 * num_classes)},
            {'action_cl': None},
            {'ctr': torch.nn.Linear(ndim, num_classes)}
        ]

        for i in range(nheads):
            for name, value in heads[i].items():
                setattr(self, name, value)

        setattr(self, 'nheads', nheads)
                
        self.weights = weights if weights else [1.0, 1.0, 1.0]

    def forward(self, x):
        x = self.layer_norm(x)

        x = self.fc(x)
        
        if hasattr(self, 'cls') and hasattr(self, 'loc') and hasattr(self, 'ctr'):
            return self.cls(x), self.loc(x), self.ctr(x)
        elif hasattr(self, 'cls') and hasattr(self, 'loc'):
             return self.cls(x), self.loc(x)
        else:
             return self.cls(x)
         
class DSNetAFwithActionsHead(DSNetAFHead):
    def __init__(self, ndim, nheads, num_classes, num_actions, weights, film):
        super().__init__(ndim, nheads, num_classes, weights)

        self.action_cl = torch.nn.Sequential(
            torch.nn.Conv1d(
                in_channels=ndim,
                out_channels=num_actions,
                padding=1,
                kernel_size=3,
                stride=1
            ),
            torch.nn.BatchNorm1d(num_features=num_actions)
        )

        if film:
            self.film_layer = FiLMLayer(condition_dim=num_actions, feature_dim=1)
        
    def forward(self, x):
        x_cl = self.action_cl(x.permute(0,2,1)).permute(0,2,1)

        x = self.layer_norm(x)
        
        x = self.fc(x)

        if hasattr(self, 'cls') and hasattr(self, 'loc') and hasattr(self, 'ctr'):
            return self.cls(x), self.loc(x), x_cl, self.ctr(x)
        elif hasattr(self, 'cls') and hasattr(self, 'loc') and not hasattr(self, 'film_layer'):
            return self.cls(x), self.loc(x) , x_cl
        elif hasattr(self, 'cls') and hasattr(self, 'loc') and hasattr(self, 'film_layer'):
            return self.film_layer(self.cls(x), x_cl), self.loc(x)
        else:
            return self.cls(x), x_cl
        
    def load_weights(self, weights, freeze=True):
        self.action_cl.load_state_dict(weights, strict=False)
        print("Spotting head weights succesfully loaded")

        if freeze:
            for param in self.action_cl.parameters():
                param.requires_grad = False

class FiLMLayer(torch.nn.Module):
    def __init__(self, condition_dim, feature_dim):
        super(FiLMLayer, self).__init__()
        self.feature_dim = feature_dim
        self.condition_to_gamma_beta = torch.nn.Linear(condition_dim, 2 * feature_dim)
    
    def forward(self, x, y):
        # Get gamma and beta from condition (Y)
        gamma_beta = self.condition_to_gamma_beta(y)
        gamma, beta = gamma_beta.chunk(2, dim=-1)  # Split into two parts along the feature dimension
        # Apply FiLM
        return gamma * x + beta