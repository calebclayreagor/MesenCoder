import torch.nn as nn

class MesNet(nn.Module):
    def __init__(self, input_dim, target_dim, num_classes, hidden_dim, n_layers, latent_dim = 2):
        super().__init__()
        self.encoder = self.make_layers(input_dim, hidden_dim, n_layers, latent_dim)
        self.regression_head = nn.Linear(latent_dim, target_dim)
        self.classifier_head = nn.Linear(latent_dim, num_classes)
        self._initialize_weights()

    def forward(self, x):
        latent = self.encoder(x)
        y_pred = self.regression_head(latent)
        c_logits = self.classifier_head(latent)
        return y_pred, c_logits, latent
    
    def make_layers(self, input_dim, hidden_dim, n_layers, latent_dim, activation = nn.ReLU):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if i < n_layers - 1:
                layers.append(activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, latent_dim))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
