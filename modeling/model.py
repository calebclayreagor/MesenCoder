import torch
import torch.nn as nn
import torch.nn.functional as F

class MesNet(nn.Module):
    def __init__(self, input_dim, target_dim, num_classes, hidden_dim, n_layers, latent_dim = 2):
        super().__init__()
        self.encoder = self.make_layers(input_dim, hidden_dim, n_layers)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        self.regression_head = nn.Linear(latent_dim, target_dim)
        self.classifier_head = nn.Linear(latent_dim, num_classes)
        self._initialize_weights()

    def forward(self, x, deterministic = False, logvar_min = -3.):
        # variational encoder
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        logvar = logvar_min + F.softplus(logvar - logvar_min)
        if deterministic:
            z = mu
        else:
            std = torch.exp(.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std

        # regression & classification
        y_pred = self.regression_head(z)
        c_logits = self.classifier_head(z)
        return y_pred, c_logits, mu, logvar, z
    
    def make_layers(self, input_dim, hidden_dim, n_layers, activation = nn.ReLU):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if i < n_layers - 1:
                layers.append(activation())
            input_dim = hidden_dim
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
