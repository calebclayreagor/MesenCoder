import torch
import torch.nn as nn
import torch.nn.functional as F

class MesNet(nn.Module):
    def __init__(self, input_dim, n_source, hidden_dim, n_layers, latent_dim):
        super().__init__()
        self.n_source = n_source
        self.latent_dim = latent_dim
        enc_in_dim = input_dim + n_source
        dec_in_dim = latent_dim + n_source
        self.encoder = self.make_layers(enc_in_dim, hidden_dim, n_layers, latent_dim)
        self.decoder = self.make_layers(dec_in_dim, hidden_dim, n_layers, input_dim)
        self._initialize_weights()

    def forward(self, x, src):
        src = F.one_hot(src, self.n_source).float()
        x_cond = torch.cat((x, src), dim = -1)
        z = self.encoder(x_cond)
        z_cond = torch.cat((z, src), dim = -1)
        x_hat = self.decoder(z_cond)
        return x_hat, z
    
    def make_layers(self, input_dim, hidden_dim, n_layers, output_dim, activation = nn.ReLU):
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if i < n_layers - 1:
                layers.append(activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
