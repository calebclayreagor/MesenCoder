import torch
import torch.nn as nn
import torch.nn.functional as F

class MesNet(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_source: int,
                 hidden_dim: int,
                 n_layers: int,
                 latent_dim: int
                 ) -> None:
        super().__init__()
        self.n_source = n_source
        input_dim_enc = input_dim + n_source
        input_dim_dec = latent_dim + n_source

        # conditional autoencoder
        self.encoder = self.make_layers(
            input_dim = input_dim_enc,
            hidden_dim = hidden_dim,
            n_layers = n_layers,
            output_dim = latent_dim)
        self.decoder = self.make_layers(
            input_dim = input_dim_dec,
            hidden_dim = hidden_dim,
            n_layers = n_layers,
            output_dim = input_dim)
        
        # scaling (x/src/z/out)
        self.scale_x = nn.Parameter(torch.zeros(input_dim))
        self.scale_src = nn.Parameter(torch.zeros(n_source))
        self.scale_z = nn.Parameter(torch.zeros(latent_dim))
        self.scale_out = nn.Parameter(torch.zeros(input_dim))

        # correction (src/z)
        self.residual_src = nn.Linear(n_source, n_source)
        self.residual_z = nn.Linear(latent_dim, latent_dim)

        # noise (x/src/z)
        self.mu_x = nn.Parameter(torch.zeros(input_dim))
        self.mu_src = nn.Linear(n_source, n_source)
        self.mu_z = nn.Parameter(torch.zeros(latent_dim))
        self.logvar_x = nn.Parameter(torch.zeros(input_dim))
        self.logvar_src = nn.Linear(n_source, n_source)
        self.logvar_z = nn.Parameter(torch.zeros(latent_dim))
        self._initialize_weights()

    def forward(self, x: torch.Tensor, src: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        
        # features (+ noise/scaling)
        std = torch.exp(.5 * self.logvar_x)
        eps = torch.randn_like(x)
        x = x + self.mu_x + std * eps
        x = x * self.scale_x.exp()

        # domain labels (+ noise/correction/scaling)
        src = F.one_hot(src, self.n_source).float()
        std = torch.exp(.5 * self.logvar_src(src))
        eps = torch.randn_like(src)
        src = src + self.mu_src(src) + std * eps
        src = src + self.residual_src(src)
        src = src * self.scale_src.exp()

        # conditional encoder
        x_cond = torch.cat((x, src), dim = -1)
        z = self.encoder(x_cond)

        # latents (+ noise/correction/scaling/constraint)
        std = torch.exp(.5 * self.logvar_z)
        eps = torch.randn_like(z)
        z = z + self.mu_z + std * eps
        z = z + self.residual_z(z)
        z = z * self.scale_z.exp()
        z_norm = torch.norm(z, dim = -1, keepdim = True)
        z_scale = torch.log1p(z_norm)
        z = z * (z_scale / (z_norm + 1e-9))

        # conditional decoder (+ scaling)
        z_cond = torch.cat((z, src), dim = -1)
        x_hat = self.decoder(z_cond)
        x_hat = x_hat * self.scale_out.exp()
        return x_hat, z

    def make_layers(self,
                    input_dim: int,
                    hidden_dim: int,
                    n_layers: int,
                    output_dim: int,
                    activation: type[nn.Module] = nn.ReLU
                    ) -> nn.Sequential:
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(input_dim, hidden_dim))
            if i < n_layers - 1:
                layers.append(activation())
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                nn.init.zeros_(m.bias)
