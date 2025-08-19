import torch
import torch.nn as nn

class MesenCoder(nn.Module):
    def __init__(self,
                 input_dim: int,
                 n_source: int,
                 n_layers_enc: int,
                 n_layers_dec: int,
                 hidden_dim_enc: int,
                 hidden_dim_dec: int,
                 latent_dim: int = 1,
                 ) -> None:
        super().__init__()

        # feature encoder
        self.encoder = self.make_layers(
            input_dim = input_dim,
            n_layers = n_layers_enc,
            hidden_dim = hidden_dim_enc,
            output_dim = latent_dim,
            batchnorm_out = True)

        # source baseline
        self.baseline_src = nn.Embedding(
            num_embeddings = n_source,
            embedding_dim = input_dim)
        
        # conditional decoder (residual)
        input_dim_dec = (latent_dim + input_dim)
        self.decoder = self.make_layers(
            input_dim = input_dim_dec,
            n_layers = n_layers_dec,
            hidden_dim = hidden_dim_dec,
            output_dim = input_dim)
        
        # init weights
        self._initialize_weights()
        nn.init.xavier_uniform_(
            self.encoder[-2].weight,
            gain = nn.init.calculate_gain('tanh'))
        nn.init.normal_(
            self.baseline_src.weight,
            mean = 0., std = .05)

    def forward(self,
                x: torch.Tensor,
                src: torch.Tensor | None
                ) -> torch.Tensor:

        # feature encoder
        u = self.encoder(x)
        z = torch.tanh(u)

        if src is not None:
            # source baseline
            src = self.baseline_src(src)

            # conditional decoder (residual)
            z_cond = torch.cat((z, src), dim = -1)
            x_hat = src + self.decoder(z_cond)
            return x_hat
        else:
            return z

    def make_layers(self,
                    input_dim: int,
                    n_layers: int,
                    hidden_dim: int,
                    output_dim: int,
                    batchnorm_out: bool = False,
                    activation: type[nn.Module] = nn.ReLU,
                    ) -> nn.Sequential:
        layers = list()
        for _ in range(n_layers):
            layers.extend([nn.Linear(input_dim, hidden_dim), activation()])
            input_dim = hidden_dim
        layers.append(nn.Linear(input_dim, output_dim, bias = not batchnorm_out))
        if batchnorm_out:
            layers.append(nn.BatchNorm1d(output_dim, affine = False))
        return nn.Sequential(*layers)
    
    def _initialize_weights(self,
                            nonlinearity: str = 'relu'
                            ) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight, nonlinearity = nonlinearity)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
