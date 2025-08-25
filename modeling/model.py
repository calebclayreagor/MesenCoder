import torch
import torch.nn as nn
import torch.nn.functional as F

class MesenCoder(nn.Module):
    def __init__(self,
                 n_feature: int,
                 n_source: int,
                 hidden_dim: int,
                 latent_dim_src: int,
                 latent_dim: int = 1,
                 activation: type[nn.Module] = nn.ReLU
                 ) -> None:
        super().__init__()
        self.activation = activation

        # vanilla encoder (+ BN, tied input)
        self.encoder = nn.Sequential(
            nn.Linear(n_feature, hidden_dim), activation(),
            nn.Linear(hidden_dim, latent_dim),
            nn.BatchNorm1d(latent_dim, affine = False))
        
        # conditional decoder (+ tied output)
        in_dim_dec = latent_dim + latent_dim_src
        self.decoder = nn.Sequential(
            nn.Linear(in_dim_dec, hidden_dim), activation(),
            TiedLinear(self.encoder[0]))

        # source embedding
        self.embed_src = nn.Embedding(
            num_embeddings = n_source,
            embedding_dim = latent_dim_src)
        
        # init weights
        self._initialize_weights()
        nn.init.normal_(
            self.embed_src.weight,
            mean = 0., std = .05)

    def forward(self,
                x: torch.Tensor,
                src: torch.Tensor,
                eps: float = 1e-6
                ) -> tuple[torch.Tensor,
                           torch.Tensor]:

        # encoder
        u = self.encoder(x)
        z = torch.log1p(F.softplus(u) + eps)

        # source embedding
        v = self.embed_src(src)

        # decoder
        h = torch.cat((z, v), dim = -1)
        x_hat = self.decoder(h)
        return x_hat, z

    def _initialize_weights(self) -> None:
        if self.activation is nn.ReLU:
            nonlinearity = 'relu'
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(
                    m.weight,
                    nonlinearity = nonlinearity)
                nn.init.zeros_(m.bias)

class TiedLinear(nn.Module):
    def __init__(self,
                 tied_to: nn.Linear
                 ) -> None:
        super().__init__()
        self.tied_to = tied_to
        self.bias = nn.Parameter(torch.zeros(tied_to.in_features))

    def forward(self,
                x: torch.Tensor
                ) -> torch.Tensor:
        return F.linear(x, self.tied_to.weight.t(), self.bias)
