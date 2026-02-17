from __future__ import annotations

import torch
import torch.nn as nn


class MaskedVideoModel(nn.Module):
    def __init__(
        self,
        N: int,
        D: int,
        d_model: int = 256,
        n_layers: int = 4,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.N = N
        self.D = D
        self.d_model = d_model

        self.in_proj = nn.Linear(D, d_model)

        # Learned position embedding for N token positions
        self.pos_embed = nn.Embedding(N, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=int(d_model * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        # Learned mask token in latent space for slots we didn't encode
        self.mask_token = nn.Parameter(torch.zeros(d_model))

        # Simple decoder: latent -> pixel tubelet vector
        self.out_proj = nn.Sequential(
            nn.Linear(d_model, int(d_model * mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(d_model * mlp_ratio), D),
        )

        nn.init.normal_(self.mask_token, std=0.02)

        self.cls_token = nn.Parameter(torch.zeros(d_model))
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, tokens_visible: torch.Tensor, ids_visible: torch.Tensor) -> torch.Tensor:
        """
        tokens_visible: [B, Nv, D]
        ids_visible:    [B, Nv] positions in [0..N-1]
        returns pred_tokens: [B, N, D]
        """
        B, Nv, D = tokens_visible.shape
        assert D == self.D
        assert ids_visible.shape == (B, Nv)

        x = self.in_proj(tokens_visible)  # [B, Nv, d_model]
        pos = self.pos_embed(ids_visible) # [B, Nv, d_model]
        x = x + pos

        x = self.encoder(x)  # [B, Nv, d_model]

        # Build full latent grid initialized as mask_token
        full = self.mask_token.view(1, 1, -1).expand(B, self.N, -1).clone()

        # Scatter visible latents back into their positions
        full.scatter_(dim=1, index=ids_visible.unsqueeze(-1).expand(-1, -1, self.d_model), src=x)

        pred = self.out_proj(full)  # [B, N, D]
        return pred

    def encode_visible(self, tokens_visible: torch.Tensor, ids_visible: torch.Tensor) -> torch.Tensor:
        """
        Returns latent tokens INCLUDING CLS at position 0:
          x: [B, 1+Nv, d_model]
        """
        B, Nv, D = tokens_visible.shape
        x = self.in_proj(tokens_visible)
        x = x + self.pos_embed(ids_visible)

        cls = self.cls_token.view(1, 1, -1).expand(B, 1, -1)
        x = torch.cat([cls, x], dim=1)  # [B, 1+Nv, d_model]

        x = self.encoder(x)
        return x
