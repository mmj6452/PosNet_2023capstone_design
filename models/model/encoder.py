from torch import nn

from models.blocks.encoder_layer import EncoderLayer


class Encoder(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, n_layers, drop_prob, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  drop_prob=drop_prob)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):

        for layer in self.layers:
            x = layer(x, src_mask).to(self.device)

        return x