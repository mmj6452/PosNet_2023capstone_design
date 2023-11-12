
import torch
from torch import nn

from models.model.encoder import Encoder
from models.layers.position_wise_feed_forward import PositionwiseFeedForward
from models.layers.position_wise_feed_forward import Mlp

class Transformer(nn.Module):
    def __init__(self, input_size=512, output_size=3, d_model=512, n_head=8, generator_input_size=512*200,
                 ffn_hidden=2048, n_layers=6, drop_prob=0.1, device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")):
        super().__init__()
        self.device = device
        self.encoder = Encoder(d_model=d_model,
                               n_head=n_head,
                               ffn_hidden=ffn_hidden,
                               drop_prob=drop_prob,
                               n_layers=n_layers,
                               device=device).to(self.device)
        self.generator = Mlp(generator_input_size, d_model_1=output_size, hidden=ffn_hidden, drop_prob=drop_prob).to(self.device)
        self.embed = nn.Linear(input_size, d_model)

    def forward(self, src):
        print(src.shape)
        embed_src = self.embed(src)
        enc_src = self.encoder(embed_src, None)
        enc_src = enc_src.view(enc_src.size(0), -1)
        print(enc_src.shape)
        output = self.generator(enc_src)
        return output
