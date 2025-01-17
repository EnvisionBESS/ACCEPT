from typing import Dict
import os
import numpy as np
import pandas as pd

import torch
from torch import nn
from pytorch_forecasting.data import TimeSeriesDataSet

from .embeddings import TemporalFusionTransformer, ConvModule

PHYSICAL_SEQ_LEN = os.get_env('PHYSICAL_SEQ_LEN')

class ACCEPT(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 negative_path: str,
                 training_operational: TimeSeriesDataSet,
                 hidden_size: int,
                 hidden_cont_size: int,
                 queue_size: int,
                 hidden_size_conv: int,
                 kernel_size: int,
                 attention_head_size: int,
                 dropout: float,
                 temperature: float
                 ):
        super().__init__()
        self.temperature = temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / self.temperature))
        self.negative_path = negative_path

        self.operational_encoding = TemporalFusionTransformer.from_dataset(
                training_operational,
                hidden_size=hidden_size,
                attention_head_size=attention_head_size,
                dropout=dropout,
                hidden_continuous_size=hidden_cont_size,  # set to <= hidden_size
                embed_size = embed_dim
            )

        self.simulated_encoding = ConvModule(
            1,
            hidden_size_conv,
            kernel_size,
            embed_dim,)


        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.operational_encoding.to(self.device)
        self.simualated_encoding.to(self.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self._initialize_simulated_queue(queue_size)


    def _initialize_simulated_queue(self, queue_size):
        self.queue_size = queue_size
        df = pd.read_csv(self.negative_path, index_col = 0).reset_index()
        df = df[:PHYSICAL_SEQ_LEN]
        self.register_buffer("simulated_queue", torch.Tensor(df.values))
        self.simulated_queue = self.simulated_queue.unsqueeze(2) # make 3d


    def get_simulated_queue(self):
        # Randomly select queue_size indices for columns
        random_indices = torch.randint(0, self.simulated_queue.size(1), (self.queue_size,))
        # Select the columns using the random indices
        selected_curves = self.simulated_queue[:, random_indices]

        return selected_curves.permute(1,0,2)

    def encode_operational(self, x):
        return self.operational_encoding(x)

    def encode_simualted(self, x):
        return self.simulated_encoding(x)

    def forward(self,
                operational: Dict[str, Dict[str, torch.Tensor]],
                simulated: Dict[str, Dict[str, torch.Tensor]]):

        operational_features = self.encode_operational(operational)
        simulated_features = self.encode_simualted(simulated)

        # normalized features
        operational_features = operational_features / operational_features.norm(dim=1, keepdim=True)
        simulated_features = simulated_features / simulated_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_operational = logit_scale * operational_features @ simulated_features.t()
        logits_per_simulated = logits_per_operational.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_operational, logits_per_simulated