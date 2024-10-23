from typing import Dict

import numpy as np
import pandas as pd
import torch
from torch import nn

from pytorch_forecasting.data import TimeSeriesDataSet

from .embeddings import TemporalFusionTransformer, TimeSeriesTransformer

class ACCEPT(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 training_operational: TimeSeriesDataSet,
                 queue_size: int = 24
                 ):
        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.operational_encoding = TemporalFusionTransformer.from_dataset(
                training_operational,
                hidden_size=24,
                attention_head_size=4,
                dropout=0.15,
                hidden_continuous_size=20,  # set to <= hidden_size
                embed_size = embed_dim
            )

        self.physical_encoding = TimeSeriesTransformer(
            input_size = 1,
            d_model = embed_dim,
            nhead = 4,
            num_layers = 2,
            dim_feedforward = 32,
            dropout = 0.1
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.operational_encoding.to(self.device)
        self.physical_encoding.to(self.device)

        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self._initialize_physical_queue(queue_size)


    def _initialize_physical_queue(self, queue_size):
        self.queue_size = queue_size
        df = pd.read_csv('samples/negatives.csv', index_col = 0)
        self.register_buffer("physical_queue", torch.Tensor(df.values))
        self.physical_queue = self.physical_queue.unsqueeze(2) # make 3d


    def get_physical_queue(self):
        # Randomly select queue_size indices for columns
        random_indices = torch.randint(0, self.physical_queue.size(1), (self.queue_size,))
        # Select the columns using the random indices
        selected_curves = self.physical_queue[:, random_indices]

        return selected_curves.permute(1,0,2)

    def encode_operational(self, x):
        return self.operational_encoding(x)

    def encode_physical(self, x):
        return self.physical_encoding(x)

    def forward(self,
                operational: Dict[str, Dict[str, torch.Tensor]],
                physical: Dict[str, Dict[str, torch.Tensor]]):

        operational_features = self.encode_operational(operational)
        physical_features = self.encode_physical(physical)

        # normalized features
        operational_features = operational_features / operational_features.norm(dim=1, keepdim=True)
        physical_features = physical_features / physical_features.norm(dim=1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_operational = logit_scale * operational_features @ physical_features.t()
        logits_per_physical = logits_per_operational.t()

        # shape = [global_batch_size, global_batch_size]
        return logits_per_operational, logits_per_physical