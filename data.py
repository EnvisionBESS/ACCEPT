import pandas as pd
from typing import List, Dict, Tuple
import numpy as np

import torch
from torch.nn.utils import rnn
from pytorch_forecasting.data import TimeSeriesDataSet, DataLoader


def pad_sequences_to_length(sequences, target_length, pad_value=0):
    """
    Pad a list of sequences to a given target length.

    Args:
        sequences (list of Tensor): The list of tensor sequences.
        target_length (int): The fixed length to pad/truncate each sequence.
        pad_value (int): The value used for padding.

    Returns:
        Tensor: A tensor containing padded sequences with shape (batch_size, target_length).
    """
    # Create a new tensor to hold the padded sequences
    padded_sequences = torch.full((len(sequences), target_length), pad_value, dtype=sequences[0].dtype)

    # Iterate through each sequence and pad/truncate to the target length
    for i, seq in enumerate(sequences):
        end_idx = min(len(seq), target_length)
        padded_sequences[i, :end_idx] = seq[:end_idx]

    return padded_sequences



def collate_fn(
        batches: List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        """
        Collate function to combine items into mini-batch for dataloader.

        Args:
            batches (List[Tuple[Dict[str, torch.Tensor], torch.Tensor]]): List of samples generated with
                :py:meth:`~__getitem__`.

        Returns:
            Tuple[Dict[str, torch.Tensor], Tuple[Union[torch.Tensor, List[torch.Tensor]], torch.Tensor]: minibatch
        """
        # collate function for dataloader
        # lengths
        encoder_lengths = torch.tensor([batch[0]["encoder_length"] for batch in batches], dtype=torch.long)
        decoder_lengths = torch.tensor([batch[0]["decoder_length"] for batch in batches], dtype=torch.long)

        # ids
        decoder_time_idx_start = (
            torch.tensor([batch[0]["encoder_time_idx_start"] for batch in batches], dtype=torch.long) + encoder_lengths
        )
        decoder_time_idx = decoder_time_idx_start.unsqueeze(1) + torch.arange(decoder_lengths.max()).unsqueeze(0)
        groups = torch.stack([batch[0]["groups"] for batch in batches])

        # features
        encoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        encoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][:length] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        decoder_cont = rnn.pad_sequence(
            [batch[0]["x_cont"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )
        decoder_cat = rnn.pad_sequence(
            [batch[0]["x_cat"][length:] for length, batch in zip(encoder_lengths, batches)], batch_first=True
        )

        # target scale
        if isinstance(batches[0][0]["target_scale"], torch.Tensor):  # stack tensor
            target_scale = torch.stack([batch[0]["target_scale"] for batch in batches])
        elif isinstance(batches[0][0]["target_scale"], (list, tuple)):
            target_scale = []
            for idx in range(len(batches[0][0]["target_scale"])):
                if isinstance(batches[0][0]["target_scale"][idx], torch.Tensor):  # stack tensor
                    scale = torch.stack([batch[0]["target_scale"][idx] for batch in batches])
                else:
                    scale = torch.from_numpy(
                        np.array([batch[0]["target_scale"][idx] for batch in batches], dtype=np.float32),
                    )
                target_scale.append(scale)
        else:  # convert to tensor
            target_scale = torch.from_numpy(
                np.array([batch[0]["target_scale"] for batch in batches], dtype=np.float32),
            )

        # target and weight
        if isinstance(batches[0][1][0], (tuple, list)):
            target = [
                rnn.pad_sequence([batch[1][0][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
            encoder_target = [
                rnn.pad_sequence([batch[0]["encoder_target"][idx] for batch in batches], batch_first=True)
                for idx in range(len(batches[0][1][0]))
            ]
        else:
            target = rnn.pad_sequence([batch[1][0] for batch in batches], batch_first=True)
            encoder_target = rnn.pad_sequence([batch[0]["encoder_target"] for batch in batches], batch_first=True)

        if batches[0][1][1] is not None:
            weight = rnn.pad_sequence([batch[1][1] for batch in batches], batch_first=True)
        else:
            weight = None

        return (
            dict(
                encoder_cat=encoder_cat,
                encoder_cont=encoder_cont,
                encoder_target=encoder_target,
                encoder_lengths=encoder_lengths,
                decoder_cat=decoder_cat,
                decoder_cont=decoder_cont,
                decoder_target=target,
                decoder_lengths=decoder_lengths,
                decoder_time_idx=decoder_time_idx,
                groups=groups,
                target_scale=target_scale,
            ),
            (target, weight),
        )


def load_data(path: str,
              batch_size: int, 
              time_idx: str, 
              target: str,
              groups: List[str], 
              static_categoricals: List[str], 
              static_reals: List[str], 
              time_varying_unknown_reals: List[str],
              time_varying_known_reals: List[str],
              lower_cycle_idx: int = 100,
              upper_cycle_idx: int = 500,
              ) -> DataLoader:

    train_df = pd.read_csv(path)
    # Only want data starting at cycle 1
    training_op_total = None
    for i in range(lower_cycle_idx, upper_cycle_idx):
        tmp = train_df[train_df['Cycle'] <= i]

        min_encoder_length = i - 1
        max_encoder_length = i - 1
        min_prediction_length = 1
        max_prediction_length = 1

        training_op = TimeSeriesDataSet(
            tmp,
            time_idx=time_idx,
            target=target,
            group_ids=groups,
            static_categoricals = static_categoricals,
            static_reals = static_reals,
            min_encoder_length=min_encoder_length,  # keep encoder length long (as it is in the validation set)
            max_encoder_length=max_encoder_length,
            min_prediction_length=min_prediction_length,
            max_prediction_length=max_prediction_length,
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=time_varying_unknown_reals,
            time_varying_known_reals = time_varying_known_reals,
            add_relative_time_idx=True,
            add_target_scales=False,
            add_encoder_length=False,
            allow_missing_timesteps = True
        )

        if training_op_total is None:
            training_op_total = training_op
        else:
            training_op_total = torch.utils.data.ConcatDataset([training_op_total, training_op])

    train_dataloader = DataLoader(training_op_total, collate_fn=collate_fn, batch_size=batch_size)

    return train_dataloader
