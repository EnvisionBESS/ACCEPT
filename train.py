import pandas as pd
import argparse
import tqdm

import torch
import torch.nn.functional as F

from .data import load_data
from accept import ACCEPT


def info_nce_loss_from_logits(logits, scale = 0.1):
    """
    Compute the InfoNCE loss given logits.

    Args:
        logits: Tensor of shape [batch_size, batch_size]
        scale: Scalar parameter

    Returns:
        loss: Scalar tensor representing the loss
    """
    # Apply scaling
    logits = logits / scale

    # Labels: correct indices
    batch_size = logits.size(0)
    labels = torch.arange(batch_size).long().to(logits.device)

    # Compute cross-entropy loss
    loss = F.cross_entropy(logits, labels)

    return loss

def train(train_dataloader,
          optimizer,
          simulated_matches,
          device,
          n_epochs):

    for epoch in range(n_epochs):
        print(f'Starting epoch: {epoch}')
        bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for _ , operational in bar:
            # operational = Tuple[features, target]
            operational = operational[0] # just need the features rather than the target
            operational = {k: v.to(device) for k, v in operational.items()}
            # Fetch curves
            groups = operational['groups']
            target_tensors_physical = torch.stack([simulated_matches[i] for i in groups[:,0]])

            target_tensors_physical = target_tensors_physical.unsqueeze(2) # Ensure 3-d as required by model
            target_tensors_physical = target_tensors_physical.to(device)

            # fetch additional negatives from queue and append
            simulated_queue = model.get_simulated_queue().to(device)
            physical_all = torch.cat([target_tensors_physical, simulated_queue], dim=0)

            optimizer.zero_grad()

            # Forward pass
            logits_operational, _ = model(operational, physical_all)

            # Enable gradient scaling to reduce memory usage
            with torch.cuda.amp.autocast(enabled=True):
                loss = info_nce_loss_from_logits(logits_operational)

            # Perform backward pass and optimization with gradient scaling
            # Create a GradScaler object
            scaler = torch.cuda.amp.GradScaler()

            # Scale the loss and perform backward pass
            scaler.scale(loss).backward()

            # Unscale the gradients and update model parameters
            scaler.step(optimizer)
            scaler.update()

            optimizer.step()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Load data with specific parameters.")

    # Adding arguments
    parser.add_argument("--path", type=str, required=True, help="Path to the data file.")
    parser.add_argument("--matches_path", type=str, required=True, help="Path to positive matches of curves")
    parser.add_argument("--time_idx", type=str, required=True, help="Index column for time.")
    parser.add_argument("--target", type=str, required=True, help="Target column for prediction.")
    parser.add_argument("--groups", type=str, nargs='+', required=True, help="List of group identifiers.")
    parser.add_argument("--static_categoricals", type=str, nargs='+', required=True, help="List of static categorical features.")
    parser.add_argument("--static_reals", type=str, nargs='+', required=True, help="List of static real-valued features.")
    parser.add_argument("--time_varying_unknown_reals", type=str, nargs='+', required=True, help="List of time-varying unknown real-valued features.")
    parser.add_argument("--time_varying_known_reals", type=str, nargs='+', required=True, help="List of time-varying known real-valued features.")
    parser.add_argument("--lower_cycle_idx", type=int, default=100, help="Lower cycle index for data filtering.")
    parser.add_argument("--upper_cycle_idx", type=int, default=500, help="Upper cycle index for data filtering.")
    parser.add_argument("--embed_dim", type=int, required=True, help="Embedding dimension size.")
    parser.add_argument("--negative_path", type=str, required=True, help="Path for negative samples.")
    parser.add_argument("--hidden_size", type=int, default=24, help="Size of the hidden layer.")
    parser.add_argument("--hidden_cont_size", type=int, default=64, help="Size of the continuous hidden layer.")
    parser.add_argument("--queue_size", type=int, default=1024, help="Size of the queue.")
    parser.add_argument("--hidden_size_conv", type=int, default=256, help="Size of the convolutional hidden layer.")
    parser.add_argument("--kernel_size", type=int, default=22, help="Kernel size for convolutions.")
    parser.add_argument("--attention_head_size", type=int, default=4, help="Size of the attention head.")
    parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate.")
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature parameter for softmax.")
    parser.add_argument("--lr", type=float, default=0.0001, help="Leaning rate")
    parser.add_argument("--n_epochs", type=int, default=100, help="Number of epochs")

    # Parse arguments
    args = parser.parse_args()

    # Call the function with parsed arguments
    train_dataloader = load_data(
        path=args.path, 
        time_idx=args.time_idx, 
        target=args.target, 
        groups=args.groups, 
        static_categoricals=args.static_categoricals, 
        static_reals=args.static_reals, 
        time_varying_unknown_reals=args.time_varying_unknown_reals, 
        time_varying_known_reals=args.time_varying_known_reals, 
        lower_cycle_idx=args.lower_cycle_idx, 
        upper_cycle_idx=args.upper_cycle_idx
    )

    simulated_matches = pd.read_csv(args.matches_path)

    model = ACCEPT(embed_dim=args.embed_dim, negative_path=args.negative_path, training_operational=None,
                   hidden_size=args.hidden_size, hidden_cont_size=args.hidden_cont_size,
                   queue_size=args.queue_size, hidden_size_conv=args.hidden_size_conv,
                   kernel_size=args.kernel_size, attention_head_size=args.attention_head_size,
                   dropout=args.dropout, temperature=args.temperature)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train(
        train_dataloader,
        optimizer,
        simulated_matches,
        args.device,
        args.n_epochs
    )