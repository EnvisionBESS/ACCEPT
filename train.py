import pandas as pd
import argparse
import tqdm

import torch

from .data import load_data
from accept import ACCEPT


def info_nce_loss_from_logits(logits, scale = 0.1):
    """
    Compute the InfoNCE loss given logits.

    Args:
        logits: Tensor of shape [batch_size, batch_size]
        scale: Scalarparameter

    Returns:
        loss: Scalar tensor representing the loss
    """
    # Apply temperature scaling
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
          device):


    number_of_epochs = 8
    model.train()
    for epoch in range(number_of_epochs):
        bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        total_epoch_loss = 0

        for _ , operational in bar:
            # operational = Tuple[features, target]
            operational = operational[0] # just need the features rather than the target
            operational = {k: v.to(device) for k, v in operational.items()}
            # Fetch curves
            groups = operational['groups']
            target_tensors_physical = torch.stack([simulated_matches[i] for i in groups[:,0]])

            target_tensors_physical = target_tensors_physical.unsqueeze(2)
            target_tensors_physical = target_tensors_physical.to(device)

            simulated_queue = model.get_simulated_queue().to(device)

            physical_all = torch.cat([target_tensors_physical, simulated_queue], dim=0)
            #model.dequeue_and_enqueue(target_tensors_physical)
            optimizer.zero_grad()
            #targets_physical = torch.arange(batch_size, dtype=torch.long, device='cpu')

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
    parser.add_argument("--time_idx", type=str, required=True, help="Index column for time.")
    parser.add_argument("--target", type=str, required=True, help="Target column for prediction.")
    parser.add_argument("--groups", type=str, nargs='+', required=True, help="List of group identifiers.")
    parser.add_argument("--static_categoricals", type=str, nargs='+', required=True, help="List of static categorical features.")
    parser.add_argument("--static_reals", type=str, nargs='+', required=True, help="List of static real-valued features.")
    parser.add_argument("--time_varying_unknown_reals", type=str, nargs='+', required=True, help="List of time-varying unknown real-valued features.")
    parser.add_argument("--time_varying_known_reals", type=str, nargs='+', required=True, help="List of time-varying known real-valued features.")
    parser.add_argument("--lower_cycle_idx", type=int, default=100, help="Lower cycle index for data filtering.")
    parser.add_argument("--upper_cycle_idx", type=int, default=500, help="Upper cycle index for data filtering.")

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


    simualated_matches = pd.read_csv(matches_path)



    model = ACCEPT()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train()