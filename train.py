import pandas as pd

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
          ):
    device = 


    number_of_epochs = 8
    model.train()
    for epoch in range(number_of_epochs):
        bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        total_epoch_loss = 0

        for _ , operational in bar:
            # operational = Tuple[features, target]
            operational = operational[0] # just need the features rather than the target
            operational = {k: v.to(device) for k, v in operational.items()}
            # Fetch real physically generated curves and append them to list
            groups = operational['groups']
            target_tensors_physical = torch.stack([simulated_matches[i] for i in groups[:,0]])

            target_tensors_physical = target_tensors_physical.unsqueeze(2)
            target_tensors_physical = target_tensors_physical.to(device)

            physical_queue = model.get_physical_queue().to(device)

            physical_all = torch.cat([target_tensors_physical, physical_queue], dim=0)
            #model.dequeue_and_enqueue(target_tensors_physical)
            optimizer.zero_grad()
            #targets_physical = torch.arange(batch_size, dtype=torch.long, device='cpu')

            # Forward pass
            logits_operational, logits_physical = model(operational, physical_all)

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

    train_dataloader = load_data()
    simualated_matches = pd.read_csv(matches_path)
    model = ACCEPT()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    train()