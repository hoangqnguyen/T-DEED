import torch
import matplotlib.pyplot as plt
import numpy as np

def convert_target_to_prediction_shape(target, P):
    """
    Convert the target tensor from shape (B, T, 2) to match the shape (B, T, P, P, 3) with offsets from patch centers.
    
    Parameters:
    - target: Tensor of shape (B, T, 2) where the last dimension represents (x, y) coordinates in range [0, 1].
    - P: Number of patches per dimension.

    Returns:
    - Tensor of shape (B, T, P, P, 3) where the last dimension represents (object_presence, x_offset, y_offset).
    """
    B, T, _ = target.shape
    
    # Initialize the output tensor with zeros
    output = torch.zeros((B, T, P, P, 3))

    # Compute the size of each patch in normalized coordinates
    patch_size = 1.0 / P

    # Compute patch indices (vectorized)
    x_idx = torch.clamp((target[..., 0] * P).long(), max=P-1)
    y_idx = torch.clamp((target[..., 1] * P).long(), max=P-1)

    # Compute patch centers (vectorized)
    patch_centers_x = (x_idx.float() + 0.5) * patch_size
    patch_centers_y = (y_idx.float() + 0.5) * patch_size

    # Compute offsets from patch centers (vectorized)
    x_offset = target[..., 0] - patch_centers_x
    y_offset = target[..., 1] - patch_centers_y

    # Set object presence flag and offsets in the output tensor
    object_presence = ((target[..., 0] != 0) | (target[..., 1] != 0)).float()
    output[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 0] = object_presence
    output[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 1] = x_offset
    output[torch.arange(B).unsqueeze(1), torch.arange(T), x_idx, y_idx, 2] = y_offset

    return output

def visualize_prediction_grid(target, output, P):
    """
    Visualizes the original target positions and the converted offsets on a grid of patches.
    
    Parameters:
    - target: Tensor of shape (B, T, 2) where the last dimension represents (x, y) coordinates in range [0, 1].
    - output: Tensor of shape (B, T, P, P, 3) where the last dimension represents (object_presence, x_offset, y_offset).
    - P: Number of patches per dimension.
    """
    B, T, _ = target.shape
    patch_size = 1.0 / P

    for b in range(B):
        fig, axes = plt.subplots(1, T, figsize=(15, 5))
        
        for t in range(T):
            ax = axes[t]
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect('equal')

            # Draw the patch grid
            for i in range(P + 1):
                ax.axhline(i * patch_size, color='gray', linestyle='--', linewidth=0.5)
                ax.axvline(i * patch_size, color='gray', linestyle='--', linewidth=0.5)

            # Plot the original target position
            x, y = target[b, t]
            ax.plot(x, y, 'ro', label='Original Position')

            # Skip if there is no object
            if x == 0 and y == 0:
                continue
            
            # Calculate patch indices
            x_idx = min(int(x * P), P - 1)
            y_idx = min(int(y * P), P - 1)

            # Plot the patch center
            patch_center_x = (x_idx + 0.5) * patch_size
            patch_center_y = (y_idx + 0.5) * patch_size
            ax.plot(patch_center_x, patch_center_y, 'go', label='Patch Center')

            # Plot the offset position within the patch
            x_offset = output[b, t, x_idx, y_idx, 1]
            y_offset = output[b, t, x_idx, y_idx, 2]
            pred_x = patch_center_x + x_offset
            pred_y = patch_center_y + y_offset
            ax.plot(pred_x, pred_y, 'bx', label='Predicted Offset Position')

            ax.legend()
            ax.set_title(f'Frame {t + 1}')

        plt.suptitle(f'Batch {b + 1}')
        plt.show()

# # Example usage with batch size 2
# P = 4  # Number of patches per dimension
# target = torch.tensor([
#     [[0.1, 0.2], [0.5, 0.5], [0.7, 0.8]],  # First batch
#     [[0.3, 0.3], [0.0, 0.0], [0.9, 0.9]]   # Second batch
# ])  # Example target tensor with batch size 2
# output = convert_target_to_prediction_shape(target, P)

# # Visualize the results
# visualize_prediction_grid(target, output, P)
