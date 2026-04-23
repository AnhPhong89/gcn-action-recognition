"""
Checkpoint save / load utilities.
"""

import torch
from pathlib import Path


def save_checkpoint(state: dict,
                    save_dir: str,
                    filename: str = 'checkpoint.pt',
                    is_best: bool = False):
    """Save a training checkpoint.

    Args:
        state: Dict containing model_state_dict, optimizer_state_dict,
               epoch, best_acc, etc.
        save_dir: Directory to save checkpoints.
        filename: Checkpoint filename.
        is_best: If True, also save a copy as 'best_model.pt'.
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    filepath = save_dir / filename
    torch.save(state, filepath)

    if is_best:
        best_path = save_dir / 'best.pt'
        torch.save(state, best_path)


def load_checkpoint(checkpoint_path: str, model, optimizer=None, device='cpu'):
    """Load a checkpoint and restore model / optimizer states.

    Args:
        checkpoint_path: Path to the .pt checkpoint file.
        model: The model to load weights into.
        optimizer: (Optional) optimizer to restore state.
        device: Device to map tensors to.

    Returns:
        Dict with 'epoch' and 'best_acc' from the checkpoint.
    """
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    model.load_state_dict(ckpt['model_state_dict'])

    if optimizer is not None and 'optimizer_state_dict' in ckpt:
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])

    return {
        'epoch': ckpt.get('epoch', 0),
        'best_acc': ckpt.get('best_acc', 0.0),
    }
