"""
Learning rate scheduler factory.
"""

import torch.optim.lr_scheduler as lr_scheduler


def build_scheduler(optimizer, scheduler_cfg: dict):
    """Create a learning rate scheduler from config dict.

    Args:
        optimizer: The optimizer to schedule.
        scheduler_cfg: Dict with keys:
            - type: 'step', 'multistep', 'cosine', or 'plateau'
            - step_size (int): For StepLR.
            - milestones (list[int]): For MultiStepLR.
            - gamma (float): LR decay factor. Default 0.1.
            - T_max (int): For CosineAnnealingLR (usually = total epochs).
            - patience (int): For ReduceLROnPlateau.

    Returns:
        A PyTorch LR scheduler instance.
    """
    sched_type = scheduler_cfg.get('type', 'multistep')
    gamma = scheduler_cfg.get('gamma', 0.1)

    if sched_type == 'step':
        step_size = scheduler_cfg.get('step_size', 30)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif sched_type == 'multistep':
        milestones = scheduler_cfg.get('milestones', [30, 60])
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif sched_type == 'cosine':
        T_max = scheduler_cfg.get('T_max', 80)
        eta_min = scheduler_cfg.get('eta_min', 1e-6)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif sched_type == 'plateau':
        patience = scheduler_cfg.get('patience', 10)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=gamma, patience=patience, verbose=True
        )

    else:
        raise ValueError(f"Unknown scheduler type: {sched_type}. "
                         f"Choose from: step, multistep, cosine, plateau")
