"""
Training entry point for ST-GCN Action Recognition.

Usage:
    python scripts/train.py                          # default config
    python scripts/train.py --config configs/base.yaml
    python scripts/train.py --debug                  # quick test with 100 samples
    python scripts/train.py --resume runs/exp/checkpoints/best_model.pt
"""

import argparse
import sys
from pathlib import Path

import yaml
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import STGCNModel, STGCNTwoStreamModel
from src.data import build_dataloaders
from src.training import Trainer, build_loss, build_scheduler
from src.utils import set_seed, setup_logger


def load_config(config_path: str) -> dict:
    """Load YAML configuration file."""
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def build_model(cfg: dict, device: str) -> torch.nn.Module:
    """Instantiate model from config."""
    model_cfg = cfg['model']
    num_classes = cfg['data']['num_classes']

    model_type = model_cfg.get('type', 'stgcn')
    kwargs = {}
    if model_cfg.get('dropout', 0) > 0:
        kwargs['dropout'] = model_cfg['dropout']

    if model_type == 'stgcn':
        model = STGCNModel(
            in_channels=model_cfg['in_channels'],
            num_class=num_classes,
            graph_args=model_cfg['graph_args'],
            edge_importance_weighting=model_cfg['edge_importance_weighting'],
            **kwargs,
        )
    elif model_type == 'stgcn_twostream':
        model = STGCNTwoStreamModel(
            in_channels=model_cfg['in_channels'],
            num_class=num_classes,
            graph_args=model_cfg['graph_args'],
            edge_importance_weighting=model_cfg['edge_importance_weighting'],
            **kwargs,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    return model.to(device)


def build_optimizer(model, cfg: dict):
    """Create optimizer from config."""
    train_cfg = cfg['training']
    lr = train_cfg['learning_rate']
    wd = train_cfg['weight_decay']

    opt_type = train_cfg.get('optimizer', 'sgd')
    if opt_type == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=train_cfg.get('momentum', 0.9),
            nesterov=train_cfg.get('nesterov', True),
            weight_decay=wd,
        )
    elif opt_type == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
    elif opt_type == 'adamw':
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=wd,
        )
    else:
        raise ValueError(f"Unknown optimizer: {opt_type}")

    return optimizer


def main():
    # ── Args ───────────────────────────────────────────────────
    parser = argparse.ArgumentParser(description='Train ST-GCN Action Recognition')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--debug', action='store_true',
                        help='Quick run with 100 samples')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # ── Config ─────────────────────────────────────────────────
    cfg = load_config(args.config)

    # CLI overrides
    if args.debug:
        cfg['debug'] = True
    if args.resume:
        cfg['resume'] = args.resume

    debug = cfg.get('debug', False)
    seed = cfg.get('seed', 42)

    # ── Setup ──────────────────────────────────────────────────
    set_seed(seed)

    output_dir = cfg['output']['dir']
    logger = setup_logger('gcn', log_dir=output_dir)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Device: {device}")
    logger.info(f"Config: {args.config}")
    if debug:
        logger.info("⚡ DEBUG MODE: using only 100 samples")

    # ── Data ───────────────────────────────────────────────────
    dl_cfg = cfg['dataloader']
    train_cfg = cfg['training']

    train_loader, val_loader = build_dataloaders(
        processed_dir=cfg['data']['processed_dir'],
        batch_size=train_cfg['batch_size'],
        num_workers=dl_cfg.get('num_workers', 2),
        pin_memory=dl_cfg.get('pin_memory', True),
        normalize=dl_cfg.get('normalize', True),
        window_size=dl_cfg.get('window_size', -1),
        debug=debug,
    )
    logger.info(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    # ── Model ──────────────────────────────────────────────────
    model = build_model(cfg, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg['model']['type']} | "
                f"Params: {total_params:,} total, {trainable_params:,} trainable")

    # ── Loss, Optimizer, Scheduler ─────────────────────────────
    loss_cfg = train_cfg.get('loss', {})
    criterion = build_loss(loss_cfg.get('type', 'cross_entropy'),
                           **{k: v for k, v in loss_cfg.items() if k != 'type'})

    optimizer = build_optimizer(model, cfg)

    scheduler = None
    if 'scheduler' in train_cfg:
        scheduler = build_scheduler(optimizer, train_cfg['scheduler'])

    # ── Trainer ────────────────────────────────────────────────
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        output_dir=output_dir,
        logger=logger,
        num_epochs=train_cfg['epochs'],
        save_every=cfg['output'].get('save_every', 10),
        use_amp=cfg.get('use_amp', False),
    )

    # ── Resume ─────────────────────────────────────────────────
    if cfg.get('resume'):
        trainer.resume(cfg['resume'])

    # ── Train ──────────────────────────────────────────────────
    best_acc = trainer.fit()
    logger.info(f"🏁 Final best accuracy: {best_acc:.2f}%")


if __name__ == '__main__':
    main()
