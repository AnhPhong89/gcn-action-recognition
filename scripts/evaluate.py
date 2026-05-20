"""
Evaluation script for ST-GCN Action Recognition.

Loads a trained model and evaluates it on a dataset (e.g., validation set),
calculating accuracy, precision, recall, f1-score, and plotting a confusion matrix.

Usage:
    python scripts/evaluate.py --checkpoint runs/exp/checkpoints/best.pt
    python scripts/evaluate.py --checkpoint runs/exp/checkpoints/best.pt --config configs/base.yaml
"""

import argparse
import sys
from pathlib import Path
import time
import csv

import numpy as np
import yaml
import torch
from sklearn.metrics import classification_report, accuracy_score

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models import STGCNModel, STGCNTwoStreamModel
from src.data.dataloader import build_dataloader
from src.evaluation.visualize import plot_confusion_matrix
from src.utils import set_seed, setup_logger


def load_config(config_path: str) -> dict:
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    return cfg


def load_model(checkpoint_path: str, cfg: dict, device: str) -> torch.nn.Module:
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

    print(f"Loading weights from {checkpoint_path}...")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state)
    model.to(device).eval()
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate ST-GCN Action Recognition Model')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint (.pt file)')
    parser.add_argument('--config', type=str, default='configs/base.yaml',
                        help='Path to YAML config file')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'],
                        help='Which dataset split to evaluate on (e.g., val)')
    parser.add_argument('--out-dir', type=str, default='runs/evaluation',
                        help='Directory to save evaluation results (e.g., confusion matrix)')
    parser.add_argument('--debug', action='store_true',
                        help='Quick run with a subset of data')
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = cfg.get('seed', 42)
    set_seed(seed)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    logger = setup_logger('evaluate', log_dir=str(out_dir))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Evaluation starting on device: {device}")
    
    # --- Data Loading ---
    processed_dir = Path(cfg['data']['processed_dir'])
    data_path = str(processed_dir / f"{args.split}_data.npy")
    label_path = str(processed_dir / f"{args.split}_label.pkl")
    
    dl_cfg = cfg['dataloader']
    class_names = cfg['data'].get('class_names', [])
    
    logger.info(f"Loading '{args.split}' data from: {data_path}")
    
    dataloader = build_dataloader(
        data_path=data_path,
        label_path=label_path,
        batch_size=cfg['training'].get('batch_size', 32),
        shuffle=False, # No need to shuffle for evaluation
        num_workers=dl_cfg.get('num_workers', 2),
        pin_memory=dl_cfg.get('pin_memory', True),
        drop_last=False,
        random_choose=False, # Disable augmentations for evaluation
        random_shift=False,
        random_move=False,
        window_size=dl_cfg.get('window_size', -1),
        normalize=dl_cfg.get('normalize', True),
        debug=args.debug,
    )
    
    # --- Model Loading ---
    model = load_model(args.checkpoint, cfg, device)
    
    # --- Evaluation Loop ---
    all_preds = []
    all_labels = []
    
    logger.info("Running inference...")
    start_time = time.time()
    
    with torch.no_grad():
        for batch_idx, (data, label) in enumerate(dataloader):
            data = data.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)
            
            output = model(data)
            
            # Get class predictions
            preds = output.argmax(dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Processed {batch_idx + 1}/{len(dataloader)} batches")
                
    eval_time = time.time() - start_time
    logger.info(f"Inference completed in {eval_time:.2f}s")
    
    # --- Metrics ---
    acc = accuracy_score(all_labels, all_preds)
    logger.info(f"Top-1 Accuracy: {acc * 100:.2f}%")
    
    target_names = class_names if class_names and len(class_names) == cfg['data']['num_classes'] else [str(i) for i in range(cfg['data']['num_classes'])]
    
    report_str = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    report_dict = classification_report(all_labels, all_preds, target_names=target_names, output_dict=True)
    logger.info(f"\nClassification Report:\n{report_str}")
    
    # --- Save Metrics to CSV ---
    csv_path = out_dir / f"evaluation_metrics_{args.split}.csv"
    with open(csv_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Hành vi', 'Số lượng mẫu', 'Precision', 'Recall', 'F1-Score'])
        
        for class_name in target_names:
            if class_name in report_dict:
                metrics = report_dict[class_name]
                writer.writerow([
                    class_name,
                    int(metrics['support']),
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}"
                ])
                
        writer.writerow([])
        for avg_name in ['macro avg', 'weighted avg']:
            if avg_name in report_dict:
                metrics = report_dict[avg_name]
                writer.writerow([
                    avg_name,
                    int(metrics['support']),
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['f1-score']:.4f}"
                ])
    logger.info(f"Detailed metrics saved to CSV: {csv_path}")
    
    # --- Confusion Matrix ---
    cm_path = out_dir / f"confusion_matrix_{args.split}.png"
        
    plot_confusion_matrix(all_labels, all_preds, class_names=target_names, out_path=cm_path, normalize=True)
    logger.info(f"Confusion matrix saved to: {cm_path}")
    
if __name__ == '__main__':
    main()
