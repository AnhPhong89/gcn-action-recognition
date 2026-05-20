import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import confusion_matrix

def plot_confusion_matrix(y_true, y_pred, class_names, out_path, normalize=True):
    """
    Plots and saves a confusion matrix.
    
    Args:
        y_true (list or np.ndarray): Ground truth labels.
        y_pred (list or np.ndarray): Predicted labels.
        class_names (list): List of string names for classes.
        out_path (str or Path): File path to save the generated image.
        normalize (bool): If True, normalizes the matrix to show percentages.
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fmt = 'd'
    if normalize:
        # Normalize along the rows (true classes) to get recall/accuracy per class
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_norm = np.nan_to_num(cm_norm) # handle division by zero
        cm_display = cm_norm
        fmt = '.2f'
    else:
        cm_display = cm

    plt.figure(figsize=(10, 8))
    sns.set(font_scale=1.2) # Adjust font size
    
    ax = sns.heatmap(
        cm_display, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues', 
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        annot_kws={"size": 14}
    )
    
    # Add count below the percentage if normalized
    if normalize:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j+0.5, i+0.7, f"({cm[i, j]})", 
                        ha="center", va="center", color="black" if cm_display[i, j] < 0.5 else "white", fontsize=10)

    plt.ylabel('True Label', fontsize=14, fontweight='bold')
    plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
    plt.title('Confusion Matrix' + (' (Normalized)' if normalize else ''), fontsize=16, pad=20)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    out_dir = Path(out_path).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(str(out_path), dpi=300, bbox_inches='tight')
    plt.close()
    
    return out_path
