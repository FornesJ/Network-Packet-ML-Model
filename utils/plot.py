import matplotlib.pyplot as plt
from config import Config
conf = Config()

def plot_precision_recall_f1(metrics, plot_path, n_epochs=conf.epochs):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    })
    epochs = range(1, n_epochs + 1)

    plots = [
        [
            ("precision_macro", "Macro Precision"),
            ("precision_weighted", "Weighted Precision"),
            ("precision_micro", "Micro Precision")
        ],
        [
            ("recall_macro", "Macro Recall"),
            ("recall_weighted", "Weighted Recall"),
            ("recall_micro", "Micro Recall")
        ],
        [
            ("f1_macro", "Macro F1"),
            ("f1_weighted", "Weighted F1"),
            ("f1_micro", "Micro F1")
        ]
    ]
    
    titles = ["Precision", "Recall", "F1 Score"]
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    for ax, plot, name in zip(axes, plots, titles):
        for (key, label) in plot:
            metric = [met[key] for met in metrics]
            ax.plot(epochs, metric, label=label)

        ax.set_title(name)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(name)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()



def plot_fpr_tpr_roc_auc(metrics, plot_path):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    })
    
    plt.plot(
        metrics["fpr_macro"],
        metrics["tpr_macro"],
        label=f"Macro-average ROC (AUC = {metrics['roc_auc_macro']:.3f})"
    )

    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1)

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(frameon=False)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.grid(alpha=0.3)
    plt.tight_layout()
    #plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()





def plot_loss(train_loss, val_loss, plot_path, n_epochs=conf.epochs):
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "figure.dpi": 300,
    })

    epochs = range(1, n_epochs + 1)
    plots = [
        (train_loss, "Train Loss"),
        (val_loss, "Validation Loss")
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    for ax, (loss, label) in zip(axes, plots):
        ax.plot(epochs, loss, label=label)

        ax.set_title(label)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.legend(frameon=False)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.show()
