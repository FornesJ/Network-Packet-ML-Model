import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
from config import Config
conf = Config()

def plot_precision_recall_f1(metrics, plot_path, n_epochs=conf.epochs):
    plt.rcdefaults()
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
    plt.rcdefaults()

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
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()





def plot_loss(train_loss, val_loss, plot_path, n_epochs=conf.epochs):
    plt.rcdefaults()
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


def plot_confusion_matrix(cm, class_names, plot_path):
    plt.rcdefaults()
    # Global matplotlib settings (paper-friendly)
    plt.rcParams.update({
        "font.size": 9,
        "axes.labelsize": 9,
        "axes.titlesize": 10,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "font.family": "serif",
        "axes.grid": False,
    })

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=class_names,
    )

    disp.plot(include_values=False, cmap="Greys", values_format=None)
    
    ax = disp.ax_

    # Square cells
    ax.set_aspect("equal")

    # Cell-aligned grid
    n = disp.confusion_matrix.shape[0]
    ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Clean spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    #plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()


def plot_benchmark(dpu_csv, host_csv, plot_path):

    # -----------------------------
    # Global matplotlib styling
    # -----------------------------
    plt.rcdefaults()
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.dpi": 300,
        "axes.linewidth": 0.8,
    })

    # -----------------------------
    # Helper: parse CPU cores
    # -----------------------------
    def cores_to_percent(val):
        """
        Convert 'used/total' cores string to utilization percentage.
        Example: '5.55/8' -> 69.4
        """
        try:
            used, total = val.split("/")
            return float(used) / float(total) * 100
        except Exception:
            return None

    # -----------------------------
    # Load raw CSVs (keep originals!)
    # -----------------------------
    dpu_raw = pd.read_csv(dpu_csv)
    host_raw = pd.read_csv(host_csv)

    # Remove Info section
    dpu_raw = dpu_raw[dpu_raw["Section"] != "Info"]
    host_raw = host_raw[host_raw["Section"] != "Info"]

    # -----------------------------
    # Convert numeric metrics
    # -----------------------------
    dpu = dpu_raw.copy()
    host = host_raw.copy()

    dpu["Value"] = pd.to_numeric(dpu["Value"], errors="coerce")
    host["Value"] = pd.to_numeric(host["Value"], errors="coerce")

    # -----------------------------
    # Handle CPU Avg. (cores)
    # -----------------------------
    for df_raw, df in [(dpu_raw, dpu), (host_raw, host)]:
        cpu_mask = (df_raw["Section"] == "CPU") & (df_raw["Metric"] == "Avg. (cores)")
        cpu_vals = df_raw.loc[cpu_mask, "Value"].apply(cores_to_percent)

        df.loc[cpu_mask, "Value"] = cpu_vals
        df.loc[cpu_mask, "Metric"] = "Avg. CPU Utilization (%)"

    # Drop remaining non-numeric values
    dpu = dpu.dropna(subset=["Value"])
    host = host.dropna(subset=["Value"])

    # -----------------------------
    # Drop unwanted metrics
    # -----------------------------
    drop_rules = {
        "Latency": ["Total (ms)"],
        "Throughput": ["Runtime (s)"],
        "CPU": ["Runtime (s)"],
    }

    def drop_metrics(df):
        for section, metrics in drop_rules.items():
            df = df[~((df["Section"] == section) & (df["Metric"].isin(metrics)))]
        return df

    dpu = drop_metrics(dpu)
    host = drop_metrics(host)

    # -----------------------------
    # Merge Host & DPU
    # -----------------------------
    df = pd.merge(
        dpu, host,
        on=["Section", "Metric"],
        suffixes=("_DPU", "_Host")
    )

    sections = df["Section"].unique()
    n_sections = len(sections)

    # -----------------------------
    # Create subplots
    # -----------------------------
    fig, axes = plt.subplots(
        n_sections, 1,
        figsize=(7.5, 2.8 * n_sections),
        sharex=False
    )

    if n_sections == 1:
        axes = [axes]

    def get_ylabel(section_name):
        if section_name == "Memory":
            return "MB"
        elif section_name == "Latency":
            return "ms"
        elif section_name == "Throughput":
            return "#/s"
        elif section_name == "CPU":
            return "%"
        else:
            return "Value"

    for ax, section in zip(axes, sections):
        sdf = df[df["Section"] == section]

        x = np.arange(len(sdf))
        width = 0.36

        host_vals = sdf["Value_Host"]
        dpu_vals = sdf["Value_DPU"]

        ax.bar(
            x - width/2, host_vals, width,
            label="Host",
            color="0.75",
            edgecolor="black",
            linewidth=0.6
        )
        ax.bar(
            x + width/2, dpu_vals, width,
            label="DPU",
            color="0.35",
            edgecolor="black",
            linewidth=0.6,
            hatch="//"
        )

        # Gain / loss annotation
        gain = (dpu_vals - host_vals) / host_vals * 100
        for i, g in enumerate(gain):
            ax.text(
                x[i],
                max(host_vals.iloc[i], dpu_vals.iloc[i]) * 1.03,
                f"{g:+.1f}%",
                ha="center",
                va="bottom",
                fontsize=9
            )

        ax.set_title(section, pad=6)
        ax.set_ylabel(get_ylabel(section))
        ax.set_xticks(x)
        ax.set_xticklabels(sdf["Metric"], rotation=30, ha="right")

        ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.6)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.02)
    )

    plt.tight_layout(rect=[0, 0, 1, 0.98])

    plt.savefig(plot_path, bbox_inches="tight")
    plt.show()
