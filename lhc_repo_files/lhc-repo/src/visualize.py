import numpy as np
import matplotlib.pyplot as plt

BG = "#0D1117"
COLORS = {
    "Gluon"    : "#4E8EFF",
    "Quark"    : "#2ECC71",
    "W boson"  : "#FF6B35",
    "Z boson"  : "#F1C40F",
    "Top quark": "#E74C3C",
}


def _style(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors="#8B949E")
    ax.grid(alpha=0.25, color="#30363D")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363D")


def plot_training_curve(history, save_path=None):
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor(BG)
    _style(ax)
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], color="#4E8EFF", lw=2, label="train")
    ax.plot(epochs, history["val"],   color="#FF6B35", lw=2, label="val")
    ax.set_xlabel("epoch",        color="#8B949E")
    ax.set_ylabel("MSE loss",     color="#8B949E")
    ax.set_title("autoencoder training curve", color="white")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()


def plot_anomaly_scores(err_bkg, err_sig, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(BG)
    clip = np.percentile(np.concatenate([err_bkg, err_sig]), 99)

    for ax, yscale in zip(axes, ["linear", "log"]):
        _style(ax)
        ax.hist(err_bkg, bins=80, density=True, alpha=0.65,
                color="#4E8EFF", label=f"QCD background  u={err_bkg.mean():.3f}")
        ax.hist(err_sig, bins=80, density=True, alpha=0.65,
                color="#FF6B35", label=f"W/Z/top signal  u={err_sig.mean():.3f}")
        ax.set_xlim(0, clip)
        ax.set_yscale(yscale)
        ax.set_xlabel("reconstruction error (MSE)", color="#8B949E")
        ax.set_ylabel("density", color="#8B949E")
        ax.legend(fontsize=9)

    plt.suptitle("anomaly score distribution", color="white", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()


def plot_roc(fpr, tpr, auc, save_path=None):
    fig, ax = plt.subplots(figsize=(7, 6))
    fig.patch.set_facecolor(BG)
    _style(ax)
    ax.plot(fpr, tpr, color="#9B59B6", lw=2.5, label=f"autoencoder  AUC={auc:.4f}")
    ax.plot([0,1],[0,1], "--", color="#3D444D", lw=1, label="random")
    ax.set_xlabel("false positive rate", color="#8B949E")
    ax.set_ylabel("true positive rate",  color="#8B949E")
    ax.set_title("ROC curve", color="white")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()


def plot_per_type(errors_by_type, save_path=None):
    fig, ax = plt.subplots(figsize=(12, 5))
    fig.patch.set_facecolor(BG)
    _style(ax)
    for name, err in errors_by_type.items():
        clip = np.percentile(err, 98)
        ax.hist(err, bins=60, density=True, alpha=0.6,
                color=COLORS.get(name, "white"),
                label=f"{name}  u={err.mean():.3f}",
                histtype="stepfilled", range=(0, clip))
    ax.set_xlabel("reconstruction error (MSE)", color="#8B949E")
    ax.set_ylabel("density", color="#8B949E")
    ax.set_title("anomaly score by jet type", color="white")
    ax.legend(fontsize=9)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()


def plot_umap(Z_2d, labels, type_map, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.patch.set_facecolor(BG)

    # all types
    ax = axes[0]
    ax.set_facecolor(BG)
    _style(ax)
    for jet_id, name in type_map.items():
        mask = labels == jet_id
        ax.scatter(Z_2d[mask, 0], Z_2d[mask, 1],
                   c=COLORS.get(name, "white"), label=name,
                   alpha=0.4, s=8, linewidths=0)
    ax.set_title("latent space -- all jet types", color="white")
    ax.set_xlabel("UMAP 1", color="#8B949E")
    ax.set_ylabel("UMAP 2", color="#8B949E")
    ax.legend(fontsize=9, markerscale=3)

    # background vs signal
    ax = axes[1]
    ax.set_facecolor(BG)
    _style(ax)
    bkg_ids = {0, 1}
    is_bkg  = np.array([l in bkg_ids for l in labels])
    ax.scatter(Z_2d[is_bkg,  0], Z_2d[is_bkg,  1],
               c="#4E8EFF", label="QCD background", alpha=0.3, s=8, linewidths=0)
    ax.scatter(Z_2d[~is_bkg, 0], Z_2d[~is_bkg, 1],
               c="#FF6B35", label="signal (W+Z+top)", alpha=0.4, s=8, linewidths=0)
    ax.set_title("latent space -- background vs signal", color="white")
    ax.set_xlabel("UMAP 1", color="#8B949E")
    ax.set_ylabel("UMAP 2", color="#8B949E")
    ax.legend(fontsize=9, markerscale=3)

    plt.suptitle("autoencoder latent space -- trained on QCD only",
                 color="white", fontsize=13)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.show()
