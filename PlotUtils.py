import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.lines import Line2D


def plotGradFlow(named_parameters, verbose=1, legend=False):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plotGradFlow(self.model.named_parameters())" to visualize the gradient flow"""
    matplotlib.use("TkAgg")

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad is None:
            print(f"Parameter {n} is none!")
        else:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                max_grads.append(p.grad.abs().max().cpu().detach().numpy())
                if (ave_grads[-1] == 0 or max_grads[-1] == 0) and verbose > 0:
                    print(n)

    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    if legend:
        plt.legend([Line2D([0], [0], color="c", lw=4),
                    Line2D([0], [0], color="b", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.ion()
    plt.show()
    plt.pause(0.001)


def plot_grad_flow(named_parameters, verbose=1, legend=False):
    """Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow"""
    # plt.rcParams['figure.dpi'] = 300
    # plt.rcParams["figure.figsize"] = (plt.rcParamsDefault["figure.figsize"][0] * 2,  plt.rcParamsDefault["figure.figsize"][1])

    ave_grads = []
    max_grads = []
    layers = []
    for n, p in named_parameters:
        if p.grad is None:
            print(f"Parameter {n} is none!")
        else:
            if p.requires_grad and ("bias" not in n):
                layers.append(n)
                ave_grads.append(p.grad.abs().mean().cpu().detach().numpy())
                max_grads.append(p.grad.abs().max().cpu().detach().numpy())
                if ave_grads[-1] == 0 or max_grads[-1] == 0 and verbose > 0:
                    print(f"Paameter {n} is 0!")
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.4, lw=1, color="darkorange")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.5, lw=1, color="lime")
    plt.hlines(0, 0, len(ave_grads) + 1, lw=2, color="k")
    layers = [layers[i].replace(".weight", "") for i in range(len(layers))]
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation=90)
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=np.max(ave_grads) / 2)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    if legend:
        plt.legend([Line2D([0], [0], color="deepskyblue", lw=4),
                    Line2D([0], [0], color="steelblue", lw=4),
                    Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])
    plt.tight_layout()
    plt.show()
    plt.close()

    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.rcParams['figure.dpi'] = plt.rcParamsDefault['figure.dpi']


def plotTrainingStatistics(path, filename, train_losses, train_loss_names, title="Training Statistics"):
    colors = ["navy", "b", "cornflowerblue", "dodgerblue", "lightskyblue"]
    os.makedirs(f"Results/{path}", exist_ok=True)

    for i, loss in enumerate(train_losses):
        plt.plot(np.arange(train_losses.shape[1]), loss, colors[i], label=train_loss_names[i], linestyle="--" if i > 0 else "-")

    plt.title(f"{title}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig(f"Results/{path}/{filename}.png")
    plt.close()


def plotTestStatistics(path, filename, test_losses, test_loss_names, title="Test Statistics"):
    colors = [(0.56, 0., 1., 1),
              (0., 0.2, 0.6, 0.4),
              (0.81, 0.05, 0.05, 0.4),
              (0., 0.73, 0.18, 0.4),
              (1., 0.53, 0., 0.4)]

    os.makedirs(f"Results/{path}", exist_ok=True)

    for i, loss in enumerate(test_losses):
        plt.plot(np.arange(test_losses.shape[1]), loss, color=colors[i], label=test_loss_names[i])

    plt.title(f"{title} - {test_loss_names[0]}: {np.min(test_losses[0])}")
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.savefig(f"Results/{path}/{filename}.png")
    plt.close()


def plotLabelsStat(labels, labels_to_compare=None):
    if labels_to_compare is None:
        fig, axes = plt.subplots(1, 2, sharey="all")
    else:
        fig, axes = plt.subplots(2, 2, sharex="col", sharey="all")

    for i in range(labels.shape[-1]):
        # calcola le statistiche delle labels

        if labels_to_compare is not None:
            unique_labels, counts = np.unique(labels[:, -1, i], return_counts=True)
            label_freq = counts / len(labels)

            # crea un grafico a barre per le frequenze delle labels

            axes[0][i].bar(unique_labels, label_freq)
            axes[0][i].set_xticks(unique_labels)
            axes[0][i].set_xlabel(["Valence", "Arousal"][i])
            axes[0][i].set_ylabel('% Frequency')

            unique_compare_labels, compare_counts = np.unique(labels_to_compare[:, -1, i], return_counts=True)
            compare_label_freq = compare_counts / len(labels_to_compare)

            axes[1][i].bar(unique_compare_labels, compare_label_freq)
            axes[1][i].set_xticks(unique_compare_labels)
            axes[1][i].set_xlabel(["Valence", "Arousal"][i])
            axes[1][i].set_ylabel('% Frequency')
        else:
            unique_labels, counts = np.unique(labels[:, -1, i], return_counts=True)
            label_freq = counts / len(labels)

            # crea un grafico a barre per le frequenze delle labels

            axes[i].bar(unique_labels, label_freq)
            axes[i].set_xticks(unique_labels)
            axes[i].set_xlabel(["Valence", "Arousal"][i])
            axes[i].set_ylabel('% Frequency')

    fig.tight_layout()
    plt.show()


def plotPredVSGroundT(ground_truth, pred_values, title=""):
    if len(ground_truth) > 200:
        fig, axes = plt.subplots(2, 1, sharex="all", sharey="all", figsize=(20, 10))
    else:
        fig, axes = plt.subplots(2, 1, sharex="all", sharey="all")

    for i, ax in enumerate(axes):
        # indices = torch.argsort(ground_truth[:, i])
        # ax.plot(np.arange(len(ground_truth)), pred_values[:, i][indices].flatten(), color="red", label="pred_values")
        ax.plot(np.arange(len(ground_truth)), pred_values[:, i].flatten(), color="red", label="pred_values")
        ax.plot(np.arange(len(ground_truth)), ground_truth[:, i].flatten(), color="blue", label="ground_truth")
        # ax.plot(np.arange(len(ground_truth)), ground_truth[:, i][indices].flatten(), color="blue", label="ground_truth")
        ax.set_xlabel('Sample')
        ax.set_ylabel('Label')
        # ax.set_ylim(bottom=-1, top=1)
        ax.set_ylim(bottom=-10, top=10)
        # ax.set_title([f"{title} Valence", f"{title} Arousal"][i])
        ax.set_title([f"{str(round(pred_values[-1, 0].item(), 1))} : Valence", f"{str(round(pred_values[-1, 1].item(), 1))} Arousal"][i])

    fig.tight_layout()
    return fig




class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
