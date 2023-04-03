import gc
import os
import pickle as pkl
import shutil
import sys
from datetime import datetime
from time import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from BET import BET
from Losses import VideoBETLoss
from PlotUtils import plotTrainingStatistics, plotTestStatistics, bcolors, plotLabelsStat, plotPredVSGroundT
from Utils import AverageMeter, ProgressMeter, countParametersMB


def main():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    datasets = ["dataset100BIN8STEP_BINACCU", "dataset100BIN8STEP_NONEACCU"]

    for dataset_name in datasets:
        model = BET(100)
        model.to(device)
        countParametersMB(model)

        gc.disable()
        with open(f"EventDatasets/train_{dataset_name}.pkl", "rb") as dataset_file:
            train_dataset = pkl.load(dataset_file)
        gc.enable()

        gc.disable()
        with open(f"EventDatasets/validation_{dataset_name}.pkl", "rb") as dataset_file:
            validation_dataset = pkl.load(dataset_file)
        gc.enable()

        plotLabelsStat(train_dataset.labels.cpu().numpy(), validation_dataset.labels.cpu().numpy())

        # train_dataset.compressLabels(model.compressor.compression_ratio)
        # validation_dataset.compressLabels(model.compressor.compression_ratio)

        train_dataset.normalizeLabels()
        validation_dataset.labels = train_dataset.normalizeLabels(validation_dataset.labels)

        mean, std = train_dataset.standardizeEventsStandard()
        validation_dataset.standardizeEventsStandard(mean, std)

        if "BINACCU" in dataset_name:
            criterion_list = [
                # VideoBETLoss("wmse", labels=train_dataset.labels[:, -1, :], alpha=0.75, beta=0.25, gamma=0),
                # VideoBETLoss("wmse", labels=train_dataset.labels[:, -1, :], alpha=0.50, beta=0.50, gamma=0),
                VideoBETLoss("wmse", labels=train_dataset.labels[:, -1, :], alpha=0.75, beta=0, gamma=0.25),
                VideoBETLoss("wmse", labels=train_dataset.labels[:, -1, :], alpha=0.50, beta=0, gamma=0.50),
                VideoBETLoss("wmse", labels=train_dataset.labels[:, -1, :], alpha=0.50, beta=0.25, gamma=0.25)
            ]
        elif "NONEACCU" in dataset_name:
            criterion_list = [
                # VideoBETLoss("rmse", labels=train_dataset.labels[:, -1, :], alpha=0.75, beta=0.25, gamma=0),
                # VideoBETLoss("rmse", labels=train_dataset.labels[:, -1, :], alpha=0.50, beta=0.50, gamma=0),
                VideoBETLoss("rmse", labels=train_dataset.labels[:, -1, :], alpha=0.75, beta=0, gamma=0.25),
                VideoBETLoss("rmse", labels=train_dataset.labels[:, -1, :], alpha=0.50, beta=0, gamma=0.50),
                VideoBETLoss("rmse", labels=train_dataset.labels[:, -1, :], alpha=0.50, beta=0.25, gamma=0.25)
            ]
        else:
            raise

        for CRITERION in criterion_list:
            EXPERIMENT_NAME = f"{dataset_name}_CommonBranch_{CRITERION.type.upper()}_BETA{CRITERION.beta}_GAMMA{CRITERION.gamma}"

            trainNetwork(model, train_dataset, validation_dataset, CRITERION, device, experiment_name=EXPERIMENT_NAME)


def trainNetwork(model, train_dataset, test_dataset, CRITERION, device, epochs=100, experiment_name=None):
    if experiment_name is None:
        EXPERIMENT_NAME = "Exp" + datetime.now().strftime('%d%m%Y-%H%M')
    else:
        EXPERIMENT_NAME = experiment_name

    SAVE_PATH = f"Results/{EXPERIMENT_NAME}"

    if os.path.exists(SAVE_PATH):
        shutil.rmtree(SAVE_PATH)
    summary_writer = SummaryWriter(log_dir=f"{SAVE_PATH}/tensorboard")

    # Added to not use 100% of CPU  (The copy of a tensor is made in parallel in pytorch)
    torch.set_num_threads(4)

    BATCH_SIZE = 32

    optimizer = torch.optim.AdamW(model.parameters())

    start = time()
    train_losses = np.empty((CRITERION.num_losses, 0))
    test_losses = np.empty((CRITERION.num_losses, 0))

    for epoch in range(epochs):
        model.train()
        dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

        batch_time = AverageMeter('Time', ':6.3f')
        loss_meters = [AverageMeter(name, ':.6f') for name in CRITERION.names]

        progress = ProgressMeter(
            len(dataloader),
            [batch_time] + loss_meters,
            prefix='Training: ')

        end = time()

        pred_values_to_plot = torch.empty((0, train_dataset.labels.shape[-1]))
        ground_truth = torch.empty((0, train_dataset.labels.shape[-1]))
        embeddings_to_plot = torch.empty((0, 20))

        for sampler, (batched_video_chunks, labels) in enumerate(dataloader):
            # To GPU memory
            batched_video_chunks = batched_video_chunks.to(device)
            video_labels = labels[:, -1, :].to(device)

            if sampler == 0 and epoch == 0:
                summary_writer.add_graph(model, batched_video_chunks)

            # Optimization step
            optimizer.zero_grad()

            pred_values, embeddings = model(batched_video_chunks)

            losses = CRITERION(pred_values, video_labels, embeddings)

            for id, loss in enumerate(losses):
                loss_meters[id].update(loss.item(), BATCH_SIZE)

            losses[0].backward()
            optimizer.step()

            # measure elapsed time
            batch_time.update(time() - end)
            end = time()

            ground_truth = torch.vstack((ground_truth, video_labels.cpu()))
            pred_values_to_plot = torch.vstack((pred_values_to_plot, pred_values.cpu()))
            embeddings_to_plot = torch.vstack((embeddings_to_plot, embeddings.cpu()))

            if sampler % 10 == 0:
                tqdm.write(progress.get_string(sampler))

        # Plot training statistics to tensorboard
        statistics_dict = {}
        for i, train_meter in enumerate(loss_meters):
            statistics_dict[train_meter.name] = train_meter.avg
            summary_writer.add_scalar(f"Train/0{i}_{train_meter.name}", train_meter.avg, epoch)

        summary_writer.add_scalars("Train/SummaryLoss", statistics_dict, epoch)

        # plot_grad_flow(model.named_parameters())

        train_losses = np.hstack((train_losses, np.array([loss_meter.avg for loss_meter in loss_meters], ndmin=2).reshape((-1, 1))))

        to_write = bcolors.BOLD + f"Epoch {epoch + 1}/{epochs} completed in ({int(time() - start)}s) -->   "

        for loss_meter in loss_meters:
            to_write += f"{loss_meter.name}: {loss_meter.avg:.3f}   "
        tqdm.write(to_write + bcolors.ENDC)

        test_meters, test_pred_values = testNetwork(model=model, dataset=test_dataset, batch_size=BATCH_SIZE, device=device)

        to_write = bcolors.OKGREEN + f"Validation Test -->"
        statistics_dict = {}
        for i, test_meter in enumerate(test_meters):
            to_write += f"    {test_meter.name}: {test_meter.avg:.3f}"
            statistics_dict[test_meter.name] = test_meter.avg
            summary_writer.add_scalar(f"Val/0{i}_{test_meter.name}", test_meter.avg, epoch)
        summary_writer.add_scalars("Val/SummaryLoss", statistics_dict, epoch)
        to_write += bcolors.ENDC + "\n"

        if epoch > 0 and test_meters[0].avg < np.min(test_losses[0]):
            with torch.no_grad():
                train_fig = plotPredVSGroundT(ground_truth, pred_values_to_plot.numpy(), title="Training")
                val_fig = plotPredVSGroundT(test_dataset.labels[:, -1, :], test_pred_values.numpy(), title="Validation")
                summary_writer.add_figure("Train/PredVSGroundTruth", train_fig)
                summary_writer.add_figure("Val/PredVSGroundTruth", val_fig)
                val_fig.savefig(f"Results/{EXPERIMENT_NAME}/ValPredVSGroundTruth.png")
                train_fig.savefig(f"Results/{EXPERIMENT_NAME}/TrainPredVSGroundTruth.png")
            model.save(f"{SAVE_PATH}")
        tqdm.write(to_write)

        test_losses = np.hstack((test_losses, np.array([test_meter.avg for test_meter in test_meters], ndmin=2).reshape((-1, 1))))
        for i, loss in enumerate(test_losses):
            to_write += f"\n Min {test_meters[i].name}: {np.min(loss)}"

        with open(f'{SAVE_PATH}/bestNetStatistics.txt', 'w') as f:
            f.write(f'Epoch:{epoch}\n' + to_write)

        plotTrainingStatistics(EXPERIMENT_NAME, "trainingStatistics", train_losses, [loss_meter.name for loss_meter in loss_meters])
        plotTestStatistics(EXPERIMENT_NAME, "validationStatistics", test_losses, [test_meter.name for test_meter in test_meters])
    summary_writer.close()


def testNetwork(model, dataset, batch_size, device):
    model.eval()
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    criterion = VideoBETLoss(type="mae", alpha=1, beta=0, gamma=0)
    loss_meters = [AverageMeter(name, ':.6f') for name in criterion.names]

    pred_values_to_plot = torch.empty((0, dataset.labels.shape[-1]))
    with torch.no_grad():
        for batched_video_chunks, batched_labels in dataloader:

            # To GPU memory
            batched_video_chunks = batched_video_chunks.to(device)
            batched_labels = batched_labels.to(device)

            pred_values, embeddings = model(batched_video_chunks)
            pred_values_to_plot = torch.vstack((pred_values_to_plot, pred_values.cpu()))

            video_labels = batched_labels[:, -1, :]  # Prendo la label dell'ultimo bin (istante di tempo nel chunk)
            losses = criterion(pred_values, video_labels, embeddings)

            for id, loss in enumerate(losses):
                loss_meters[id].update(loss.item(), batch_size)

    model.train()
    return loss_meters, pred_values_to_plot


def qualitativeResults():
    os.chdir(os.path.dirname(sys.argv[0]))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = BET(100)
    model.load("Results/dataset100BIN8STEP_BINACCU_CommonBranch_RMSE_BETA0_GAMMA0/network.pt")
    model.to(device)
    countParametersMB(model)

    gc.disable()
    with open(f"EventDatasets/train_dataset100BIN8STEP_BINACCU.pkl", "rb") as dataset_file:
        train_dataset = pkl.load(dataset_file)
    gc.enable()

    gc.disable()
    with open(f"EventDatasets/149_dataset.pkl", "rb") as dataset_file:
        validation_dataset = pkl.load(dataset_file)
    gc.enable()

    train_dataset.normalizeLabels()
    validation_dataset.labels = train_dataset.normalizeLabels(validation_dataset.labels)
    mean, std = train_dataset.standardizeEventsStandard()
    validation_dataset.standardizeEventsStandard(mean, std)

    _, pred_values = testNetwork(model, validation_dataset, 32, device)

    frames = []

    for i, pred_value in enumerate(pred_values):
        video_labels = validation_dataset.labels[:, -1, :]
        fig = plotPredVSGroundT(video_labels[:i+1]*10, pred_values[:i+1]*10)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(data)

    ImageSequenceClip(frames, fps=len(frames) / 10).write_videofile(f"VIDEOs/149/RMSEqualitativeResult.mp4")

    frames = []

    for i, pred_value in enumerate(pred_values):
        video_labels = validation_dataset.labels[:, -1, :]
        fig = plotEmotion(pred_value[0] * 10, pred_value[1] * 10)
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(data)

    ImageSequenceClip(frames, fps=len(frames) / 10).write_videofile(f"VIDEOs/149/emotions.mp4")


def plotEmotion(valence, arousal):
    fig, ax = plt.subplots(1, figsize=(10, 10))

    ax.scatter(valence, arousal, c="red", s=300, alpha=1, edgecolors="black", linewidths=2, zorder=100)
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')

    ticks = np.arange(-10, 11)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)

    ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False)
    ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False)

    ax.set_xlabel('VALENCE', loc="right", fontsize='x-large', fontweight=700)
    ax.set_ylabel('AROUSAL', loc="top", fontsize='x-large', fontweight=700)

    ax.text(-5, 5, "ANGRY", color="red", fontsize='large', fontweight=700)
    ax.text(5, 5, "HAPPY", color="orangered", fontsize='large', fontweight=700)
    ax.text(-5, -5, "SAD", color="blue", fontsize='large', fontweight=700)
    ax.text(5, -5, "RELAXED", color="green", fontsize='large', fontweight=700)

    ax.set(xlim=(-11, 11), ylim=(-11, 11))

    circle = plt.Circle((0, 0), 10, color='lightskyblue', alpha=0.5, zorder=-1)
    ax.add_patch(circle)

    return fig


if __name__ == "__main__":
    qualitativeResults()
