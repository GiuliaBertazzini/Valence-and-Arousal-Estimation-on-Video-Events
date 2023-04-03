import itertools
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from natsort import os_sorted
from tqdm import tqdm
import pickle as pkl
from ACE import getAETChunk, eventVoxelizationAllPast, eventVoxelizationVoxelPast, eventVoxelizationNoAccumulation
from EventsUtils import readEventSerie

from sklearn.preprocessing import MinMaxScaler


class EventDataset(torch.utils.data.Dataset):
    def __init__(self):

        self.events = None
        self.labels = None

        self.compressed = False
        self.compression_ratio = None

        self.video_id = []

        self.labels_normalizer = None

    def __getitem__(self, idx):
        return self.events[idx], self.labels[idx]

    def __len__(self):
        return len(self.labels)

    def generateFromRawEvents(self, events_path, events_per_chunk, event_h, event_w, accumulation_mode="all"):
        self.events = torch.empty((0, events_per_chunk, event_h, event_w))
        self.labels = torch.empty((0, events_per_chunk, 2))
        for video_id in tqdm(os_sorted(os.listdir(events_path)), position=0, leave=False, desc=f"Generating from video", colour="white", ncols=120):
            AET, AET_data = readEventSerie(os.path.join(events_path, video_id), event_h, event_w)

            if len(AET) != len(AET_data):
                tqdm.write(f"Video {video_id} inconsistency!")
                continue

            AETs, AETs_data = getAETChunk(AET, AET_data)
            for (chunk, chunk_data) in zip(AETs, AETs_data):
                self.video_id.append(video_id)
                if accumulation_mode == "all":
                    VCE, VCE_data = eventVoxelizationAllPast(chunk, chunk_data)
                elif accumulation_mode == "bin":
                    VCE, VCE_data = eventVoxelizationVoxelPast(chunk, chunk_data)
                elif accumulation_mode == "none":
                    VCE, VCE_data = eventVoxelizationNoAccumulation(chunk, chunk_data)
                else:
                    raise()
                self.events = torch.vstack((self.events, VCE[None, :]))
                self.labels = torch.vstack((self.labels, VCE_data[None, :, 1:3]))

    def compressLabels(self, compress_ratio):
        self.compressed = True
        self.compression_ratio = compress_ratio

        labels = torch.empty((0, self.labels.shape[1] // self.compression_ratio, self.labels.shape[-1]))
        for sample_labels in self.labels:
            new_sample_labels = torch.empty((0, self.labels.shape[-1]))
            for compression_chunk in range(0, len(sample_labels) - 1, self.compression_ratio):
                chunk_label = torch.mean(sample_labels[compression_chunk: compression_chunk + self.compression_ratio], dim=0)
                new_sample_labels = torch.vstack((new_sample_labels, chunk_label))

            labels = torch.vstack((labels, new_sample_labels[None, :]))

        self.labels = labels

    def normalizeLabels(self, labels=None):
        frame_dimension = self.labels.shape[-2]
        feature_dimension = self.labels.shape[-1]
        device = self.labels.device

        if labels is None:
            self.labels_normalizer = MinMaxScaler((-1, 1), clip=True)
            self.labels = self.labels_normalizer.fit_transform(self.labels.view(-1, feature_dimension))
            self.labels = torch.tensor(self.labels, device=device, dtype=torch.float32).reshape((-1, frame_dimension, feature_dimension))
            return None
        else:
            if self.labels_normalizer is None:
                raise
            labels = self.labels_normalizer.transform(labels.view(-1, feature_dimension))
            labels = torch.tensor(labels, device=device, dtype=torch.float32).reshape((-1, frame_dimension, feature_dimension))
            return labels

    def save(self, name):
        os.makedirs("EventDatasets", exist_ok=True)
        with open(f"EventDatasets/{name}.pkl", "wb") as dataset_file:
            pkl.dump(self, dataset_file)

    def standardizeEventsOur(self, mean=None, std=None):
        if mean is None or std is None:
            mean = torch.mean(self.events, dim=(0, 1))
            std = torch.std(self.events, dim=(0, 1))

        mask = std != 0

        self.events = (self.events - mean)
        self.events[:, :, mask] = self.events[:, :, mask] / std[mask]

        return mean, std

    def standardizeEventsStandard(self, mean=None, std=None):
        if mean is None or std is None:
            mean = torch.mean(self.events)
            std = torch.std(self.events)

        self.events = (self.events - mean) / std
        return mean, std

    def makeUniform(self):
        unique_valences, counts = torch.unique(self.labels[:, -1, 0], return_counts=True)
        valences_freq = counts / len(self.labels)
        valences_freq_mean = torch.mean(valences_freq)
        valences_classes_over_mean = torch.argwhere(valences_freq > valences_freq_mean)

        unique_arousals, counts = torch.unique(self.labels[:, -1, 1], return_counts=True)
        arousals_freq = counts / len(self.labels)
        arousals_freq_mean = torch.mean(arousals_freq)
        arousals_classes_over_mean = torch.argwhere(arousals_freq > arousals_freq_mean)

        classes_combinations = itertools.product(valences_classes_over_mean, arousals_classes_over_mean)
        for label in self.labels:
            a = 1
