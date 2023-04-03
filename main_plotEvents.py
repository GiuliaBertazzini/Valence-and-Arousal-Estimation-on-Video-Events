import numpy as np

from ACE import getAETChunk, eventVoxelizationAllPast
from BET import EncoderCNN, VideoBranch
from EventDataset import EventDataset
from EventsUtils import readEventSerie, plotET, plotETSerie, generateRGBMP4, eventSerieToMP4
from EventsCompressor import EventsCompressor

import pickle as pkl
import os

import imageio


def main():

    # generateRGBMP4("Resources/287/original_scene", "VIDEOs/287/original_scene")
    generateRGBMP4("Resources/267/original_scene", "VIDEOs/267/original_scene")
    # generateRGBMP4("Resources/287/cropped_scene", "VIDEOs/287/cropped_scene")

    # with open("EventDatasets/test_dataset100BIN8STEP_BINACCU.pkl", "rb") as dataset_file:
    #     dataset = pkl.load(dataset_file)
    #
    # event_id = np.argwhere(np.array([int(video_id) for video_id in dataset.video_id]) == 287)[0][0]
    #
    # event_serie, event_labels = dataset[event_id]
    # video_id = dataset.video_id[event_id]
    #
    # eventSerieToMP4(event_serie, event_labels, "VIDEOs/287/BinAcc", video_id)
    #
    # with open("EventDatasets/test_dataset100BIN8STEP_ALLACCU.pkl", "rb") as dataset_file:
    #     dataset = pkl.load(dataset_file)
    #
    # event_id = np.argwhere(np.array([int(video_id) for video_id in dataset.video_id]) == 287)[0][0]
    #
    # event_serie, event_labels = dataset[event_id]
    # video_id = dataset.video_id[event_id]
    #
    # eventSerieToMP4(event_serie, event_labels, "VIDEOs/287/AllAcc", video_id)
    #
    # with open("EventDatasets/test_dataset100BIN8STEP_NONEACCU.pkl", "rb") as dataset_file:
    #     dataset = pkl.load(dataset_file)
    #
    # event_id = np.argwhere(np.array([int(video_id) for video_id in dataset.video_id]) == 287)[0][0]
    #
    # event_serie, event_labels = dataset[event_id]
    # video_id = dataset.video_id[event_id]
    #
    # eventSerieToMP4(event_serie, event_labels, "VIDEOs/287/NoneAcc", video_id)



if __name__ == "__main__":
    main()
