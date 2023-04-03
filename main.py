from ACE import getAETChunk, eventVoxelizationAllPast
from BET import EncoderCNN, VideoBranch
from EventDataset import EventDataset
from EventsUtils import readEventSerie, plotET, plotETSerie
from EventsCompressor import EventsCompressor

import pickle as pkl
import os

def main():

    # with open(f"EventDatasets/validation_dataset100BIN8STEP_ALLACCU.pkl", "rb") as dataset_file:
    #     train_dataset = pkl.load(dataset_file)

    dataset = EventDataset()
    dataset.generateFromRawEvents("../AFEW_VA_Dataset/267_dataset", 100, 200, 200, accumulation_mode="bin")
    dataset.save(name="267_dataset")

    # compressor = EventsCompressor()
    #
    # dataset = EventDataset()
    # dataset.generateFromRawEvents("../AFEW_VA_Dataset/test_dataset", 100, 200, 200, accumulation_mode="none")
    # dataset.save(name="test_dataset100BIN8STEP_NONEACCU")
    # #
    # dataset = EventDataset()
    # dataset.generateFromRawEvents("../AFEW_VA_Dataset/validation_dataset", 100, 200, 200, accumulation_mode="none")
    # dataset.save(name="validation_dataset100BIN8STEP_NONEACCU")
    # #
    # dataset = EventDataset()
    # dataset.generateFromRawEvents("../AFEW_VA_Dataset/train_dataset", 100, 200, 200, accumulation_mode="none")
    # dataset.save(name="train_dataset100BIN8STEP_NONEACCU")
    #
    #


    #
    #

    # compressed_events = compressor(events)
    #
    # encoder = EncoderCNN()
    # encoded = encoder(compressed_events)
    #
    # video_branch = VideoBranch(encoder.output_dim)
    # valence = video_branch(encoded)
    # print(compressed_events)


if __name__ == "__main__":
    main()
