import glob
import os
from copy import deepcopy

import imageio.v3 as iio
import matplotlib
import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip
from natsort import os_sorted
from pygifsicle import optimize
from ACE import fromEventToTensor


def readEventData(event_data_path):
    list_of_dicts = np.empty((0, 3))
    with open(event_data_path, "r") as timestamp_file:
        for line in timestamp_file.readlines()[1:]:
            values = line.replace("\n", "").split(sep="$")
            list_of_dicts = np.vstack((list_of_dicts, [float(value) for value in values]))
    return list_of_dicts


def readEventSerie(event_serie_path, h, w):
    events_data = readEventData(os.path.join(event_serie_path, "frame_events_data.txt"))
    event_files = sorted(glob.glob(os.path.join(event_serie_path, "*.npz")))
    AET = torch.zeros((0, h, w))
    for event_file in event_files:
        event = np.load(event_file)
        AET = torch.concat((AET, fromEventToTensor(event, h, w)[None, :]), dim=0)
    return AET, events_data


def render(shape, ET):
    img = np.full(shape=shape + [3], fill_value=255, dtype="uint8")
    mask = ET != 0
    p = np.int32(np.sign(ET.numpy()))
    img[mask, :] = 0
    intensity = np.vstack((np.full(shape=p[mask].shape, fill_value=255, dtype="uint8"), np.abs(ET.numpy()[mask]) * 80))
    img[mask, p[mask]] = np.min(intensity, axis=0)
    return img


def plotET(ET):
    ET = torch.transpose(ET, 0, 1)
    matplotlib.use('TkAgg')
    shape = [ET.shape[-2], ET.shape[-1]]
    handle = plt.imshow(render(shape, ET))
    plt.show()


def plotETSerie(ETs):
    matplotlib.use('TkAgg')
    ETs = torch.transpose(ETs, -1, -2)
    fig, ax = plt.subplots()
    shape = [ETs.shape[-2], ETs.shape[-1]]
    handle = plt.imshow(render(shape, ETs[0]))
    plt.show(block=False)
    plt.pause(0.2)

    for ET in ETs[1:]:
        img = render(shape, ET)
        handle.set_data(img)
        plt.pause(0.2)


def eventSerieToMP4(ETs, event_labels, save_path, video_id):
    os.makedirs(save_path, exist_ok=True)
    frames = []

    matplotlib.use('Agg')
    ETs = torch.transpose(ETs, -1, -2)
    fig, (ax_event, ax_labels) = plt.subplots(1, 2)
    shape = [ETs.shape[-2], ETs.shape[-1]]
    handle = ax_event.imshow(render(shape, ETs[0]))
    ax_labels.plot(np.arange(0, 1), event_labels[0, 0], "blue",  label="Valence")
    ax_labels.plot(np.arange(0, 1), event_labels[0, 1], "red",  label="Arousal")

    plt.legend()

    plt.title(f"Video {video_id} - Event 0")
    plt.tight_layout()
    fig.canvas.draw()
    data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    frames.append(data)

    for id, ET in enumerate(ETs[1:]):
        plt.title(f"Video {video_id} - Event {id}")
        img = render(shape, ET)
        handle.set_data(img)
        ax_labels.plot(np.arange(0, id + 1), event_labels[:id + 1, 0], "blue", label="Valence")
        ax_labels.plot(np.arange(0, id + 1), event_labels[:id + 1, 1], "red", label="Arousal")
        plt.tight_layout()
        fig.canvas.draw()
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        frames.append(data)

    ImageSequenceClip(frames, fps=len(frames) / 10).write_videofile(f"{save_path}/video.mp4")



def generateRGBMP4(path, save_path, to_expand=None):
    os.makedirs(save_path, exist_ok=True)
    frames = []

    if to_expand is None:
        to_expand = len(os.listdir(path))

    to_replicate = to_expand // len(os.listdir(path))

    for file_name in os_sorted(os.listdir(path)):
        i = 0
        image = Image.open(os.path.join(path, file_name))
        # image = iio.imread(os.path.join(path, file_name))
        while i < to_replicate:
            frames.append(np.array(image.convert('RGB')))
            i += 1

    while len(frames) < to_expand:
        frames.append(image)


    ImageSequenceClip(frames, fps=len(frames) / 10).write_videofile(f"{save_path}/video.mp4")

