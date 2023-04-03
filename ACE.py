import numpy as np
import torch


def fromEventToTensor(event, h, w):
    event_tensor = torch.zeros((h, w))
    event_tensor[event["x"], event["y"]] = torch.from_numpy(np.float32(event["p"]))
    return event_tensor


def eventVoxelizationAllPast(AET: torch.tensor, events_data: np.ndarray):
    """

    :param AET: Aligned Event Tensor ( MxHxW where M is the number of timestamps in one recording)
    :param timestamps: event timestamps
    :return: VCE (Mq x H x W where q (QUAN_BIN) is the number of bin)
    """
    QUAN_BIN = 100

    quantization = {i: [] for i in range(QUAN_BIN)}
    timestamps = events_data[:, 0]
    min_t = min(timestamps)
    max_t = max(timestamps)
    bin_time_interval = (max_t - min_t) / (QUAN_BIN * 2)
    VET_timestamps = torch.tensor([(min_t + bin_time_interval) * i for i in range(1, QUAN_BIN + 1)])
    for i, timestamp in enumerate(timestamps[:-1]):
        bin_idx = np.floor((QUAN_BIN * (timestamp - min_t)) / (max_t - min_t))
        quantization[bin_idx].append(i)

    quantization[QUAN_BIN - 1].append(len(timestamps) - 1)

    for bin, events_indices in reversed(list(quantization.items())[:-1]):
        if not events_indices:
            quantization[bin].append(quantization[bin + 1][0])

    QET = torch.empty((QUAN_BIN, AET.shape[1], AET.shape[2]))
    valences = torch.empty(QUAN_BIN)
    arousals = torch.empty(QUAN_BIN)

    for bin, indices in quantization.items():
        QET[bin] = AET[indices].sum(dim=0)
        valences[bin] = np.round(np.mean(events_data[indices, 1]))
        arousals[bin] = np.round(np.mean(events_data[indices, 2]))

    VET = torch.cumsum(QET, dim=0)
    VET_events_data = torch.transpose(torch.vstack((VET_timestamps, valences, arousals)), 0, 1)
    return VET, VET_events_data


def eventVoxelizationVoxelPast(AET: torch.tensor, events_data: np.ndarray):
    """

    :param AET: Aligned Event Tensor ( MxHxW where M is the number of timestamps in one recording)
    :param timestamps: event timestamps
    :return: VCE (Mq x H x W where q (QUAN_BIN) is the number of bin)
    """
    QUAN_BIN = 100

    quantization = {i: [] for i in range(QUAN_BIN)}
    timestamps = events_data[:, 0]
    min_t = min(timestamps)
    max_t = max(timestamps)
    bin_time_interval = (max_t - min_t) / (QUAN_BIN * 2)
    VET_timestamps = torch.tensor([(min_t + bin_time_interval) * i for i in range(1, QUAN_BIN + 1)])
    for i, timestamp in enumerate(timestamps[:-1]):
        bin_idx = np.floor((QUAN_BIN * (timestamp - min_t)) / (max_t - min_t))
        quantization[bin_idx].append(i)

    quantization[QUAN_BIN - 1].append(len(timestamps) - 1)

    for bin, events_indices in reversed(list(quantization.items())[:-1]):
        if not events_indices:
            quantization[bin].append(quantization[bin + 1][0])

    QET = torch.empty((QUAN_BIN, AET.shape[1], AET.shape[2]))
    valences = torch.empty(QUAN_BIN)
    arousals = torch.empty(QUAN_BIN)

    for bin, indices in quantization.items():
        QET[bin] = AET[indices].sum(dim=0)
        valences[bin] = np.round(np.mean(events_data[indices, 1]))
        arousals[bin] = np.round(np.mean(events_data[indices, 2]))

    VET = QET
    VET_events_data = torch.transpose(torch.vstack((VET_timestamps, valences, arousals)), 0, 1)
    return VET, VET_events_data

def eventVoxelizationNoAccumulation(AET: torch.tensor, events_data: np.ndarray):
    """

    :param AET: Aligned Event Tensor ( MxHxW where M is the number of timestamps in one recording)
    :param timestamps: event timestamps
    :return: VCE (Mq x H x W where q (QUAN_BIN) is the number of bin)
    """
    QUAN_BIN = 100

    quantization = {i: [] for i in range(QUAN_BIN)}
    timestamps = events_data[:, 0]
    min_t = min(timestamps)
    max_t = max(timestamps)
    bin_time_interval = (max_t - min_t) / (QUAN_BIN * 2)
    VET_timestamps = torch.tensor([(min_t + bin_time_interval) * i for i in range(1, QUAN_BIN + 1)])
    for i, timestamp in enumerate(timestamps[:-1]):
        bin_idx = np.floor((QUAN_BIN * (timestamp - min_t)) / (max_t - min_t))
        quantization[bin_idx].append(i)

    quantization[QUAN_BIN - 1].append(len(timestamps) - 1)

    for bin, events_indices in reversed(list(quantization.items())[:-1]):
        if not events_indices:
            quantization[bin].append(quantization[bin + 1][0])

    QET = torch.empty((QUAN_BIN, AET.shape[1], AET.shape[2]))
    valences = torch.empty(QUAN_BIN)
    arousals = torch.empty(QUAN_BIN)

    for bin, indices in quantization.items():
        middle_index = len(indices)//2
        QET[bin] = AET[indices[middle_index]]
        valences[bin] = events_data[indices[middle_index], 1]
        arousals[bin] = events_data[indices[middle_index], 2]

    VET = QET
    VET_events_data = torch.transpose(torch.vstack((VET_timestamps, valences, arousals)), 0, 1)
    return VET, VET_events_data




def getAETChunk(AET: torch.tensor, events_data: torch.tensor):
    SLIDING_WINDOW_TIME = 0.5  # width of sliding window in seconds
    STEP_TIME = 1 / 30  # in seconds DEFAULT: 8/30  TODO: riportare a default
    timestamps = events_data[:, 0]

    chunks_num = int((max(timestamps) - SLIDING_WINDOW_TIME) // STEP_TIME)
    new_AETs, new_events_data = [], []

    for i in range(chunks_num):
        chunk_indices = np.argwhere(np.logical_and( (i * STEP_TIME) <= timestamps, timestamps <= (i * STEP_TIME) + SLIDING_WINDOW_TIME)).flatten()
        new_AETs.append(AET[chunk_indices])
        new_events_data.append(events_data[chunk_indices])

    return new_AETs, new_events_data
