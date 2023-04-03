import os

import torch
import torchvision
from torch import nn

from EventsCompressor import EventsCompressor


class EncoderCNN(nn.Module):
    def __init__(self, pretrained=None):
        super(EncoderCNN, self).__init__()

        resnet18 = torchvision.models.resnet18(weights=pretrained, progress=True)
        self.cnn = torch.nn.Sequential(*(list(resnet18.children())[i] for i in [0, 1, 2, 3, 4, 5, 6,7, 8]))
        self.output_dim = 512

    def forward(self, ACEs):  # B x C x M_star x H x W
        ACEs = torch.swapaxes(ACEs, 1, 2)  # B x M_star x C x H x W
        ACEs_batch = ACEs.shape[0]
        ACEs_M_star = ACEs.shape[1]
        encoded_ACEs = self.cnn(torch.flatten(ACEs, end_dim=1)).squeeze()
        encoded_ACEs = torch.swapaxes(encoded_ACEs.reshape((ACEs_batch, ACEs_M_star, -1)), 1, 2)
        return encoded_ACEs


class VideoBranch(nn.Module):
    def __init__(self, input_channels, output_dim):
        super(VideoBranch, self).__init__()

        self.layers = nn.Sequential(
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=0),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(input_channels, input_channels, kernel_size=3, padding=0),
            nn.BatchNorm1d(input_channels),
            nn.LeakyReLU()
        )

        self.pooling = nn.AdaptiveAvgPool1d(output_size=1)

        self.fc_layers = nn.Sequential(
            nn.Linear(input_channels, input_channels // 4),
            nn.LeakyReLU(),
            nn.Linear(input_channels // 4, output_dim),
            nn.LeakyReLU()
        )

        self.embed_dim = output_dim

    def forward(self, encoded_ACEs):
        encoded_ACEs = self.layers(encoded_ACEs)
        encoded_ACEs = self.pooling(encoded_ACEs).squeeze()

        return self.fc_layers(encoded_ACEs)


class FrameBranch(nn.Module):
    def __init__(self, in_channels, compressed_dim):  # M_star = M_cappuccio / compress_ratio
        super(FrameBranch, self).__init__()

        self.grouped_convs = nn.Conv1d(compressed_dim, compressed_dim, kernel_size=in_channels, padding=0, groups=compressed_dim)
        self.normalization = nn.BatchNorm1d(compressed_dim)
        self.activation = nn.LeakyReLU()

        self.embed_dim = compressed_dim

    def forward(self, encoded_ACEs):  # B x C x M_star
        encoded_ACEs = torch.swapaxes(encoded_ACEs, -2, -1)  # B x M_star x C
        return self.activation(self.normalization(self.grouped_convs(encoded_ACEs).squeeze(-1)))


class VALARRegressor(nn.Module):
    def __init__(self, encoding_dim, compressed_dim):
        super(VALARRegressor, self).__init__()

        self.video_branch = VideoBranch(encoding_dim, compressed_dim)
        self.frame_branch = FrameBranch(encoding_dim, compressed_dim)

        # self.regressor = nn.Linear(self.video_branch.embed_dim + self.frame_branch.embed_dim, 1, bias=False)
        self.regressor = nn.Linear(self.video_branch.embed_dim + self.frame_branch.embed_dim, 2, bias=False)
        self.output_limiter = nn.Tanh()

    def forward(self, encoded_ACEs):
        video_pred = self.video_branch(encoded_ACEs)
        frame_preds = self.frame_branch(encoded_ACEs)
        concat_feats = torch.hstack((video_pred, frame_preds))
        return self.output_limiter(self.regressor(concat_feats)), concat_feats


class BET(nn.Module):
    def __init__(self, quantization_bin):
        super(BET, self).__init__()

        self.compressor = EventsCompressor()
        self.encoder = EncoderCNN()
        #
        # self.regressor_VAL = VALARRegressor(self.encoder.output_dim, quantization_bin // self.compressor.compression_ratio)
        # self.regressor_AR = VALARRegressor(self.encoder.output_dim, quantization_bin // self.compressor.compression_ratio)
        #
        self.regressor_VALAR = VALARRegressor(self.encoder.output_dim, quantization_bin // self.compressor.compression_ratio)

    def forward(self, video_chunk):
        # encoded_ACEs = self.encoder(self.compressor(video_chunk))
        # pred_val, = self.regressor_VAL(encoded_ACEs)
        # pred_ar =  self.regressor_AR(encoded_ACEs)
        # return torch.hstack((,))
        return self.regressor_VALAR(self.encoder(self.compressor(video_chunk)))

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.state_dict(), path + "/network.pt")

    def load(self, path):
        self.load_state_dict(torch.load(path))
        self.eval()