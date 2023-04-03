import torch
from online_triplet_loss.losses import batch_hard_triplet_loss
from torch import nn


class WMSELoss(nn.Module):
    def __init__(self, labels, regularization=False):
        super(WMSELoss, self).__init__()
        unique_labels, counts = torch.unique(labels, return_counts=True)
        label_freq = counts / len(labels)

        self.label_to_index = {str(unique_labels[i].item()): i for i in range(len(unique_labels))}

        self.weights = 1 / label_freq
        self.weights = self.weights / torch.sum(self.weights)

        if regularization:
            self.regularizer_coef = 100
        else:
            self.regularizer_coef = 1

    def forward(self, preds, labels):
        sample_weights = self.weights.to(labels.device)[[self.label_to_index[str(value.item())] for value in labels]]
        return self.regularizer_coef * (torch.sum((sample_weights * (labels - preds) ** 2)) / len(labels))


class VideoBETLoss(nn.Module):
    def __init__(self, type="mae", alpha=0.5, beta=0.25, gamma=0.25, **kwargs):
        super(VideoBETLoss, self).__init__()

        if type == "rmse":
            loss_type = RMSELoss()

        elif type == "mse":
            loss_type = nn.MSELoss()

        elif type == "wmse":
            self.valence_loss = WMSELoss(kwargs["labels"][:, 0], regularization=True)
            self.arousal_loss = WMSELoss(kwargs["labels"][:, 1], regularization=True)
            loss_type = self.valence_loss

        elif type == "mae":
            loss_type = nn.L1Loss()

        else:
            raise

        if type != "wmse":
            self.valence_loss = loss_type
            self.arousal_loss = loss_type

        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

        self.type = type

        self.num_losses = 5

        self.bce = nn.BCELoss()

        self.names = [name + loss_type.__class__.__name__ for name in ["COMB_", "VALENCE_", "AROUSAL_"]] + ["TRIPLET", "BCE"]

    def forward(self, regression_values, labels, embeddings):

        # type loss
        valence_loss = self.valence_loss(regression_values[:, 0], labels[:, 0])
        arousal_loss = self.arousal_loss(regression_values[:, 1], labels[:, 1])
        type_loss = (valence_loss + arousal_loss) / 2

        # triplet_loss
        MARGIN = 0.5
        embeddings = torch.nn.functional.normalize(embeddings)
        valence_triplet = batch_hard_triplet_loss(labels[:, 0], embeddings, margin=MARGIN)
        arousal_triplet = batch_hard_triplet_loss(labels[:, 1], embeddings, margin=MARGIN)
        triplet = (valence_triplet + arousal_triplet) / 2


        # Binary Cross Entropy
        # sgns = torch.sign(regression_values)
        labels_sgns = torch.sign(labels)
        # sgns[sgns == 0] = 1
        labels_sgns[labels_sgns == 0] = 1
        labels_sgns[labels_sgns == -1] = 0

        valence_bce = self.bce(torch.sigmoid(regression_values[:, 0]), labels_sgns[:, 0])
        arousal_bce = self.bce(torch.sigmoid(regression_values[:, 1]), labels_sgns[:, 1])
        # valence_bce = self.bce(torch.sigmoid(sgns[:, 0]), labels_sgns[:, 0])
        # arousal_bce = self.bce(torch.sigmoid(sgns[:, 1]), labels_sgns[:, 1])
        bce = (valence_bce + arousal_bce) / 2

        comb = (self.alpha * type_loss) + (self.beta * triplet) + (self.gamma * bce)
        return comb, valence_loss, arousal_loss, triplet, bce


class RMSELoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(RMSELoss, self).__init__()
        self.mse = nn.MSELoss()
        self.eps = eps

    def forward(self, pred, y):
        return torch.sqrt(self.mse(pred, y) + self.eps)
