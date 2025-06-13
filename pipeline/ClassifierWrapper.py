from enum import Enum
import torch
from typing import List, Union


def binary_probs(logits, y):
    probs = logits.squeeze().sigmoid()
    return torch.cat((probs[y == 1], (1 - (probs[y == 0]))))


class ClassifierWrapper(torch.nn.Module):
    def __init__(self, classifier: torch.nn.Module, classifier_type: Union["multiclass", "binary"]):
        super().__init__()
        self.classifier = classifier
        self.device = list(self.classifier.parameters())[0].device
        self.classifier_type = classifier_type

        if classifier_type == "multiclass":
            self.loss_fn = torch.nn.CrossEntropyLoss(reduction="sum")
            self.logit_fn = lambda logits, labels: logits[range(len(labels)), labels]
            self.pred_fn = lambda logits: logits.argmax(dim=1)
            self.label_dtype = torch.long
            self.probs_fn = lambda logits, labels: logits.softmax(dim=1)[
                range(0, logits.shape[0]), labels.long()
            ].squeeze()
        elif classifier_type == "binary":
            self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="sum")
            self.pred_fn = lambda logits: (logits.squeeze().sigmoid() >= 0.5).long()
            self.label_dtype = torch.float
            self.probs_fn = binary_probs

    def initialize(self, device: torch.device = None, precision=None, weights_path=None):
        if weights_path is not None:
            self.load(weights_path)
        if device is not None:
            self.classifier.to(device)
        if precision is not None:
            self.classifier = self.classifier.to(precision)
        self.classifier.eval()
        self.device = list(self.classifier.parameters())[0].device
        self.precision = precision

    def extend_target_classes(self, sampling_target_class, sample_size):
        sampling_target_class = str(sampling_target_class)
        return (torch.ones(sample_size, dtype=torch.long) * eval(sampling_target_class)).to(self.device)

    def predict_labels(self, images: torch.Tensor):
        logits = self.classifier(images.to(self.device).to(self.precision))
        label_pred = self.pred_fn(logits)
        return label_pred

    def loss(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.classifier(x.to(self.device).to(self.precision))
        return self.loss_fn(logits.squeeze(), y.to(self.device).to(self.label_dtype))

    def loss_preds(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.classifier(x.to(self.device).to(self.precision))
        return self.loss_fn(logits.squeeze(), y.to(self.device).to(self.label_dtype)), self.pred_fn(logits)

    def preds_probs(self, x: torch.Tensor, y: torch.Tensor):
        logits = self.classifier(x.to(self.device).to(self.precision))
        return self.pred_fn(logits), self.probs_fn(logits, y)

    def __call__(self, *args, **kwargs):
        return self.classifier(*args, **kwargs)

    def parameters(self):
        return self.classifier.parameters()

    def train(self):
        self.classifier.train()

    def eval(self):
        self.classifier.eval()

    def load(self, path):
        from safetensors.torch import load_model

        load_model(self, path)
