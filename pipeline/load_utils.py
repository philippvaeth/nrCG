import torch
from torch.utils.data import random_split
import torchvision
import torchvision.transforms as transforms
from torchvision.models import mobilenet_v3_small, mobilenet_v3_large, squeezenet1_1, efficientnet_b3
from safetensors.torch import load_model

from datasets import load_dataset


def load_sportballs_dataset(path, shape, split_ratio=0.8):
    # Loads a custom dataset from a folder of images
    transform = transforms.Compose(
        [
            transforms.Resize(shape),
            transforms.ToTensor(),
            transforms.Lambda(lambda t: (t * 2) - 1),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(root=path, transform=transform)

    # split
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return {"train": train_dataset, "val": val_dataset, "all": dataset}


def load_huggingface_dataset(dataset_str, shape, channels):

    def transform(examples):
        pytorch_transform = transforms.Compose(
            [
                transforms.Lambda(lambda t: t.convert("RGB") if channels == 3 else t.convert("L")),
                transforms.Resize(shape),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )
        examples["image"] = [pytorch_transform(image) for image in examples["image"]]
        return examples

    dataset = load_dataset(dataset_str, cache_dir="/data")
    dataset.set_transform(transform)
    return dataset


def load_classifier(classifier: str, num_classes, in_channels, weights_path=None, wrapped=False):
    if classifier == "squeeze_net":
        model = squeezenet1_1(num_classes=num_classes)
        if in_channels is not None:
            model.features[0] = torch.nn.Conv2d(
                in_channels,
                model.features[0].out_channels,
                kernel_size=model.features[0].kernel_size,
                stride=model.features[0].stride,
                padding=model.features[0].padding,
            )  # Necessary for grayscale inputs
    elif classifier == "mobilenet_v3_small":
        model = mobilenet_v3_small(num_classes=num_classes)
        if in_channels is not None:
            model.features[0][0] = torch.nn.Conv2d(
                in_channels,
                model.features[0][0].out_channels,
                kernel_size=model.features[0][0].kernel_size,
                stride=model.features[0][0].stride,
                padding=model.features[0][0].padding,
                bias=model.features[0][0].bias,
            )  # Necessary for grayscale inputs
    elif classifier == "mobilenet_v3_large":
        model = mobilenet_v3_large(num_classes=num_classes)
        if in_channels is not None:
            model.features[0][0] = torch.nn.Conv2d(
                in_channels,
                model.features[0][0].out_channels,
                kernel_size=model.features[0][0].kernel_size,
                stride=model.features[0][0].stride,
                padding=model.features[0][0].padding,
                bias=model.features[0][0].bias,
            )  # Necessary for grayscale inputs
    elif classifier == "efficientnet_b3":
        model = efficientnet_b3(num_classes=num_classes)
    elif classifier == "linear":
        # define a VERY simple linear model
        model = torch.nn.Sequential(
            # flatten the input tensor
            torch.nn.Flatten(),
            # linear layer
            torch.nn.Linear(3 * 64 * 64, num_classes),
        )
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(64 * 32 * 32, num_classes),
        )
        model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Dropout(0.25),
            torch.nn.Flatten(),
            torch.nn.Linear(32 * 32 * 32, num_classes),
        )

    else:
        raise NotImplementedError(f"classifier {classifier} not supported, must be 'mnist' or 'celeba'!")

    if weights_path is not None:
        if weights_path.endswith(".safetensors"):
            load_model(model=model, filename=weights_path)

        elif weights_path.endswith(".pt") or weights_path.endswith(".pth"):
            import torch.nn as nn

            class Model(nn.Module):
                def __init__(self, model):
                    super().__init__()
                    self.model = model

                def forward(self, x):
                    return self.model(x)

            mx = Model(model)
            mx.load_state_dict(torch.load(weights_path))
    return model


def load_sample_images(num_samples, device, dataset, cls_sub_index=None):
    assert num_samples
    if cls_sub_index is not None:
        images = [dataset[i][0] for i in cls_sub_index]
        labels = [dataset[i][1] for i in cls_sub_index]
    else:
        # load the first N images from mnist that have cls as target class
        # grab the first num_samples
        images, labels = [], []
        count = 0
        for m in dataset:
            if len(images) >= num_samples:
                break
            if m[1].item() == 1:
                print(count)
                images.append(m[0])
                labels.append(m[1])
            count += 1
    images = torch.stack(images, dim=0).to(device)

    if isinstance(labels, list) and not isinstance(labels[0], torch.Tensor):
        labels = torch.tensor(labels).to(device)
    else:
        labels = torch.stack(labels, dim=0).to(device)

    return images, labels
