import argparse
import json
import os
import pandas as pd
from torchvision import transforms

from custom_datasets.celeba_dataset import BinarizedCelebA
from pipeline.Guidance import EMA, ADAMGradientStabilization
from pipeline.utils import FixedSeed, l2_distance

import torch
from torch.utils.data import DataLoader

# import wandb
from custom_datasets.sportballs_dataset import SportBallsDataset
from pipeline.GuidedPipeline import GuidedDDPMPipeline
from pipeline.load_utils import load_classifier
from pipeline.ClassifierWrapper import ClassifierWrapper
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

parser = argparse.ArgumentParser(description="Classifier comparison.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument(
    "--debug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Toggle experiment tracking for debugging runs",
)
parser.add_argument(
    "--notes",
    type=str,
    required=False,
    help="Notes for runs",
)
args = parser.parse_args()


DEVICE = "cuda:0"
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

CLASSIFIER_CLASS = "mobilenet_v3_small"
BATCH_SIZE = 2**9
TARGET_CLASS = 0

DATALOADER_NUM_WORKERS = 8
SEED = 1234

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

if args.dataset == "sportballs":
    weights_path_nonrobust = f"pretrained_models/sportballs/sleek-river-105.safetensors"
    weights_path_robust = f"pretrained_models/sportballs/upbeat-leaf-106.safetensors"
    PATH_PIPELINE = "./pretrained_models/sportballs"
    DDPM_SCHEDULER_PATH = "pretrained_models/sportballs/scheduler/scheduler_config.json"
    CLASSIFIER_CLASSES = 4
    CLASSIFIER_TYPE = "multiclass"
    valset, _ = torch.utils.data.random_split(SportBallsDataset(), [0.2, 0.8], torch.Generator().manual_seed(SEED))
elif args.dataset == "celeba":
    weights_path_nonrobust = "pretrained_models/celeba/classic-aardvark-104.safetensors"
    weights_path_robust = "pretrained_models/celeba/golden-leaf-103.safetensors"
    PATH_PIPELINE = "./pretrained_models/celeba"
    DDPM_SCHEDULER_PATH = "pretrained_models/celeba/scheduler/scheduler_config.json"
    CLASSIFIER_TYPE = "binary"
    CELEBA_CLASS_IDX = 20
    CLASSIFIER_CLASSES = 1
    valset = BinarizedCelebA(CELEBA_CLASS_IDX, root="/data/", split="valid", transform=transform, download=True)

valloader = DataLoader(
    valset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATALOADER_NUM_WORKERS,
    persistent_workers=True,
    drop_last=True,
)


def prep_cls(path):
    classifier = load_classifier(
        classifier=CLASSIFIER_CLASS,
        num_classes=CLASSIFIER_CLASSES,
        in_channels=IMAGE_CHANNELS,
    )

    cls = ClassifierWrapper(classifier=classifier, classifier_type=CLASSIFIER_TYPE)
    cls.initialize(
        device=DEVICE,
        precision=torch.float32,
        weights_path=path,
    )
    return cls


ddpm = GuidedDDPMPipeline.from_pretrained(PATH_PIPELINE).to(DEVICE)

classifiers = {
    "Non-robust": prep_cls(weights_path_nonrobust),
    "Non-robust-xzero": prep_cls(weights_path_nonrobust),
    # "Robust": prep_cls(weights_path_robust),
}


ddpm_scheduler = DDPMScheduler.from_pretrained(DDPM_SCHEDULER_PATH)
timestep_iter = range(1, ddpm_scheduler.config.num_train_timesteps)

# Logging strategy: 4 classifiers * 77 batches * 400 time steps * 4 metrics * 256 images in a batch * 32 bits per metric = 500 MB
log_tensor = torch.empty((len(classifiers), len(valloader), len(timestep_iter), 4, BATCH_SIZE), device=DEVICE)

##? Use EMA oder ADAM
# ema_t = EMA(
#     input_shape=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE, beta=0.99, bias_correction=False
# )
# ema_tm1 = EMA(
#     input_shape=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE, beta=0.99, bias_correction=False
# )
ema_t = ADAMGradientStabilization(
    input_shape=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE, bias_correction=False
)
ema_tm1 = ADAMGradientStabilization(
    input_shape=(BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE, bias_correction=False
)


with torch.no_grad(), FixedSeed(SEED):
    for cls_idx, (name, cls) in enumerate(classifiers.items()):
        print(f"Starting classifier {name}")

        for batch_idx, (images, labels) in enumerate(valloader):
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            targets = cls.extend_target_classes(TARGET_CLASS, images.shape[0]).to(cls.device).to(cls.label_dtype)

            for t in reversed(timestep_iter):

                noise = torch.randn_like(images, device=DEVICE)

                # x_0 -> x_t
                images_t = ddpm_scheduler.add_noise(
                    images, noise, (torch.ones((images.shape[0],)) * t).long()
                ).requires_grad_(True)
                with torch.enable_grad():
                    # x_0-prediction
                    if name in ["Non-robust-xzero", "Robust-xzero"]:
                        ddpm_output = ddpm.unet(images_t, t).sample
                        images_x0_t = ddpm.scheduler.step(ddpm_output, t, images_t).pred_original_sample
                        cls_logits_t = cls(images_x0_t)
                    else:
                        cls_logits_t = cls(images_t)

                    cls_grad_t = torch.autograd.grad(
                        -cls.loss_fn(cls_logits_t.squeeze(), targets),
                        images_t,
                    )[0]

                # x_0 -> x_{t-1}
                images_tm1 = ddpm_scheduler.add_noise(
                    images, noise, (torch.ones((images.shape[0],)) * (t - 1)).long()
                ).requires_grad_(True)
                with torch.enable_grad():
                    # x_0-prediction
                    if name in ["Non-robust-xzero", "Robust-xzero"]:
                        ddpm_output = ddpm.unet(images_tm1, t).sample
                        images_x0_tm1 = ddpm.scheduler.step(ddpm_output, t, images_tm1).pred_original_sample
                        cls_logits_tm1 = cls(images_x0_tm1)
                    else:
                        cls_logits_tm1 = cls(images_tm1)

                    cls_grad_tm1 = torch.autograd.grad(
                        -cls.loss_fn(cls_logits_tm1.squeeze(), targets),
                        images_tm1,
                    )[0]

                # Compare output logits differences
                outdist = l2_distance(cls_logits_t, cls_logits_tm1)
                log_tensor[cls_idx, batch_idx, t - 1, 0] = outdist

                # Compare gradient differences
                graddist = l2_distance(cls_grad_t, cls_grad_tm1)
                log_tensor[cls_idx, batch_idx, t - 1, 1] = graddist

                # Compare input distances
                indist = l2_distance(images_t, images_tm1)
                log_tensor[cls_idx, batch_idx, t - 1, 2] = indist

                # Compare EMA
                graddist_ema = l2_distance(ema_t(cls_grad_t), ema_tm1(cls_grad_tm1))
                log_tensor[cls_idx, batch_idx, t - 1, 3] = graddist_ema

torch.save(log_tensor, "logs_adam.pt")
