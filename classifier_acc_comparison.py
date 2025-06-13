import argparse
import os
from torchvision import transforms

from custom_datasets.celeba_dataset import BinarizedCelebA
from pipeline.utils import FixedSeed

import torch
from torch.utils.data import DataLoader
import wandb
from custom_datasets.sportballs_dataset import SportBallsDataset
from pipeline.GuidedPipeline import GuidedDDPMPipeline
from pipeline.load_utils import load_classifier
from pipeline.ClassifierWrapper import ClassifierWrapper
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

parser = argparse.ArgumentParser(description="Classifier comparison.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument(
    "--debug", type=str, default="False", required=False, help="Toggle experiment tracking for debugging runs"
)
args = parser.parse_args()

# %% Initialize wandb logging
if eval(args.debug):
    os.environ["WANDB_MODE"] = "dryrun"
else:
    os.environ["WANDB_MODE"] = "online"

wandb.init(
    project="nrCG_Classifier_Accuracy",
)


DEVICE = "cuda:0"
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

CLASSIFIER_CLASS = "mobilenet_v3_small"
BATCH_SIZE = 2**10
TRAINING_EPOCHS = 200

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
        precision=torch.float16,
        weights_path=path,
    )
    return cls


ddpm = GuidedDDPMPipeline.from_pretrained(PATH_PIPELINE).to(DEVICE)


classifiers = {
    "Non-robust": prep_cls(weights_path_nonrobust),
    "Non-robust-xzero": prep_cls(weights_path_nonrobust),
    "Robust": prep_cls(weights_path_robust),
    "Robust-xzero": prep_cls(weights_path_robust),
}

ddpm_scheduler = DDPMScheduler.from_pretrained(DDPM_SCHEDULER_PATH)


with torch.no_grad(), FixedSeed(SEED):
    for t in range(0, ddpm_scheduler.config.num_train_timesteps):
        for name, cls in classifiers.items():
            val_correct = 0
            softmax_scores = []
            for batch in valloader:
                images, labels = batch
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                if t > 0:
                    images = ddpm_scheduler.add_noise(
                        images, torch.randn_like(images), (torch.ones((images.shape[0],)) * t).long()
                    )

                if name in ["Non-robust-xzero", "Robust-xzero"]:
                    ddpm_output = ddpm.unet(images, t).sample
                    images = ddpm.scheduler.step(ddpm_output, t, images).pred_original_sample

                preds, probs = cls.preds_probs(images, labels)
                val_correct += (preds == labels).float().sum()
                softmax_scores.append(probs)

            # Validation accuracy
            val_acc = val_correct / len(valloader.dataset)
            prob_std, prob_mean = torch.std_mean(torch.cat(softmax_scores))

            # Class-conditional FID

            wandb.log(
                {f"{name}_valacc": val_acc, f"{name}_probmean": prob_mean, f"{name}_probstd": prob_std, "timestep": t}
            )
