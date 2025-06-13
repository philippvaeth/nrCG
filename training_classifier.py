import argparse
import torchvision
from torchvision import transforms
from tqdm.auto import tqdm
from custom_datasets.sportballs_dataset import SportBallsDataset
from custom_datasets.celeba_dataset import BinarizedCelebA
from pipeline.load_utils import load_classifier, load_huggingface_dataset
from pipeline.ClassifierWrapper import ClassifierWrapper
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from accelerate import Accelerator
from torch.utils.data import DataLoader
import torch
import os

parser = argparse.ArgumentParser(description="Classifier model training.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument(
    "--robustclassifier", type=str, default="False", required=False, help="Toggle robust or non robust classifier"
)
parser.add_argument(
    "--debug", type=str, default="False", required=False, help="Toggle experiment tracking for debugging runs"
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DATASET_NAME = args.dataset
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

TRAINING_LR = 1e-4
CLASSIFIER_CLASS = "mobilenet_v3_small"
CLASSIFIER_ROBUST = eval(args.robustclassifier)
SEED = 1234

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)


TRAINING_USE_FP16 = True
TRAINING_BATCH_SIZE = 128
TRAINING_EPOCHS = 20

DATALOADER_NUM_WORKERS = 8

config_vars = [
    "DEVICE",
    "DATASET_NAME",
    "IMAGE_SIZE",
    "IMAGE_CHANNELS",
    "CLASSIFIER_CLASSES",
    "SEED",
    "DDPM_SCHEDULER_PATH",
    "TRAINING_LR",
    "TRAINING_BATCH_SIZE",
    "TRAINING_EPOCHS",
    "CLASSIFIER_TYPE",
    "CLASSIFIER_ROBUST",
]
# %% Load the data set
collate_fn = None

if args.dataset == "celeba":
    DDPM_SCHEDULER_PATH = "pretrained_models/celeba/scheduler/scheduler_config.json"
    CLASSIFIER_TYPE = "binary"
    CELEBA_CLASS_IDX = 20
    CLASSIFIER_CLASSES = 1
    trainset = BinarizedCelebA(CELEBA_CLASS_IDX, root="/data/", split="train", transform=transform, download=True)
    valset = BinarizedCelebA(CELEBA_CLASS_IDX, root="/data/", split="valid", transform=transform, download=True)
    config_vars.append("CELEBA_CLASS_IDX")
elif args.dataset == "sportballs":
    DDPM_SCHEDULER_PATH = "pretrained_models/sportballs/scheduler/scheduler_config.json"
    CLASSIFIER_TYPE = "multiclass"
    CLASSIFIER_CLASSES = 4
    valset, trainset = torch.utils.data.random_split(
        SportBallsDataset(), [0.2, 0.8], torch.Generator().manual_seed(SEED)
    )
elif args.dataset == "celebahq":
    DDPM_SCHEDULER_PATH = "google/ddpm-celebahq-256"
    CLASSIFIER_TYPE = "binary"
    CLASSIFIER_CLASSES = 1
    IMAGE_SIZE = 256

    trainset = load_huggingface_dataset("korexyz/celeba-hq-256x256", (IMAGE_SIZE, IMAGE_SIZE), IMAGE_CHANNELS)["train"]
    valset = load_huggingface_dataset("korexyz/celeba-hq-256x256", (IMAGE_SIZE, IMAGE_SIZE), IMAGE_CHANNELS)[
        "validation"
    ]
    collate_fn = lambda i: list(torch.utils.data.default_collate(i).values())


trainloader = DataLoader(
    trainset,
    batch_size=TRAINING_BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATALOADER_NUM_WORKERS,
    persistent_workers=True,
    collate_fn=collate_fn,
)
valloader = DataLoader(
    valset,
    batch_size=TRAINING_BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=DATALOADER_NUM_WORKERS,
    persistent_workers=True,
    collate_fn=collate_fn,
)

ddpm_scheduler = DDPMScheduler.from_pretrained(DDPM_SCHEDULER_PATH)


def augment_images(clean_images):
    noise = torch.randn_like(clean_images)
    bs = clean_images.shape[0]

    # Sample a random timestep for each image
    timesteps = torch.randint(
        0,
        ddpm_scheduler.config.num_train_timesteps,
        (bs,),
        device=clean_images.device,
    ).long()

    noise_img = ddpm_scheduler.add_noise(clean_images, noise, timesteps)
    return noise_img


def evaluate(classifier, valloader, robust):
    classifier.eval()
    with torch.no_grad():
        val_loss = 0
        val_correct = 0
        for batch in valloader:
            images, labels = batch
            if robust:
                images = augment_images(images.to(classifier.device))
            loss, preds = classifier.loss_preds(images, labels)
            val_loss += loss.item()
            val_correct += (preds == labels).float().sum()
        val_loss /= len(valloader.dataset)
        val_acc = val_correct / len(valloader.dataset)

    return val_loss, val_acc


# %% Initialize the model
classifier_model = load_classifier(
    classifier=CLASSIFIER_CLASS,
    num_classes=CLASSIFIER_CLASSES,
    in_channels=IMAGE_CHANNELS,
)

classifier = ClassifierWrapper(classifier=classifier_model, classifier_type=CLASSIFIER_TYPE)
classifier.initialize(device=DEVICE)

# %% Define optimizer and loss for training
optimizer = torch.optim.AdamW(classifier.parameters(), lr=TRAINING_LR)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(trainloader) * TRAINING_EPOCHS),
)

# %% Initialize accelerator and wandb logging
if eval(args.debug):
    os.environ["WANDB_MODE"] = "dryrun"
else:
    os.environ["WANDB_MODE"] = "online"


accelerator = Accelerator(
    mixed_precision="fp16" if TRAINING_USE_FP16 else "no",
    gradient_accumulation_steps=1,
    log_with="wandb",
    project_dir="ClassifierTraining",
)

(
    classifier,
    optimizer,
    trainloader,
    valloader,
    lr_scheduler,
) = accelerator.prepare(classifier, optimizer, trainloader, valloader, lr_scheduler)

accelerator.init_trackers(
    "ClassifierTraining",
    init_kwargs={"entity": "anonymous"},
    config={k: v for k, v in locals().items() if k in config_vars},
)

# %% Print model information
model_parameter_count = sum(p.numel() for p in classifier.parameters() if p.requires_grad)
accelerator.log({"classifier_parameter_count": model_parameter_count})
print("Classifier Parameters:     ", f"{model_parameter_count:,}")

# %% Training
prev_val_acc = 0
for epoch in tqdm(range(TRAINING_EPOCHS), desc="Epoch"):
    classifier.train()
    for batch in tqdm(trainloader, desc="Batch"):
        images, labels = batch
        if CLASSIFIER_ROBUST:
            images = augment_images(images.to(classifier.device))
        optimizer.zero_grad()
        loss = classifier.loss(images, labels)
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
    print(f"{loss=}")
    val_loss, val_acc = evaluate(classifier, valloader, robust=CLASSIFIER_ROBUST)
    accelerator.log({"val_loss": val_loss, "val_acc": val_acc})
    print(f"Epoch {epoch}, val_loss: {val_loss:.4f}, val_acc: {val_acc*100:.2f} %")

    # Early stopping if validation accuracy does not improve more than 1 % in an epoch, after epoch 5
    if val_acc - prev_val_acc < 0.01 and epoch > 5:
        print("Early stopping!")
        accelerator.set_trigger()

    prev_val_acc = val_acc

    accelerator.save_model(
        classifier,
        f"{accelerator.trackers[0].run.dir}/" if not eval(args.debug) else "temp/",
    )

    if accelerator.check_trigger():
        break

classifier = accelerator.unwrap_model(classifier, keep_fp32_wrapper=False)
torch.save(
    classifier.state_dict(),
    f"{accelerator.trackers[0].run.dir}/statedict.pt" if not eval(args.debug) else "temp/statedict.pt",
)

# Evaluate clean accuracy
clean_val_loss, clean_val_acc = evaluate(classifier, valloader, robust=False)
# Evaluate noisy image accuracy
noisy_val_loss, noisy_val_acc = evaluate(classifier, valloader, robust=True)


accelerator.log(
    {
        "clean_val_loss": clean_val_loss,
        "clean_val_acc": clean_val_acc,
        "noisy_val_loss": noisy_val_loss,
        "noisy_val_acc": noisy_val_acc,
    }
)
accelerator.end_training()
