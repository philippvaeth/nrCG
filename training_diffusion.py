import argparse
from tqdm.auto import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DDPMScheduler
from custom_datasets.sportballs_dataset import SportBallsDataset
from pipeline.load_utils import load_dataset, load_huggingface_dataset
import wandb
import torchvision
from torchvision import transforms

from pipeline.utils import EarlyStopper

parser = argparse.ArgumentParser(description="Diffusion model training.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument("--imgchannels", type=int, default=3, required=False, help="Image channels")
parser.add_argument(
    "--debug",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Toggle experiment tracking for debugging runs",
)
args = parser.parse_args()

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATASET_NAME = args.dataset
IMAGE_SIZE = 64
IMAGE_CHANNELS = args.imgchannels  # 3
SEED = 1234

UNET_BLOCK_OUT_CHANNELS = [64, 128, 192, 256]


num_inference_steps = num_train_timesteps = 400
TRAINING_LR = 1e-4
TRAINING_BATCH_SIZE = 128
TRAINING_EPOCHS = 1000
TRAINING_USE_FP16 = True
TRAINING_SAMPLE_SIZE = 10

DATALOADER_NUM_WORKERS = 8

# %% Load the data set
if DATASET_NAME == "sportballs":
    trainset = SportBallsDataset().load_dataset()["train"]
    collate_fn = None
elif DATASET_NAME == "imagenette":
    IMAGE_SIZE = 160
    trainset = torchvision.datasets.Imagenette(
        root="/data/",
        split="train",
        size="160px",
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        ),
    )
    collate_fn = None
else:
    trainset = load_huggingface_dataset(DATASET_NAME, (IMAGE_SIZE, IMAGE_SIZE), IMAGE_CHANNELS)["train"]
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

# %% Initialize the model
model = UNet2DModel(
    sample_size=IMAGE_SIZE,
    in_channels=IMAGE_CHANNELS,
    out_channels=IMAGE_CHANNELS,
    block_out_channels=UNET_BLOCK_OUT_CHANNELS,
).to(DEVICE)


# %% Define noise scheduler for DPPM

noise_scheduler = DDPMScheduler(num_train_timesteps=num_train_timesteps)

# %% Define optimizer and loss for training
optimizer = torch.optim.AdamW(model.parameters(), lr=TRAINING_LR)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=500,
    num_training_steps=(len(trainloader) * TRAINING_EPOCHS),
)


# %% Initialize accelerator and wandb logging
accelerator = Accelerator(
    mixed_precision="fp16" if TRAINING_USE_FP16 else "no",
    gradient_accumulation_steps=1,
    log_with=None if args.debug else "wandb",
    project_dir="DDPMTraining",
)

early_stopper = EarlyStopper(patience=10, min_delta=1e-6)

model, optimizer, trainloader, lr_scheduler = accelerator.prepare(model, optimizer, trainloader, lr_scheduler)

accelerator.init_trackers(
    "DDPMTraining",
    init_kwargs={"entity": "anonymous"},
    config={
        "device": DEVICE,
        "dataset": DATASET_NAME,
        "imagesize": IMAGE_SIZE,
        "imagechannels": IMAGE_CHANNELS,
        "seed": SEED,
        "unet_block_out_channels": UNET_BLOCK_OUT_CHANNELS,
        "timesteps": num_train_timesteps,
        "lr": TRAINING_LR,
        "batchsize": TRAINING_BATCH_SIZE,
        "epochs": TRAINING_EPOCHS,
        "usefp16": TRAINING_USE_FP16,
        "num_workers": DATALOADER_NUM_WORKERS,
    },
)

# Print model information
model_parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
accelerator.log({"model_parameter_count": model_parameter_count})
print("Model Parameters:     ", f"{model_parameter_count:,}")

# %% TRAIN
for epoch in tqdm(
    range(0, TRAINING_EPOCHS),
    total=TRAINING_EPOCHS,
    desc="Epoch",
    position=1,
    disable=not accelerator.is_local_main_process,
):
    for step, (clean_images, labels) in tqdm(enumerate(trainloader), total=len(trainloader), desc="Batch", position=0):
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(
            0,
            noise_scheduler.config.num_train_timesteps,
            (bs,),
            device=clean_images.device,
        ).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        with accelerator.accumulate(model):
            # Predict the noise residual
            noise_pred = model(noisy_images, timesteps)["sample"]
            loss = F.mse_loss(noise_pred, noise)
            accelerator.backward(loss)

            accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

        logs = {
            "loss": loss.detach().item(),
            "lr": lr_scheduler.get_last_lr()[0],
        }
        accelerator.log(logs)

    if epoch % 50 == 0:
        # Sample images from model
        from pipeline.GuidedPipeline import GuidedDDPMPipeline

        pipeline = GuidedDDPMPipeline(
            unet=accelerator.unwrap_model(model),
            scheduler=noise_scheduler,
        )
        images = pipeline(
            batch_size=TRAINING_SAMPLE_SIZE,
            generator=torch.manual_seed(SEED),
            num_inference_steps=num_inference_steps,
            with_grad=False,
        ).samples
        grid = torchvision.utils.make_grid(images, nrow=TRAINING_SAMPLE_SIZE, normalize=True, scale_each=True)
        images = wandb.Image(grid, caption="Training Samples")
        accelerator.log({"training_samples": images, "epoch": epoch})
        pipeline.save_pretrained(accelerator.trackers[0].run.dir)

    if accelerator.check_trigger():
        break

# Save model
pipeline.save_pretrained(accelerator.trackers[0].run.dir)
accelerator.end_training()
