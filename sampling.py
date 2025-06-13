import argparse
import os
import torchvision
from pipeline.Guidance import EMA, ADAMGradientStabilization, DiffusionGuidance
from pipeline.utils import FixedSeed
import torch
from torch.utils.data import DataLoader
import wandb
from pipeline.GuidedPipeline import GuidedDDPMPipeline
from pipeline.load_utils import load_classifier
from pipeline.ClassifierWrapper import ClassifierWrapper
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from torcheval.metrics import FrechetInceptionDistance

parser = argparse.ArgumentParser(description="Classifier comparison.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument("--target_class", type=int, default=0, help="Target class, for celeba: 0 = female, 1 = male")
parser.add_argument("--guidance_scale", type=float, default=1.0, help="Guidance scale")
parser.add_argument("--guidance_stabilization", type=str, default="none", help="Guidance stabilization technique")
parser.add_argument("--guidance_beta", type=float, default=0.9, help="Beta in Guidance stabilization")
parser.add_argument(
    "--robust_classifier",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="Robust classifier",
)
parser.add_argument(
    "--xzeroprediction",
    default=False,
    action=argparse.BooleanOptionalAction,
    help="X-zero prediction before the classifier",
)
parser.add_argument(
    "--debug",
    type=str,
    default="False",
    required=False,
    help="Toggle experiment tracking for debugging runs",
)
parser.add_argument(
    "--notes",
    type=str,
    required=False,
    help="Notes for runs",
)
args = parser.parse_args()

# %% Initialize wandb logging
if eval(args.debug):
    os.environ["WANDB_MODE"] = "dryrun"
else:
    os.environ["WANDB_MODE"] = "online"


DEVICE = "cuda:0"
DATASET_NAME = args.dataset
IMAGE_SIZE = 64
IMAGE_CHANNELS = 3

CLASSIFIER_CLASS = "mobilenet_v3_small"
TRAINING_BATCH_SIZE = 128
TRAINING_EPOCHS = 200

DATALOADER_NUM_WORKERS = 8
SEED = 1234
SAMPLING_BATCH_SIZE = 1024  # images per batch
N_BATCHES = 49  # number of batches
TARGET_CLASS = args.target_class
CLASSIFIER_SCALING = args.guidance_scale
GRADIENT_NORMALIZATION = False


if args.dataset == "sportballs":
    weights_path_nonrobust = "pretrained_models/sportballs/light-dew-109.safetensors"
    weights_path_robust = "pretrained_models/sportballs/apricot-galaxy-108.safetensors"
    PATH_PIPELINE = "./pretrained_models/sportballs"
    DDPM_SCHEDULER_PATH = "pretrained_models/sportballs/scheduler/scheduler_config.json"
    CLASSIFIER_CLASSES = 4
    CLASSIFIER_TYPE = "multiclass"
    TARGET_CLASS = 0
    FID_UNCOND_PATH = (
        f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{SAMPLING_BATCH_SIZE*N_BATCHES}.pt"
    )
    FID_COND_PATH = (
        f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{SAMPLING_BATCH_SIZE*N_BATCHES}_class0.pt"
    )
elif args.dataset == "celeba":
    weights_path_nonrobust = "pretrained_models/celeba/classic-aardvark-104.safetensors"
    weights_path_robust = "pretrained_models/celeba/golden-leaf-103.safetensors"
    PATH_PIPELINE = "./pretrained_models/celeba"
    DDPM_SCHEDULER_PATH = "pretrained_models/celeba/scheduler/scheduler_config.json"
    CLASSIFIER_TYPE = "binary"
    CELEBA_CLASS_IDX = 20
    CLASSIFIER_CLASSES = 1
    FID_UNCOND_PATH = (
        f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{SAMPLING_BATCH_SIZE*N_BATCHES}.pt"
    )
    FID_COND_PATH = f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{SAMPLING_BATCH_SIZE*N_BATCHES}_class{CELEBA_CLASS_IDX}.pt"
elif args.dataset == "celebahq":
    weights_path_nonrobust = "pretrained_models/celebahq/summer-snowflake-110.safetensors"
    weights_path_robust = "None"
    CELEBA_CLASS_IDX = 20
    CLASSIFIER_TYPE = "binary"
    CLASSIFIER_CLASSES = 1

    PATH_PIPELINE = "google/ddpm-celebahq-256"
    DDPM_SCHEDULER_PATH = "google/ddpm-celebahq-256"
    SAMPLING_BATCH_SIZE = 64
    N_BATCHES = 16
    IMAGE_SIZE = 256

    FID_COND_PATH = f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{SAMPLING_BATCH_SIZE*N_BATCHES}_class{CELEBA_CLASS_IDX}_{TARGET_CLASS}.pt"
    FID_UNCOND_PATH = (
        f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{SAMPLING_BATCH_SIZE*N_BATCHES}.pt"
    )


INPUT_SHAPE = (SAMPLING_BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)


config_vars = [
    "DEVICE",
    "DATASET_NAME",
    "IMAGE_SIZE",
    "IMAGE_CHANNELS",
    "CLASSIFIER_CLASS",
    "TRAINING_BATCH_SIZE",
    "TRAINING_EPOCHS",
    "DATALOADER_NUM_WORKERS",
    "SEED",
    "SAMPLING_BATCH_SIZE",
    "TARGET_CLASS",
    "CLASSIFIER_SCALING",
    "GRADIENT_NORMALIZATION",
    "N_BATCHES",
]
wandb.init(
    project="nrCG",
    config={k: v for k, v in locals().items() if type(v) in [str, int, float, bool]},
)


def prep_cls(path):
    classifier = load_classifier(
        classifier=CLASSIFIER_CLASS,
        num_classes=CLASSIFIER_CLASSES,
        in_channels=IMAGE_CHANNELS,
    )
    # /REPLACE

    cls = ClassifierWrapper(classifier=classifier, classifier_type=CLASSIFIER_TYPE)
    cls.initialize(
        device=DEVICE,
        precision=torch.float32,
        weights_path=path,
    )
    return cls


if args.robust_classifier:
    cls = prep_cls(weights_path_robust)
    cls_name = "Robust"
else:
    cls = prep_cls(weights_path_nonrobust)
    cls_name = "Non-robust"

if args.xzeroprediction:
    cls_name += "-xzero"


ddpm_scheduler = DDPMScheduler.from_pretrained(DDPM_SCHEDULER_PATH)

# ? FID expects images in [0,1]
# load FID pre-computed statistics for training data
ufid = FrechetInceptionDistance(device=DEVICE)
ufid.load_state_dict(torch.load(FID_UNCOND_PATH))

cfid = FrechetInceptionDistance(device=DEVICE)
cfid.load_state_dict(torch.load(FID_COND_PATH))

target_classes = cls.extend_target_classes(TARGET_CLASS, SAMPLING_BATCH_SIZE)

if args.guidance_stabilization == "none":
    guidance_stabilization = None
elif args.guidance_stabilization == "adam":
    guidance_stabilization = ADAMGradientStabilization(input_shape=INPUT_SHAPE, device=DEVICE)
elif args.guidance_stabilization == "ema":
    guidance_stabilization = EMA(
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        beta=args.guidance_beta,
        bias_correction=False,
    )
elif args.guidance_stabilization == "ema09":
    guidance_stabilization = EMA(
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        beta=0.9,
        bias_correction=False,
    )
elif args.guidance_stabilization == "ema099":
    guidance_stabilization = EMA(
        input_shape=INPUT_SHAPE,
        device=DEVICE,
        beta=0.99,
        bias_correction=False,
    )

cf_guidance = DiffusionGuidance(
    input_shape=INPUT_SHAPE,
    classifier_wrapper=cls,
    classifier_target=target_classes,
    classifier_scaling=CLASSIFIER_SCALING,
    xzeroprediction=args.xzeroprediction,
    guidance_stabilization=guidance_stabilization,
    gradient_normalization=GRADIENT_NORMALIZATION,
    precision=torch.float32,
)

diffusion_pipeline = GuidedDDPMPipeline.from_pretrained(PATH_PIPELINE).to(DEVICE)
diffusion_pipeline.set_guidance(cf_guidance)
diffusion_pipeline.set_precision(torch.float16)
diffusion_pipeline.set_name(cls_name)

################################################################################
################################################################################
# Logging strategy: 5 metrics * batches * batch_size images in a batch * 32 bits per metric
log_image_tensor = torch.empty((N_BATCHES, SAMPLING_BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE), device=DEVICE)

for i in range(N_BATCHES):
    # %% Generate the samples
    # The pipeline will generate classifier guided samples
    with torch.enable_grad(), FixedSeed(SEED + i):
        _, batch_samples = diffusion_pipeline(
            batch_size=SAMPLING_BATCH_SIZE,
            num_inference_steps=diffusion_pipeline.scheduler.config.num_train_timesteps,
            generator=torch.manual_seed(SEED + i),
            with_grad=True if args.xzeroprediction else False,
            return_dict=False,
        )

    log_image_tensor[i] = batch_samples

samples = log_image_tensor.view(N_BATCHES * SAMPLING_BATCH_SIZE, IMAGE_CHANNELS, IMAGE_SIZE, IMAGE_SIZE)
target_classes = target_classes.repeat_interleave(N_BATCHES)

with torch.no_grad():
    # samples in [-1,1] range
    preds, probs = cls.preds_probs(samples, target_classes)
    target_acc = (preds == target_classes).float().mean()
    prob_std, prob_mean = torch.std_mean(probs)

    samples = (samples + 1) / 2  # to [0,1] range
    valloader = DataLoader(samples, batch_size=SAMPLING_BATCH_SIZE)  # for batching

    for batch in valloader:
        ufid.update(batch.to(DEVICE), is_real=False)

    # Transfer sampled batch statistics to conditional fid to avoid double computation
    cfid.fake_sum = ufid.fake_sum.clone()
    cfid.fake_cov_sum = ufid.fake_cov_sum.clone()
    cfid.num_fake_images = ufid.num_fake_images.clone()

    ufid_score = ufid.compute()
    cfid_score = cfid.compute()

sample_dir = f"samples/{wandb.run.settings.run_mode}-{wandb.run.settings.timespec}-{wandb.run.settings.run_id}"
os.makedirs(sample_dir)
torch.save(samples.mul(255).to(torch.uint8), f"{sample_dir}/samples.pt")  # Samples in range [0,1]

wandb.log(
    {
        f"10samples": wandb.Image(torchvision.utils.make_grid(samples[:10], nrow=5), caption=f"{cls_name}_samples"),
        f"target_acc": target_acc.item(),
        f"probmean": prob_mean.item(),
        f"probstd": prob_std.item(),
        f"ufid": ufid_score.item(),
        f"cfid": cfid_score.item(),
    }
)
