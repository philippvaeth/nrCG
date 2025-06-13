import argparse
import torch
from torcheval.metrics import FrechetInceptionDistance
from torchvision import transforms
import torchvision
from custom_datasets.celeba_dataset import BinarizedCelebA
from torch.utils.data import DataLoader
from datasets import load_dataset

from custom_datasets.sportballs_dataset import SportBallsDataset

parser = argparse.ArgumentParser(description="FID comparison.")
parser.add_argument("--dataset", type=str, help="Data set string")
parser.add_argument("--class_idx", type=int, default=None, help="Class IDX for conditional FID")
parser.add_argument("--target_class", type=int, default=0, help="Target class, for celeba: 0 = female, 1 = male")
parser.add_argument("--batch_size", type=int, default=256, help="Number of images in a batch")
parser.add_argument("--n_batches", type=int, default=196, help="Number of batches")
parser.add_argument("--image_size", type=int, default=64, help="Number of batches")
args = parser.parse_args()

# "--dataset", "celebahq",
# "--image_size", "256",
# "--n_batches", "196",
# "--class_idx", "20"

IMAGE_SIZE = args.image_size
SAMPLING_BATCH_SIZE = args.batch_size
N_BATCHES = args.n_batches
N_IMGS = SAMPLING_BATCH_SIZE * N_BATCHES
DEVICE = "cuda:0"
SEED = 1234
CLASS_IDX = args.class_idx


ufid = FrechetInceptionDistance(device=DEVICE)
ufid.real_sum = ufid.real_sum.to(torch.float64)
ufid.real_cov_sum = ufid.real_cov_sum.to(torch.float64)
ufid.fake_sum = ufid.fake_sum.to(torch.float64)
ufid.fake_cov_sum = ufid.fake_cov_sum.to(torch.float64)

transform = transforms.Compose(
    [
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: (t * 2) - 1),
    ]
)

if args.dataset == "celeba":
    trainset = torchvision.datasets.CelebA(root="/data/", split="train", transform=transform, download=True)
    if CLASS_IDX is not None:
        trainset = torch.stack([trainset[i][0] for i in range(len(trainset)) if trainset[i][1][CLASS_IDX] == 0], dim=0)
    else:
        trainset = torch.stack([trainset[i][0] for i in range(len(trainset))], dim=0)

    valset = torchvision.datasets.CelebA(root="/data/", split="valid", transform=transform, download=True)
    valset = torch.stack([_[0] for _ in valset], dim=0)
elif args.dataset == "sportballs":
    valset, trainset = torch.utils.data.random_split(
        SportBallsDataset(), [0.2, 0.8], torch.Generator().manual_seed(SEED)
    )

    if CLASS_IDX is not None:
        trainset = torch.stack([_[0] for _ in trainset if _[1] == CLASS_IDX], dim=0)
    else:
        trainset = torch.stack([_[0] for _ in trainset], dim=0)
    valset = torch.stack([_[0] for _ in valset], dim=0)
elif args.dataset == "celebahq":
    ds = load_dataset("korexyz/celeba-hq-256x256", cache_dir="/data")
    if CLASS_IDX is not None:
        trainset = torch.stack(
            [
                transform(ds["train"][idx]["image"])
                for idx in range(len(ds["train"]))
                if ds["train"][idx]["label"] == int(args.target_class)
            ],
            dim=0,
        )
    else:
        trainset = torch.stack([transform(ds["train"][idx]["image"]) for idx in range(len(ds["train"]))], dim=0)

    valset = torch.stack([transform(ds["validation"][idx]["image"]) for idx in range(len(ds["validation"]))], dim=0)


train_idxs = torch.randint(low=0, high=len(trainset), size=(N_IMGS,), generator=torch.manual_seed(SEED))
trainimgs = torch.stack([trainset[i] for i in train_idxs], dim=0).add(1).div(2).to(torch.float32)

assert trainimgs.min() >= 0 and trainimgs.max() <= 1.0

trainloader = DataLoader(
    trainimgs,
    batch_size=SAMPLING_BATCH_SIZE,
)

for batch in trainloader:
    ufid.update(batch.to(DEVICE), is_real=True)

# SAVE state dict of FID calculation
if CLASS_IDX is not None:
    torch.save(
        ufid.state_dict(),
        f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{N_IMGS}_class{CLASS_IDX}_{args.target_class}.pt",
    )
else:
    torch.save(ufid.state_dict(), f"pretrained_models/{args.dataset}/fid/{args.dataset}_fid_real_{N_IMGS}.pt")

valimgs = valset.add(1).div(2).to(torch.float32)
assert valimgs.min() >= 0 and valimgs.max() <= 1.0

valloader = DataLoader(
    valimgs,
    batch_size=SAMPLING_BATCH_SIZE,
)
# ? FID in torcheval expects [0,1] float32
# FID TEST
for batch in valloader:
    ufid.update(batch.to(DEVICE), is_real=False)

a = ufid.compute()
print(f"FID for {N_IMGS} images:", a)
