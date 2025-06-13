import torch
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class SportBallsDataset(Dataset):
    def __init__(self):
        "Custom images to place on the white canvas"
        self.baseball = Image.open("custom_datasets/sportballs/baseball.png", "r")
        self.basketball = Image.open("custom_datasets/sportballs/basketball.png", "r")
        self.volleyball = Image.open("custom_datasets/sportballs/volleyball.png", "r")
        self.soccerball = Image.open("custom_datasets/sportballs/soccerball.png", "r")
        self.sportballs = [self.baseball, self.basketball, self.volleyball, self.soccerball]

        [i.load() for i in self.sportballs]  # Resolves lazy loading of PIL Images for multiple workers in data loader

        self.CANVAS_SIZE = 64
        self.IMAGE_SIZE = 32
        self.N_OBJECTS = 1

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Lambda(lambda t: (t * 2) - 1),
            ]
        )

    def load_dataset(self):
        "Compatibility with huggingface data sets"
        return {"train": self}

    def __len__(self):
        "Custom length for data set"
        return 100000

    def __getitem__(self, idx):
        background = Image.new("RGBA", (self.CANVAS_SIZE, self.CANVAS_SIZE), (255, 255, 255, 255))

        class_idx = torch.randint(0, len(self.sportballs), (self.N_OBJECTS,), generator=torch.manual_seed(idx))
        pos_idx = torch.randint(
            0, self.CANVAS_SIZE - self.IMAGE_SIZE, (self.N_OBJECTS, 2), generator=torch.manual_seed(idx)
        )
        rotation = torch.randint(0, 360, (self.N_OBJECTS,), generator=torch.manual_seed(idx))
        size = torch.randint(3, 9, (self.N_OBJECTS,), generator=torch.manual_seed(idx)) / 10.0

        for cls, pos, rot, s in zip(class_idx, pos_idx, rotation, size):
            obj = self.sportballs[cls].rotate(rot).resize((int(self.IMAGE_SIZE * s), int(self.IMAGE_SIZE * s)))
            background.paste(obj, pos.tolist(), obj)

        background = background.convert("RGB")

        if self.transform:
            background = self.transform(background)

        # return the class(es) as label
        return background, class_idx.item()
