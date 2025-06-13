import torchvision


class BinarizedCelebA(torchvision.datasets.CelebA):
    def __init__(self, multilabel_class_idx, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.multilabel_class_idx = multilabel_class_idx

    def __getitem__(self, idx):
        img, label = super().__getitem__(idx)
        return img, label[self.multilabel_class_idx]
