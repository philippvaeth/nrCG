import numpy as np
from regex import T
import torch
import os
import random
import wandb
import pandas as pd


class GradientLogger(object):
    def __init__(self):
        self.prev_grad = None
        self.gradxtm1 = None
        self.xtm1 = None
        self.logging_df = pd.DataFrame()

    @torch.no_grad()
    def cos_sim_time(self, grad_t, logging_name):
        grad_t = grad_t.cpu().flatten(start_dim=1)
        if self.prev_grad is None:
            self.prev_grad = torch.zeros_like(grad_t)
        self.cos_sim(self.prev_grad, grad_t, logging_name=logging_name)
        self.prev_grad = grad_t

    @torch.no_grad()
    def grad_ratio_time(self, gradxt, xt, t, logging_name):
        if self.gradxtm1 is not None:
            outdist = l2_distance(gradxt, self.gradxtm1)
            indist = l2_distance(xt, self.xtm1)

            grad_input_ratio = outdist / (indist + 1e-8)
            grad_input_ratio_std, grad_input_ratio_mean = torch.std_mean(grad_input_ratio, dim=0)

            self.logging_df.loc[f"{wandb.run.name} - {logging_name}meanL", t.item()] = grad_input_ratio_mean.item()
            self.logging_df.loc[f"{wandb.run.name} - {logging_name}stdL", t.item()] = grad_input_ratio_std.item()
            wandb.log(
                {
                    f"{logging_name}_std": grad_input_ratio_std,
                    f"{logging_name}_mean": grad_input_ratio_mean,
                    f"timestep": t.item(),
                }
            )

        self.gradxtm1 = gradxt
        self.xtm1 = xt

    @torch.no_grad()
    def cos_sim(self, grad1, grad2, logging_name):
        grad1 = grad1.cpu().flatten(start_dim=1)
        grad2 = grad2.cpu().flatten(start_dim=1)
        cos_sim = torch.nn.functional.cosine_similarity(grad1, grad2, dim=1)
        # Log individual samples
        log_dict = dict([(f"{logging_name}_{idx}", cos_sim_i) for idx, cos_sim_i in enumerate(cos_sim)])
        wandb.log(log_dict)
        # Log combined metric
        cos_std, cos_mean = torch.std_mean(cos_sim)
        wandb.log({f"{logging_name}_std": cos_std, f"{logging_name}_mean": cos_mean})


class FixedSeed(object):
    def __init__(self, seed: int):
        self.seed = seed

        self.cudnn_benchmark_state = torch.backends.cudnn.benchmark
        self.cudnn_deterministic_state = torch.backends.cudnn.deterministic

        self.state_np = np.random.get_state()
        self.state_torch = torch.random.get_rng_state()
        self.state_torch_cuda = torch.cuda.get_rng_state()

    @staticmethod
    def set_seed(seed: int):
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

        random.seed(seed)
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.manual_seed(seed)

    def __enter__(self):
        self.set_seed(self.seed)
        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        torch.backends.cudnn.benchmark = self.cudnn_benchmark_state
        torch.backends.cudnn.deterministic = self.cudnn_deterministic_state
        return


class StorageWrapper:
    """
    A simple wrapper to store images, tensors and text in a directory.
    Images can also be stored as an animated gif.
    """

    def __init__(self, base_path):
        self.base_path = base_path

    def ensure_dir(self):
        # Ensure that the directory exists
        os.makedirs(self.base_path, exist_ok=True)

    def __enter__(self):
        self.ensure_dir()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        pass

    def write_model(self, filename, model):
        import torch as th

        path = os.path.join(self.base_path, filename)
        th.save(model.state_dict(), path)

    def write_image(self, filename, image, nrow=8):
        # Lazy import so that we don't depend on torchvision.
        import torchvision

        path = os.path.join(self.base_path, filename)
        torchvision.utils.save_image(image, path, nrow=nrow)

    def write_image_with_row_text(self, filename, image, row_texts):
        # Takes a grid of images and creates a grid with nrow images per row.
        # It will then write the row_texts above each row.
        import torchvision, PIL

        nrow = image.shape[0] // len(row_texts)

        grid = torchvision.utils.make_grid(image, nrow=nrow, padding=10, pad_value=1)
        label = torchvision.transforms.ToPILImage()(grid)
        label = label.convert("RGB")
        draw = PIL.ImageDraw.Draw(label)

        row_height = image.shape[2] + 10

        for i, text in enumerate(row_texts):
            draw.text((2, i * row_height - 1), text, (0, 0, 0))

        path = os.path.join(self.base_path, filename)
        label.save(path)

    def write_animated(self, filename, images, format="PNG", row_size=256, ms_delay=6):
        # Format can be PNG, GIF, WEBP
        # Takes a torchvision grid of colored images, takes the rows and makes a gif out of them.

        # Lazy import so that we don't depend on PIL.
        import PIL
        import torch

        imgs = []
        # grab the rows
        for i in range(images.shape[1] // row_size):
            grid = images[:, i * row_size : (i + 1) * row_size, :]
            ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to("cpu", torch.uint8).numpy()
            image = PIL.Image.fromarray(ndarr)
            image = image.convert("RGB")

            # write i on the image
            draw = PIL.ImageDraw.Draw(image)
            draw.text((0, 0), f"t={i+1}", (200, 200, 0))

            imgs.append(image)

        img = imgs[-1]

        path = os.path.join(self.base_path, filename)
        img.save(fp=path, format=format, append_images=imgs, save_all=True, duration=ms_delay, loop=1)

    def write_tensor(self, filename, tensor):
        # Lazy import so that we don't depend on torch.
        import torch as th

        path = os.path.join(self.base_path, filename)
        th.save(tensor, path)

    def write_text(self, filename, text):
        path = os.path.join(self.base_path, filename)
        with open(path, "w") as f:
            f.write(text)

    def write_json(self, filename, data):
        import json

        text = json.dumps(data, indent=4)
        self.write_text(filename, text)

    @staticmethod
    def tempdir(prefix="tmpdir", with_random_text=False):
        from datetime import datetime
        import random

        directory = f"./{prefix}/{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
        if not with_random_text:
            rnd_text = "".join([chr(random.randint(97, 122)) for _ in range(10)])
            directory = f"{directory}-{rnd_text}"

        import os

        if not os.path.exists(directory):
            os.makedirs(directory)

        return StorageWrapper(directory)


# From https://stackoverflow.com/questions/71998978/early-stopping-in-pytorch
class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float("inf")

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


def l2_distance(a, b, precision=torch.float32):
    return (a.to(precision).flatten(start_dim=1) - b.to(precision).flatten(start_dim=1)).pow(2).sum(dim=1).sqrt()


# ? Not possible in float16 due to overflow
def l2_vector_norm(v, precision=torch.float32):
    return torch.linalg.vector_norm(v.to(precision).flatten(start_dim=1), ord=2, dim=1).view(v.shape[0], 1, 1, 1)


def l2_vector_norm_minclipped(v, eps=1e-4):
    norm = l2_vector_norm(v)
    return torch.where(norm < eps, eps, norm)


def l2_normalize_gradient(grad, eps=1e-4, precision=torch.float32):
    grad_norm = l2_vector_norm(grad, precision)
    grad_norm = torch.where(grad_norm < eps, eps, grad_norm)
    gradient_normalized = grad.to(precision) / grad_norm.to(precision)
    return gradient_normalized


def cosine_similarity(a, b, precision=torch.float32):
    return torch.nn.functional.cosine_similarity(
        a.to(precision).flatten(start_dim=1),
        b.to(precision).flatten(start_dim=1),
        dim=1,
    )
