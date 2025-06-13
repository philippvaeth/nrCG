import torch
import logging
from tqdm import tqdm
from pipeline.ClassifierWrapper import ClassifierWrapper
from pipeline.utils import l2_normalize_gradient

log = logging.getLogger(__name__)

WANDB_LOGGING = True
if WANDB_LOGGING:
    import wandb


class DiffusionGuidance:
    def __init__(
        self,
        input_shape,
        classifier_wrapper: ClassifierWrapper,
        classifier_target,
        classifier_scaling,
        xzeroprediction,
        precision,
        guidance_stabilization=None,
        gradient_normalization=False,
    ):
        self.input_shape = input_shape
        self.precision = precision
        self.classifier_wrapper = classifier_wrapper
        self.classifier_target = classifier_target.to(classifier_wrapper.label_dtype)
        self.classifier_scaling = classifier_scaling
        self.classifier_loss_fn = self.classifier_wrapper.loss_fn
        self.guidance_stabilization = guidance_stabilization
        self.xzeroprediction = xzeroprediction
        self.gradient_normalization = gradient_normalization
        print("Setting xzeropred to:", self.xzeroprediction)
        print("Setting gradient_normalization to:", self.gradient_normalization)
        print("Setting guidance_stabilization to:", self.guidance_stabilization)

    def __call__(self, xt, xzeropred):
        return self.get_diffusion_guidance(xt, xzeropred)

    @torch.enable_grad()
    def get_diffusion_guidance(self, xt, xzeropred):
        # This method is drastically impacted by precision errors at the torch.autograd.grad call
        # This can lead to non-deterministic results.

        if self.xzeroprediction:
            assert xzeropred.min() >= -1 and xzeropred.max() <= 1
            logits = self.classifier_wrapper.classifier(xzeropred.to(self.classifier_wrapper.precision))
        else:
            logits = self.classifier_wrapper.classifier(xt.requires_grad_(True).to(self.classifier_wrapper.precision))

        if self.classifier_wrapper.classifier_type == "multiclass":
            classifier_loss = logits[range(len(self.classifier_target)), self.classifier_target].sum()
        else:
            classifier_loss = -self.classifier_loss_fn(logits.squeeze(), self.classifier_target)

        classifier_grad = torch.autograd.grad(classifier_loss, xt)[0].detach()

        if self.guidance_stabilization is not None:
            classifier_grad = self.guidance_stabilization(classifier_grad.to(self.classifier_wrapper.precision))

        return classifier_grad

    # copy constructor; overwrites all parameters with new values if given, otherwise keeps the old values
    def copy(self, **kwargs):
        return DiffusionGuidance(
            input_shape=kwargs.get("input_shape", self.input_shape),
            classifier_wrapper=kwargs.get("classifier_wrapper", self.classifier_wrapper),
            classifier_target=kwargs.get("classifier_target", self.classifier_target),
            classifier_scaling=kwargs.get("classifier_scaling", self.classifier_scaling),
            adam_step_size=kwargs.get("adam_step_size", self.adam_step_size),
            ema=kwargs.get("ema", self.ema),
        )


class EMA:
    def __init__(self, input_shape, device, beta, bias_correction):
        self.history = torch.zeros(input_shape).to(device)
        self.step = 1
        self.beta = beta
        self.bias_correction = bias_correction

    def __call__(self, g):
        g_ema = self.beta * self.history + (1 - self.beta) * g
        self.history = g_ema
        if self.bias_correction:
            g_ema.div_(1 - (self.beta**self.step))
        self.step += 1
        return g_ema


class ADAMGradientStabilization:
    def __init__(self, input_shape, device, beta_1=0.9, beta_2=0.999, eps=1e-4, bias_correction=False):
        super().__init__()
        self.firstmoment = EMA(input_shape=input_shape, device=device, beta=beta_1, bias_correction=bias_correction)
        self.secondmoment = EMA(input_shape=input_shape, device=device, beta=beta_2, bias_correction=bias_correction)
        self.eps = eps

    def __call__(self, g):
        vel = self.firstmoment(g)
        acc = self.secondmoment(g**2)
        x = vel / (torch.sqrt(acc) + self.eps)
        return x
