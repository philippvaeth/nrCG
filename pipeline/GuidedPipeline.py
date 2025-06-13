from os import error
import numpy as np
import PIL.Image
from diffusers.pipelines import ImagePipelineOutput, DiffusionPipeline, DDPMPipeline, DDIMPipeline
from diffusers.utils.torch_utils import randn_tensor
from diffusers.schedulers.scheduling_ddpm import DDPMSchedulerOutput
from diffusers.schedulers.scheduling_ddim import DDIMSchedulerOutput
from typing import List, Optional, Tuple, Union
import torch
import wandb

from pipeline.utils import GradientLogger, cosine_similarity, l2_vector_norm, l2_vector_norm_minclipped


class GuidedImagePipelineOutput(ImagePipelineOutput):

    def __init__(self, images, samples):
        super().__init__(images)
        self.samples = samples

    images: Union[List[PIL.Image.Image], np.ndarray]
    samples: torch.Tensor


class GuidedDDPMPipeline(DDPMPipeline):
    def __init__(self, unet, scheduler, guidance=None, logging=False):
        super().__init__(unet, scheduler)
        self.guidance = None
        self.precision = torch.float32
        self.gradientlogger = GradientLogger() if logging == True else None

    def set_guidance(self, guidance):
        self.guidance = guidance

    def set_precision(self, precision):
        self.unet = self.unet.to(precision)
        self.precision = precision

    def set_name(self, name: str):
        self.name = name

    def set_gradientlogging(self, logger):
        self.gradientlogger = logger
        self.logging = True

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        num_inference_steps: int = 1000,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        with_grad=False,
    ) -> GuidedImagePipelineOutput:
        r"""
        The call function to the pipeline for generation.

        Args:
            batch_size (`int`, *optional*, defaults to 1):
                The number of images to generate.
            generator (`torch.Generator`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            num_inference_steps (`int`, *optional*, defaults to 1000):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.ImagePipelineOutput`] instead of a plain tuple.

        Example:

        ```py
        >>> from diffusers import DDPMPipeline

        >>> # load model and scheduler
        >>> pipe = DDPMPipeline.from_pretrained("google/ddpm-cat-256")

        >>> # run pipeline in inference (sample random noise and denoise)
        >>> image = pipe().images[0]

        >>> # save image
        >>> image.save("ddpm_generated_image.png")
        ```

        Returns:
            [`~pipelines.ImagePipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.ImagePipelineOutput`] is returned, otherwise a `tuple` is
                returned where the first element is a list with the generated images
        """
        if isinstance(self.unet.config.sample_size, int):
            image_shape = (
                batch_size,
                self.unet.config.in_channels,
                self.unet.config.sample_size,
                self.unet.config.sample_size,
            )
        else:
            image_shape = (batch_size, self.unet.config.in_channels, *self.unet.config.sample_size)

        if self.device.type == "mps":
            image = randn_tensor(image_shape, generator=generator, dtype=self.precision)
            image = image.to(self.device)
        else:
            image = randn_tensor(image_shape, generator=generator, device=self.device, dtype=self.precision)

        self.scheduler.set_timesteps(num_inference_steps)

        for t in self.progress_bar(self.scheduler.timesteps):
            with torch.set_grad_enabled(with_grad):
                ##! Added parameter
                if with_grad:
                    image = image.detach().requires_grad_(True)

                # 1. predict noise model_output
                model_output = self.unet(image, t).sample

                # 2. compute previous image: x_t -> x_t-1
                scheduler_result = self.scheduler.step(model_output, t, image, generator=generator)

                # Added scheduler guidance
                if self.guidance is not None:
                    guidance_grad = self.guidance(xt=image, xzeropred=scheduler_result.pred_original_sample)
                    with torch.no_grad():
                        # Variance given by the DDPM forward process
                        ddpm_variance_t = self.scheduler._get_variance(t, predicted_variance=None).to(image.device)

                        scaled_guidance_grad = (self.guidance.classifier_scaling * ddpm_variance_t * guidance_grad).to(
                            model_output.dtype
                        )

                        # Log cosine similarity
                        if self.gradientlogger is not None:
                            self.gradientlogger.grad_ratio_time(
                                scaled_guidance_grad, image, t, logging_name=f"{self.name}_grad_"
                            )

                        scheduler_result.prev_sample = scheduler_result.prev_sample + scaled_guidance_grad

                image = scheduler_result.prev_sample.to(image.dtype)

        image = (image / 2 + 0.5).clamp(0, 1)
        image = image.detach().cpu()
        raw_samples = (image.detach().clone() * 2) - 1

        image = image.permute(0, 2, 3, 1).numpy()
        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, raw_samples)

        return GuidedImagePipelineOutput(images=image, samples=raw_samples)
