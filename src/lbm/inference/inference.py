import logging
from typing import Optional, Union

import PIL
import torch
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor

from src.lbm.models.lbm import LBMModel


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@torch.no_grad()
def evaluate(
    model: LBMModel,
    source_image: Union[PIL.Image.Image, torch.Tensor],
    shading_image: Optional[Union[PIL.Image.Image, torch.Tensor]] = None,
    num_sampling_steps: int = 1,
):
    """
    Generates an output image from a source image using the pretrained LBM model.

    Args:
        model (LBMModel):
            Pretrained LBM model used for inference.
        source_image (PIL.Image.Image or torch.Tensor):
            Input RGB image. If a tensor is provided, it must have shape [C, H, W]
            and be normalized either to [0, 1] or [-1, 1].
        shading_image (PIL.Image.Image or torch.Tensor, optional):
            Conditioning image used during inference. It must have the same spatial
            dimensions as the source image.
        num_sampling_steps (int, optional):
            Number of sampling steps performed by the model. Defaults to 1.

    Returns:
        PIL.Image.Image or torch.Tensor:
            A PIL image if the input is a PIL image, otherwise a tensor.
    """

    # ------------------------------------------------------------------
    # Source image
    # ------------------------------------------------------------------
    if isinstance(source_image, Image.Image):
        ori_w, ori_h = source_image.size

        img_tensor = ToTensor()(source_image).unsqueeze(0)
        img_tensor = img_tensor * 2 - 1
        img_tensor = img_tensor.cuda().to(torch.bfloat16)

    elif isinstance(source_image, torch.Tensor):
        if source_image.ndim != 3:
            raise ValueError("source_image tensor must have shape [C, H, W].")

        if source_image.max() <= 1.0 and source_image.min() >= 0.0:
            img_tensor = source_image * 2 - 1
        else:
            img_tensor = source_image

        img_tensor = img_tensor.unsqueeze(0).cuda().to(torch.bfloat16)
        logger.info("Source tensor shape: %s", img_tensor.shape)

    else:
        raise TypeError(
            "source_image must be either a PIL.Image.Image or a torch.Tensor."
        )

    batch = {"source_image": img_tensor}

    # ------------------------------------------------------------------
    # Shading conditioning image
    # ------------------------------------------------------------------
    if shading_image is not None:

        if isinstance(shading_image, Image.Image):
            shading_tensor = ToTensor()(shading_image).unsqueeze(0)
            shading_tensor = shading_tensor * 2 - 1
            shading_tensor = shading_tensor.cuda().to(torch.bfloat16)

        elif isinstance(shading_image, torch.Tensor):
            if shading_image.ndim != 3:
                raise ValueError("shading_image tensor must have shape [C, H, W].")

            if shading_image.max() <= 1.0 and shading_image.min() >= 0.0:
                shading_tensor = shading_image * 2 - 1
            else:
                shading_tensor = shading_image

            shading_tensor = shading_tensor.unsqueeze(0).cuda().to(torch.bfloat16)

        else:
            raise TypeError(
                "shading_image must be either a PIL.Image.Image or a torch.Tensor."
            )

        batch["albedo"] = shading_tensor

    # ------------------------------------------------------------------
    # Encode source image and generate output
    # ------------------------------------------------------------------
    z_source = model.vae.encode(batch[model.source_key])

    output_image = model.sample(
        z=z_source,
        num_steps=num_sampling_steps,
        conditioner_inputs=batch,
        max_samples=1,
    ).clamp(-1, 1)

    # ------------------------------------------------------------------
    # Convert back to PIL if necessary
    # ------------------------------------------------------------------
    if isinstance(source_image, Image.Image):
        output_image = (output_image[0].float().cpu() + 1) / 2
        output_image = ToPILImage()(output_image)
        output_image = output_image.resize((ori_w, ori_h))

    return output_image