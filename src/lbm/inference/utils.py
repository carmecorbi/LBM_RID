import logging
import os
from typing import List, Optional

import torch
import yaml
from diffusers import FlowMatchEulerDiscreteScheduler
from huggingface_hub import snapshot_download
from safetensors.torch import load_file

from src.lbm.models.embedders import (
    ConditionerWrapper,
    LatentsConcatEmbedder,
    LatentsConcatEmbedderConfig,
)
from src.lbm.models.lbm import LBMConfig, LBMModel
from src.lbm.models.unets import DiffusersUNet2DCondWrapper
from src.lbm.models.vae import AutoencoderKLDiffusers, AutoencoderKLDiffusersConfig


def get_model(
    model_dir: str,
    checkpoint_dir: str,
    save_dir: Optional[str] = None,
    torch_dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
) -> LBMModel:
    """
    Load model config from model_dir and weights from either:
      - a .safetensors inside model_dir
      - or an explicit checkpoint_dir (.ckpt)

    Args:
        model_dir (str): directory containing yaml config
        checkpoint_dir (str): path to .ckpt checkpoint
        save_dir (Optional[str]): optional HF cache dir
        torch_dtype (torch.dtype): dtype
        device (str): cuda/cpu

    Returns:
        LBMModel
    """

    # Download from HF if needed
    if not os.path.exists(model_dir):
        local_dir = snapshot_download(
            model_dir,
            local_dir=save_dir,
        )
        model_dir = local_dir

    if not os.path.exists(model_dir):
        raise ValueError(f"Model directory does not exist: {model_dir}")

    model_files = os.listdir(model_dir)

    # ------------------------------------------------------------------
    # YAML CONFIG
    # ------------------------------------------------------------------
    yaml_files = [f for f in model_files if f.endswith(".yaml")]

    if len(yaml_files) == 0:
        raise ValueError(
            f"No yaml config found in model directory: {model_dir}"
        )

    yaml_path = os.path.join(model_dir, yaml_files[0])

    with open(yaml_path, "r") as f:
        config = yaml.safe_load(f)

    logging.info(f"Loaded config: {yaml_path}")

    # ------------------------------------------------------------------
    # CREATE MODEL
    # ------------------------------------------------------------------
    model = _get_model_from_config(
        **config,
        torch_dtype=torch_dtype,
    )

    # ------------------------------------------------------------------
    # LOOK FOR SAFETENSORS
    # ------------------------------------------------------------------
    safetensors_files = sorted(
        [f for f in model_files if f.endswith(".safetensors")]
    )

    # ------------------------------------------------------------------
    # LOAD SAFETENSORS
    # ------------------------------------------------------------------
    if len(safetensors_files) > 0:

        safetensor_path = os.path.join(
            model_dir,
            safetensors_files[-1],
        )

        logging.info(f"Loading safetensors: {safetensor_path}")

        sd = load_file(safetensor_path)

        model.load_state_dict(
            sd,
            strict=True,
        )

    # ------------------------------------------------------------------
    # LOAD CKPT
    # ------------------------------------------------------------------
    else:

        if checkpoint_dir is None:
            raise ValueError(
                "No safetensors found and checkpoint_dir is None"
            )

        if not os.path.exists(checkpoint_dir):
            raise ValueError(
                f"Checkpoint does not exist: {checkpoint_dir}"
            )

        logging.info(f"Loading ckpt: {checkpoint_dir}")

        raw = torch.load(
            checkpoint_dir,
            map_location=device,
            weights_only=False,
        )

        # Lightning checkpoints
        if isinstance(raw, dict) and "state_dict" in raw:
            sd = raw["state_dict"]
        else:
            sd = raw

        # --------------------------------------------------------------
        # REMAP KEYS
        # --------------------------------------------------------------
        def remap_keys(sd):

            new_sd = {}

            for k, v in sd.items():

                k = k.replace("model.", "")

                k = k.replace(
                    "vae.vae_encoder",
                    "vae.vae_model.encoder",
                )

                k = k.replace(
                    "vae.vae_decoder",
                    "vae.vae_model.decoder",
                )

                k = k.replace(
                    "vae.vae_quant_conv",
                    "vae.vae_model.quant_conv",
                )

                k = k.replace(
                    "vae.vae_post_quant_conv",
                    "vae.vae_model.post_quant_conv",
                )

                if "lpips_loss" in k:
                    k = k.replace(".1.", ".model.1.")

                if k == "lpips_loss.lins.model.1.1.weight":
                    k = "lpips_loss.lins.1.model.1.weight"

                new_sd[k] = v

            return new_sd

        sd = remap_keys(sd)
        model.load_state_dict(
            sd,
            strict=True,
        )

    # ------------------------------------------------------------------
    # DEVICE
    # ------------------------------------------------------------------
    model.to(device)
    model.to(torch_dtype)

    model.eval()

    logging.info("Model loaded successfully")

    return model


def _get_model_from_config(
    backbone_signature: str = "stabilityai/stable-diffusion-xl-base-1.0",
    vae_num_channels: int = 4,
    unet_input_channels: int = 4,
    timestep_sampling: str = "log_normal",
    selected_timesteps: Optional[List[float]] = None,
    prob: Optional[List[float]] = None,
    conditioning_images_keys: Optional[List[str]] = None,
    conditioning_masks_keys: Optional[List[str]] = None,
    source_key: str = "source_image",
    target_key: str = "source_image_paste",
    bridge_noise_sigma: float = 0.0,
    logit_mean: float = 0.0,
    logit_std: float = 1.0,
    pixel_loss_type: str = "lpips",
    latent_loss_type: str = "l2",
    latent_loss_weight: float = 1.0,
    pixel_loss_weight: float = 0.0,
    torch_dtype: torch.dtype = torch.bfloat16,
    transformer_layers_per_block: Optional[List[int]] = None,
    **kwargs,
):

    conditioning_images_keys = conditioning_images_keys or []
    conditioning_masks_keys = conditioning_masks_keys or []

    conditioners = []

    denoiser = DiffusersUNet2DCondWrapper(
        in_channels=unet_input_channels,  # Add downsampled_image
        out_channels=vae_num_channels,
        center_input_sample=False,
        flip_sin_to_cos=True,
        freq_shift=0,
        down_block_types=[
            "DownBlock2D",
            "CrossAttnDownBlock2D",
            "CrossAttnDownBlock2D",
        ],
        mid_block_type="UNetMidBlock2DCrossAttn",
        up_block_types=["CrossAttnUpBlock2D", "CrossAttnUpBlock2D", "UpBlock2D"],
        only_cross_attention=False,
        block_out_channels=[320, 640, 1280],
        layers_per_block=2,
        downsample_padding=1,
        mid_block_scale_factor=1,
        dropout=0.0,
        act_fn="silu",
        norm_num_groups=32,
        norm_eps=1e-05,
        cross_attention_dim=[320, 640, 1280],
        transformer_layers_per_block=transformer_layers_per_block,
        reverse_transformer_layers_per_block=None,
        encoder_hid_dim=None,
        encoder_hid_dim_type=None,
        attention_head_dim=[5, 10, 20],
        num_attention_heads=None,
        dual_cross_attention=False,
        use_linear_projection=True,
        class_embed_type=None,
        addition_embed_type=None,
        addition_time_embed_dim=None,
        num_class_embeds=None,
        upcast_attention=None,
        resnet_time_scale_shift="default",
        resnet_skip_time_act=False,
        resnet_out_scale_factor=1.0,
        time_embedding_type="positional",
        time_embedding_dim=None,
        time_embedding_act_fn=None,
        timestep_post_act=None,
        time_cond_proj_dim=None,
        conv_in_kernel=3,
        conv_out_kernel=3,
        projection_class_embeddings_input_dim=None,
        attention_type="default",
        class_embeddings_concat=False,
        mid_block_only_cross_attention=None,
        cross_attention_norm=None,
        addition_embed_type_num_heads=64,
    ).to(torch_dtype)


    if conditioning_images_keys or conditioning_masks_keys:

        latents_concat_embedder_config = LatentsConcatEmbedderConfig(
            image_keys=conditioning_images_keys,
            mask_keys=conditioning_masks_keys,
        )
        latent_concat_embedder = LatentsConcatEmbedder(latents_concat_embedder_config)
        latent_concat_embedder.freeze()
        conditioners.append(latent_concat_embedder)

        # Wrap conditioners and set to device
    conditioner = ConditionerWrapper(
        conditioners=conditioners,
    )

    ## VAE ##
    # Get VAE model
    vae_config = AutoencoderKLDiffusersConfig(
        version=backbone_signature,
        subfolder="vae",
        tiling_size=(128, 128),
    )
    vae = AutoencoderKLDiffusers(vae_config).to(torch_dtype)
    vae.freeze()
    vae.to(torch_dtype)

    ## Diffusion Model ##
    # Get diffusion model
    config = LBMConfig(
        source_key=source_key,
        target_key=target_key,
        latent_loss_weight=latent_loss_weight,
        latent_loss_type=latent_loss_type,
        pixel_loss_type=pixel_loss_type,
        pixel_loss_weight=pixel_loss_weight,
        timestep_sampling=timestep_sampling,
        logit_mean=logit_mean,
        logit_std=logit_std,
        selected_timesteps=selected_timesteps,
        prob=prob,
        bridge_noise_sigma=bridge_noise_sigma,
    )

    sampling_noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        backbone_signature,
        subfolder="scheduler",
    )

    model = LBMModel(
        config,
        denoiser=denoiser,
        sampling_noise_scheduler=sampling_noise_scheduler,
        vae=vae,
        conditioner=conditioner,
    ).to(torch_dtype)

    return model