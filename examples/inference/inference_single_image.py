import os
import argparse
import logging
import torch

from PIL import Image
from src.lbm.inference import evaluate, get_model


def main(args):

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s"
    )

    logging.info("Loading model...")

    model = get_model(
        args.model_dir,
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device="cuda"
    )

    logging.info("Model loaded.")

    source_image = Image.open(
        args.image_path
    ).convert("RGB")

    shading_cond_image = Image.open(
        args.shading_cond_dir
    ).convert("RGB")

    output_pil = evaluate(
        model,
        source_image,
        shading_cond_image,
        args.inference_steps
    )

    base_name = os.path.splitext(
        os.path.basename(args.image_path)
    )[0]

    os.makedirs(
        args.output_root,
        exist_ok=True
    )

    png_path = os.path.join(
        args.output_root,
        f"{base_name}_output.png"
    )

    output_pil.save(png_path)
    logging.info(f"Saved PNG: {png_path}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Run inference using a pretrained LBM model with a source image and a shading condition image."
    )

    parser.add_argument(
        "--image_path",
        required=True,
        help="Path to the input RGB image."
    )

    parser.add_argument(
        "--model_dir",
        required=True,
        help="Path to the directory containing the pretrained model."
    )

    parser.add_argument(
        "--shading_cond_dir",
        required=True,
        help="Path to the RGB shading condition image."
    )

    parser.add_argument(
        "--checkpoint",
        required=True,
        help="Checkpoint file to load from the model directory."
    )

    parser.add_argument(
        "--output_root",
        required=True,
        help="Directory where the output image will be saved."
    )

    parser.add_argument(
        "--inference_steps",
        type=int,
        default=1,
        help="Number of inference steps to perform (default: 1)."
    )

    args = parser.parse_args()

    main(args)