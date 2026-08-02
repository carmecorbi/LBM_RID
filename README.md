# LBM Inference

This repository provides an inference pipeline for **LBM (Latent Bridge Matching)** models. It enables single-image inference using a pretrained checkpoint and an additional conditioning image (e.g., shading or normals) to generate the desired output.


## Setup 

To be up and running, you need first to create a virtual env with at least python3.10 installed and activate it

With venv
```bash
python3.10 -m venv envs/lbm-rid
source envs/lbm-rid/bin/activate
```
With conda
```
conda create -n lbm-rid python=3.10
conda activate lbm-rid
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Repository Structure

```text
examples/
└── inference/
    └── inference_single_image.py

src/
└── lbm/
    ├── inference/
    ├── models/
    └── ...
```

## Running Inference

Execute the inference script as follows:

```bash
python examples/inference/inference_single_image.py \
    --image_path <input_image> \
    --shading_cond_dir <conditioning_image> \
    --model_dir <model_directory> \
    --checkpoint <checkpoint_path> \
    --output_root <output_directory>
```

### Example

```bash
python examples/inference/inference_single_image.py \
    --image_path assets/input.png \
    --shading_cond_dir assets/shading.png \
    --model_dir checkpoints/model \
    --checkpoint checkpoints/model.ckpt \
    --output_root outputs
```

## Command Line Arguments

| Argument             | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `--image_path`       | Path to the source RGB image.                             |
| `--shading_cond_dir` | Path to the conditioning image used during inference.     |
| `--model_dir`        | Directory containing the model configuration and weights. |
| `--checkpoint`       | Path to the model checkpoint (.ckpt).                     |
| `--output_root`      | Directory where the generated image will be saved.        |
| `--inference_steps`  | Number of inference/sampling steps (default: `1`).        |

## Input

The inference script expects:

* A source RGB image.
* A conditioning image (for example, shading or surface normals).
* A pretrained LBM model.

## Output

The generated image is stored in the specified output directory using the following naming convention:

```text
<input_filename>_output.png
```
## License

See the `LICENSE` file for licensing information.
