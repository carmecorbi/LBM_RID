# LBM Inference

This repository provides an inference pipeline for **LBM (Latent Bridge Matching)** models. It enables single-image inference using a pretrained checkpoint and an additional conditioning image (e.g., shading or normals) to generate the desired output.


## Setup 
before running the inference code, create and activate a Python environment (Python 3.10 or later is recommended). 

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

Install dependencies
Once the environment is activated, install the required packages:

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
python -m examples.inference.inference_single_image 
    --image_path <input_image> \
    --shading_cond_dir <conditioning_image> \
    --model_dir <model_directory> \
    --checkpoint <checkpoint_path> \
    --output_root <output_directory>
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
