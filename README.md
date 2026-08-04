# LBM Inference

This repository provides an inference pipeline for **LBM (Latent Bridge Matching)** models. It enables single-image inference using a pretrained checkpoint and an additional conditioning image (e.g., shading or normals) to generate the desired output.


## Setup 
First, clone the repository and move into the project directory:

```bash
git clone https://github.com/carmecorbi/LBM_RID.git
cd LBM_RID
```
Before running the inference code, create and activate a Python environment (Python 3.10 or later is recommended).

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

## Structure

```text
examples/
└── inference/
    └── inference_single_image.py
models/
└── lbm/
    ├── config.yaml
src/
└── lbm/
    ├── inference/
    ├── models/
    └── ...
```

## Inference
Run the inference script:

```bash
python -m examples.inference.inference_single_image 
    --image_path <input_image> \
    --shading_cond_dir <conditioning_image> \
    --model_dir <model_directory> \
    --checkpoint <checkpoint_path> \
    --output_root <output_directory>
```
### Required Arguments

| Argument             | Description                                               |
| -------------------- | --------------------------------------------------------- |
| `--image_path`       | Path to the source RGB image.                             |
| `--shading_cond_dir` | Path to the conditioning image.                           |
| `--model_dir`        | Directory containing the model config. (`config.yaml`).   |
| `--checkpoint`       | Path to the model checkpoint (.ckpt).                     |
| `--output_root`      | Directory where the generated image will be saved.        |
| `--inference_steps`  | Number of inference/sampling steps (default: `1`).        |

## License

See the `LICENSE` file for licensing information.
