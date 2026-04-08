# Neural Machine Translation with Transformer (Keras)

This project recreates the Keras English-to-Spanish neural machine translation example as a runnable local project.

## What this includes

- Transformer encoder-decoder model in Keras 3
- Text preprocessing with `TextVectorization`
- Auto-download of the Anki English-Spanish dataset
- Training script that saves model + vocab artifacts
- Inference script for translating new English sentences

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:

   `pip install -r requirements.txt`

## Train

Run:

`python train.py --epochs 1`

Faster sanity-check run:

`python train.py --epochs 1 --limit-pairs 20000`

Notes:

- 1 epoch is quick for sanity checking.
- Use 30+ epochs for meaningful translations.
- Artifacts are saved to `artifacts/`.

## Translate

After training:

`python translate.py --sentence "I love to write."`

Browser UI (select model + test in browser):

`python web_translate.py --open-browser`

This starts a local Flask server, tries to open Chrome automatically, and lets you choose any `.keras` model in `artifacts/` from a dropdown.

Desktop GUI (no browser, pure Python):

`python gui_translate.py`

This opens a native desktop window where you can select a `.keras` model and test translations directly.

Example output format:

- Input: I love to write.
- Translation: [start] me encanta escribir [end]

## Run on Iowa State HPC (Slurm)

This repo includes an HPC submission script and optional dataset pre-download script:

- `scripts/train_hpc.slurm`
- `scripts/prepare_dataset.py`

Recommended workflow:

1. On a login node, download and extract dataset once:

   `python scripts/prepare_dataset.py --output-dir data/spa-eng`

2. Edit `scripts/train_hpc.slurm` and set your partition/account/module names for Iowa State HPC.

3. Submit the job:

   `sbatch scripts/train_hpc.slurm`

Or submit + auto-follow logs + receive email notifications:

`bash scripts/submit_and_watch_hpc.sh --email yournetid@iastate.edu --account s2026.se.4390.01 --partition instruction --epochs 30`

When HPC training finishes, `scripts/train_hpc.slurm` now auto-pushes artifacts (including model) to Git by default.
Model files are saved with epoch/date/time in the filename, for example:

`artifacts/transformer_model_epoch30_20260408_211455.keras`

Notes:

- The Slurm script uses `--dataset-file data/spa-eng/spa.txt` so training does not depend on internet access from compute nodes.
- Logs are written to `logs/`.
- If your cluster requires CUDA modules for TensorFlow GPU usage, uncomment the CUDA module line in the Slurm script.
- Auto-push uses Git LFS for model files. Ensure `git-lfs` is available on the cluster account used for training jobs.

## Project structure

- `train.py` - train and save model artifacts
- `translate.py` - load saved model and run decoding
- `src/nmt_transformer/model.py` - Transformer layers and model assembly
- `src/nmt_transformer/data.py` - dataset download and tf.data pipeline
- `src/nmt_transformer/preprocessing.py` - vectorization and vocab serialization
- `src/nmt_transformer/config.py` - default hyperparameters
