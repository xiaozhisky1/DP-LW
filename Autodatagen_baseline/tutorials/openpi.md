## Pi0

This tutorial provides a simplest version instruction to finetune Pi0 on the 2025 BEHAVIOR-1K Challenge dataset. 

### Repo Clone

```
git clone https://github.com/Franc1sNing/Autodatagen_baseline.git
git clone https://github.com/StanfordVL/BEHAVIOR-1K.git
```
This finetuning instruction is adapted from the original [openpi repo](https://github.com/Physical-Intelligence/openpi).


### Installation

Openpi use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```
cd baselines/openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .

source .venv/bin/activate

# Install behavior for server deploy 
cd $PATH_TO_BEHAVIOR_1K
uv pip install -e bddl
uv pip install -e OmniGibson[eval]
```

### Finetune OpenPi

- picking_up_trash task [here](https://drive.google.com/file/d/1G_ACu3uUP_9RmXDgqa7307aFt28G-vJN/view?usp=sharing). This checkpoint has been trained for 50k steps.

If you want to train your own model, you should add your training cfg in 

'''
b1k-baselines/baselines/openpi/src/openpi/training/config.py
'''

We already have like pi0_x7s_full, pi05_x7s

If you would like to run eval only feel free to skip to the last section. 

Before we can run training, we need to compute the normalization statistics for the training data. Change line 98 of `compute_norm_stats.py` to be the task name you want (or None to include all tasks), then run the script below

```
uv run scripts/compute_norm_stats.py --config-name pi0_x7s_full
```
This will create `norm_stats.json` under `assets/pi0_x7s_full/


After this, change line 137 of `data_loader.py` to be the task name you want (or None to include all tasks), then run the following command to fintune OpenPi:




```
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train_val.py pi0_x7s_full     \
--exp_name="pi0_fridge_full_20260306_163848"     \
--resume    \
--batch_size=64     \
--num_train_steps=300000
```


### Evaluation

After finetuning, you can run evaluation by following the steps below:

1. Deploy finetuned checkpoint:

    ```
    source .venv/bin/activate
    CUDA_VISIBLE_DEVICES=0 uv run scripts/serve_x7s.py --config pi0_x7s_full \
    --checkpoint_dir /data/Packages/openpi_checkpoints/pi0_x7s_full/pi0_fridge_full_20260306_163848/265000/
    ```
    This opens a connection listening on 0.0.0.0:8000.


2. Run the evaluation on BEHAVIOR:

    Assume you have behavior env installed (check https://github.com/StanfordVL/BEHAVIOR-1K for more details), run the following command within the BEHAVIOR-1K directory:
    ```
    python eval_pipeline.py --task_config openfridge  --enable_cameras --server_url  http://172.16.100.18:8000/infer
    ```
