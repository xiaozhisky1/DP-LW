"""
LeRobot v3 local dataset adapter for diffusion_policy training.

Loads images, states, and actions from a LeRobot dataset directory and
converts them into the ReplayBuffer + SequenceSampler format expected by the
standard diffusion_policy training pipeline.

Usage in task YAML::

    dataset:
      _target_: diffusion_policy.dataset.lerobot_replay_image_dataset.LeRobotReplayImageDataset
      shape_meta: ${shape_meta}
      data_path: /path/to/lerobot/data
      horizon: ${horizon}
      pad_before: ${eval:'${n_obs_steps}-1+${n_latency_steps}'}
      pad_after:  ${eval:'${n_action_steps}-1'}
      n_obs_steps: ${dataset_obs_steps}
      repo_id: "local/openfridge"
      action_key: action
      state_key: observation.state
      camera_key_map:
        first_person_image: observation.images.first_person_camera
        left_hand_image:    observation.images.left_hand_camera
        right_hand_image:   observation.images.right_hand_camera
      lowdim_key_map:
        agent_pos: observation.state
      use_cache: True
      seed: 42
      val_ratio: 0.02
"""

from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np
import torch
import zarr
from filelock import FileLock
from tqdm import tqdm

from diffusion_policy.common.normalize_util import (
    array_to_stats,
    get_image_range_normalizer,
    get_range_normalizer_from_stat,
)
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.replay_buffer import ReplayBuffer
from diffusion_policy.common.sampler import SequenceSampler, downsample_mask, get_val_mask
from diffusion_policy.dataset.base_dataset import BaseImageDataset
from diffusion_policy.model.common.normalizer import LinearNormalizer

try:
    from diffusion_policy.codecs.imagecodecs_numcodecs import Jpeg2k, register_codecs

    register_codecs()
    _IMG_COMPRESSOR = Jpeg2k(level=50)
except Exception:
    import numcodecs as _nc

    _IMG_COMPRESSOR = _nc.Blosc(cname="lz4", clevel=5, shuffle=_nc.Blosc.BITSHUFFLE)


# ---------------------------------------------------------------------------
# LeRobot v3 compatibility helpers (adapted from lerobot_act_dataset.py)
# ---------------------------------------------------------------------------

def _read_fps(root: str) -> int:
    info_path = Path(root) / "meta" / "info.json"
    if not info_path.exists():
        return 30
    with info_path.open("r", encoding="utf-8") as f:
        info = json.load(f)
    return int(info.get("fps", 30))


def _ensure_local_meta_jsonl(root: str) -> None:
    """Create minimal JSONL metadata files required by the lerobot loader."""
    root_path = Path(root)
    meta_dir = root_path / "meta"
    info_path = meta_dir / "info.json"
    if not info_path.exists():
        return
    try:
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
    except Exception:
        return

    total_episodes = int(info.get("total_episodes", 1))
    tasks_jsonl = meta_dir / "tasks.jsonl"
    episodes_jsonl = meta_dir / "episodes.jsonl"
    episodes_stats_jsonl = meta_dir / "episodes_stats.jsonl"

    if (not tasks_jsonl.exists()) or tasks_jsonl.stat().st_size == 0:
        task_text = "task"
        tasks_parquet = meta_dir / "tasks.parquet"
        try:
            import pandas as pd
            if tasks_parquet.exists():
                df = pd.read_parquet(tasks_parquet)
                if len(df.index) > 0 and isinstance(df.index[0], str):
                    task_text = str(df.index[0])
        except Exception:
            pass
        with tasks_jsonl.open("w", encoding="utf-8") as f:
            f.write(json.dumps({"task_index": 0, "task": task_text}) + "\n")

    if (not episodes_jsonl.exists()) or episodes_jsonl.stat().st_size == 0:
        wrote_from_parquet = False
        episodes_parquet_files = sorted(
            (meta_dir / "episodes").glob("chunk-*/*.parquet")
        )
        if episodes_parquet_files:
            try:
                import pandas as pd
                episodes_df = pd.concat(
                    [pd.read_parquet(p) for p in episodes_parquet_files],
                    ignore_index=True,
                )
                episodes_df = episodes_df.sort_values("episode_index")
                with episodes_jsonl.open("w", encoding="utf-8") as f:
                    for _, row in episodes_df.iterrows():
                        tasks = row.get("tasks", ["task"])
                        if hasattr(tasks, "tolist"):
                            tasks = tasks.tolist()
                        if isinstance(tasks, str):
                            tasks = [tasks]
                        f.write(
                            json.dumps(
                                {
                                    "episode_index": int(row["episode_index"]),
                                    "tasks": [str(x) for x in tasks],
                                    "length": int(row.get("length", 0)),
                                }
                            )
                            + "\n"
                        )
                wrote_from_parquet = True
            except Exception:
                wrote_from_parquet = False

        if not wrote_from_parquet:
            with episodes_jsonl.open("w", encoding="utf-8") as f:
                for ep_idx in range(total_episodes):
                    f.write(
                        json.dumps(
                            {"episode_index": ep_idx, "tasks": ["task"], "length": 1}
                        )
                        + "\n"
                    )

    if (not episodes_stats_jsonl.exists()) or episodes_stats_jsonl.stat().st_size == 0:
        with episodes_stats_jsonl.open("w", encoding="utf-8") as f:
            for ep_idx in range(total_episodes):
                f.write(json.dumps({"episode_index": ep_idx, "stats": {}}) + "\n")


def _patch_lerobot_offline_fallback() -> None:
    """Force lerobot to stay local and skip HuggingFace Hub lookups."""
    try:
        import lerobot.datasets.lerobot_dataset as _ld
        import lerobot.datasets.utils as _utils
    except Exception:
        return

    if getattr(_utils, "_dp_offline_patched", False):
        return

    def _local_safe_version(_repo_id: str, version: str):
        v = str(version)
        return v if v.startswith("v") else f"v{v}"

    _utils.get_safe_version = _local_safe_version
    _ld.get_safe_version = _local_safe_version
    _utils._dp_offline_patched = True


def _patch_hf_list_feature_alias() -> None:
    """Allow 'List' feature type in older parquet files."""
    try:
        from datasets.features.features import Sequence, _FEATURE_TYPES
    except Exception:
        return
    if "List" not in _FEATURE_TYPES:
        _FEATURE_TYPES["List"] = Sequence


def _apply_lerobot_compat(root: str) -> None:
    """Apply all compatibility patches for local LeRobot v3 datasets."""
    _ensure_local_meta_jsonl(root)
    _patch_lerobot_offline_fallback()
    _patch_hf_list_feature_alias()


# ---------------------------------------------------------------------------
# Image utilities
# ---------------------------------------------------------------------------

def _to_hwc_uint8(img) -> np.ndarray:
    """Convert a LeRobot image (tensor or array, any layout) to uint8 HWC numpy."""
    if isinstance(img, torch.Tensor):
        img = img.numpy()
    img = np.asarray(img)

    # Remove leading time dimension if present (TCHW or THWC -> CHW or HWC)
    if img.ndim == 4:
        img = img[0]

    # Detect CHW: first dim is 1, 3, or 4
    if img.ndim == 3 and img.shape[0] in (1, 3, 4):
        img = np.moveaxis(img, 0, -1)  # CHW -> HWC

    # Convert float [0,1] -> uint8 [0,255]
    if not np.issubdtype(img.dtype, np.unsignedinteger):
        if np.issubdtype(img.dtype, np.floating):
            img = (img.clip(0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img = img.astype(np.uint8)

    return img


# ---------------------------------------------------------------------------
# Conversion: LeRobot dataset -> ReplayBuffer
# ---------------------------------------------------------------------------

def _convert_lerobot_to_replay(
    store,
    shape_meta: dict,
    data_path: str,
    repo_id: str,
    action_key: str,
    state_key: str,
    rgb_keys: list,
    lowdim_keys: list,
    camera_key_map: Dict[str, str],
    lowdim_key_map: Dict[str, str],
) -> ReplayBuffer:
    """
    Iterate over a local LeRobot dataset and store each frame into a zarr
    store compatible with ReplayBuffer / SequenceSampler.

    Images are resized to the resolution specified in *shape_meta* and stored
    as uint8 in HWC format.  Lowdim observations and actions are stored as
    float32.
    """
    from lerobot.datasets.lerobot_dataset import LeRobotDataset  # type: ignore

    _apply_lerobot_compat(data_path)

    # Load dataset without delta_timestamps so each index returns one frame.
    lerobot_ds = LeRobotDataset(repo_id, root=data_path, video_backend="pyav")
    n_steps = len(lerobot_ds)

    obs_shape_meta = shape_meta["obs"]
    action_shape = tuple(shape_meta["action"]["shape"])

    # Determine image target shapes (HWC for storage)
    img_target_shapes: Dict[str, tuple] = {}
    for key in rgb_keys:
        c, h, w = tuple(obs_shape_meta[key]["shape"])
        img_target_shapes[key] = (h, w, c)

    # Determine episode boundaries from episode_index
    episode_indices = [int(x) for x in lerobot_ds.hf_dataset["episode_index"]]
    episode_ends: list = []
    for i in range(1, n_steps):
        if episode_indices[i] != episode_indices[i - 1]:
            episode_ends.append(i)
    episode_ends.append(n_steps)

    # Build zarr structure
    root_grp = zarr.group(store)
    data_grp = root_grp.require_group("data", overwrite=True)
    meta_grp = root_grp.require_group("meta", overwrite=True)
    meta_grp.array(
        "episode_ends",
        np.array(episode_ends, dtype=np.int64),
        dtype=np.int64,
        compressor=None,
        overwrite=True,
    )

    # Pre-allocate action array
    action_arr = data_grp.zeros(
        name="action",
        shape=(n_steps,) + action_shape,
        dtype=np.float32,
        compressor=None,
        overwrite=True,
    )

    # Pre-allocate lowdim arrays
    lowdim_arrs: Dict[str, zarr.Array] = {}
    for key in lowdim_keys:
        lowdim_shape = tuple(obs_shape_meta[key]["shape"])
        lowdim_arrs[key] = data_grp.zeros(
            name=key,
            shape=(n_steps,) + lowdim_shape,
            dtype=np.float32,
            compressor=None,
            overwrite=True,
        )

    # Pre-allocate image arrays (one chunk per frame for efficient random access)
    img_arrs: Dict[str, zarr.Array] = {}
    for key in rgb_keys:
        h, w, c = img_target_shapes[key]
        img_arrs[key] = data_grp.zeros(
            name=key,
            shape=(n_steps, h, w, c),
            chunks=(1, h, w, c),
            dtype=np.uint8,
            compressor=_IMG_COMPRESSOR,
            overwrite=True,
        )

    # Fill arrays frame by frame
    for i in tqdm(range(n_steps), desc="Converting LeRobot dataset to ReplayBuffer"):
        sample = lerobot_ds[i]

        # -- Action (single-step) --
        action = np.asarray(sample[action_key], dtype=np.float32)
        if action.ndim == 2:
            action = action[0]
        action_arr[i] = action[: action_shape[0]]

        # -- Lowdim observations --
        for key in lowdim_keys:
            lerobot_key = lowdim_key_map[key]
            state = np.asarray(sample[lerobot_key], dtype=np.float32)
            if state.ndim == 2:
                state = state[0]
            lowdim_shape = tuple(obs_shape_meta[key]["shape"])
            lowdim_arrs[key][i] = state[: lowdim_shape[0]]

        # -- RGB images --
        for key in rgb_keys:
            lerobot_key = camera_key_map[key]
            if lerobot_key not in sample:
                raise KeyError(
                    f"Camera key '{lerobot_key}' not found in LeRobot sample. "
                    f"Available keys: {[k for k in sample.keys()]}"
                )
            img = _to_hwc_uint8(sample[lerobot_key])
            h, w, c = img_target_shapes[key]
            if img.shape[:2] != (h, w):
                img = cv2.resize(img, (w, h))
            img_arrs[key][i] = img

    return ReplayBuffer(root_grp)


# ---------------------------------------------------------------------------
# Dataset class
# ---------------------------------------------------------------------------

class LeRobotReplayImageDataset(BaseImageDataset):
    """
    Adapts a local LeRobot v3 dataset for diffusion_policy training.

    Data is converted once into an in-memory (or disk-cached) zarr
    ReplayBuffer and then sampled with SequenceSampler, matching the same
    interface used by RobomimicReplayImageDataset and MujocoImageDataset.

    Parameters
    ----------
    shape_meta : dict
        Observation / action shape metadata (same format as task YAML).
        RGB keys must have ``type: rgb``; proprioceptive keys default to
        ``type: low_dim``.
    data_path : str
        Root directory of the local LeRobot dataset (contains ``meta/``).
    horizon : int
        Total trajectory length returned per sample.
    pad_before / pad_after : int
        Number of timesteps to pad at episode boundaries.
    n_obs_steps : int or None
        Only return the first *n_obs_steps* observation frames to save RAM.
    seed : int
        RNG seed for train/val split.
    val_ratio : float
        Fraction of episodes reserved for validation.
    max_train_episodes : int or None
        Subsample training episodes to this count.
    repo_id : str
        Logical repo-id passed to ``LeRobotDataset`` (e.g. ``"local/openfridge"``).
    action_key : str
        LeRobot key for the action (default ``"action"``).
    state_key : str
        LeRobot key for proprioceptive state (default ``"observation.state"``).
        Used as the default target for all ``low_dim`` entries in *shape_meta*
        when *lowdim_key_map* is not given.
    camera_key_map : dict or None
        Mapping from shape_meta camera key → LeRobot dataset key.
        E.g. ``{first_person_image: observation.images.first_person_camera}``.
        If ``None`` and ``shape_meta`` contains exactly one RGB key, that key
        is assumed to match the LeRobot key directly.
    lowdim_key_map : dict or None
        Mapping from shape_meta lowdim key → LeRobot dataset key.
        Defaults to ``{key: state_key}`` for every ``low_dim`` key.
    use_cache : bool
        Cache the converted ReplayBuffer as a ``.zarr.zip`` file next to
        ``data_path`` for faster subsequent loads.
    """

    def __init__(
        self,
        shape_meta: dict,
        data_path: str,
        horizon: int = 1,
        pad_before: int = 0,
        pad_after: int = 0,
        n_obs_steps: Optional[int] = None,
        seed: int = 42,
        val_ratio: float = 0.02,
        max_train_episodes: Optional[int] = None,
        repo_id: str = "local/dataset",
        action_key: str = "action",
        state_key: str = "observation.state",
        camera_key_map: Optional[Dict[str, str]] = None,
        lowdim_key_map: Optional[Dict[str, str]] = None,
        use_cache: bool = False,
    ):
        super().__init__()

        obs_shape_meta = shape_meta["obs"]
        rgb_keys: list = []
        lowdim_keys: list = []
        for key, attr in obs_shape_meta.items():
            t = attr.get("type", "low_dim")
            if t == "rgb":
                rgb_keys.append(key)
            elif t == "low_dim":
                lowdim_keys.append(key)

        # Default camera_key_map: identity
        if camera_key_map is None:
            camera_key_map = {k: k for k in rgb_keys}

        # Default lowdim_key_map: all lowdim keys → state_key
        if lowdim_key_map is None:
            lowdim_key_map = {k: state_key for k in lowdim_keys}

        # Validate that all required keys are mapped
        for k in rgb_keys:
            if k not in camera_key_map:
                raise ValueError(
                    f"RGB key '{k}' not in camera_key_map. "
                    f"Provide camera_key_map={{{k!r}: <lerobot_key>}} in the config."
                )
        for k in lowdim_keys:
            if k not in lowdim_key_map:
                raise ValueError(
                    f"Lowdim key '{k}' not in lowdim_key_map. "
                    f"Provide lowdim_key_map={{{k!r}: <lerobot_key>}} in the config."
                )

        # Build or load the replay buffer
        if use_cache:
            cache_zarr_path = str(data_path).rstrip("/\\") + ".lerobot_dp.zarr.zip"
            cache_lock_path = cache_zarr_path + ".lock"
            print(f"[LeRobotDataset] Acquiring cache lock: {cache_lock_path}")
            with FileLock(cache_lock_path):
                if not os.path.exists(cache_zarr_path):
                    try:
                        print("[LeRobotDataset] Cache not found. Building …")
                        replay_buffer = _convert_lerobot_to_replay(
                            store=zarr.MemoryStore(),
                            shape_meta=shape_meta,
                            data_path=data_path,
                            repo_id=repo_id,
                            action_key=action_key,
                            state_key=state_key,
                            rgb_keys=rgb_keys,
                            lowdim_keys=lowdim_keys,
                            camera_key_map=camera_key_map,
                            lowdim_key_map=lowdim_key_map,
                        )
                        print(f"[LeRobotDataset] Saving cache to {cache_zarr_path}")
                        with zarr.ZipStore(cache_zarr_path) as zip_store:
                            replay_buffer.save_to_store(store=zip_store)
                    except Exception:
                        if os.path.exists(cache_zarr_path):
                            os.remove(cache_zarr_path)
                        raise
                else:
                    print(f"[LeRobotDataset] Loading cache from {cache_zarr_path}")
                    with zarr.ZipStore(cache_zarr_path, mode="r") as zip_store:
                        replay_buffer = ReplayBuffer.copy_from_store(
                            src_store=zip_store, store=zarr.MemoryStore()
                        )
                    print("[LeRobotDataset] Cache loaded.")
        else:
            replay_buffer = _convert_lerobot_to_replay(
                store=zarr.MemoryStore(),
                shape_meta=shape_meta,
                data_path=data_path,
                repo_id=repo_id,
                action_key=action_key,
                state_key=state_key,
                rgb_keys=rgb_keys,
                lowdim_keys=lowdim_keys,
                camera_key_map=camera_key_map,
                lowdim_key_map=lowdim_key_map,
            )

        print(
            f"[LeRobotDataset] {replay_buffer.n_episodes} episodes, "
            f"{len(replay_buffer)} total frames."
        )

        # key_first_k: only load the first n_obs_steps obs frames for efficiency
        key_first_k: dict = {}
        if n_obs_steps is not None:
            for key in rgb_keys + lowdim_keys:
                key_first_k[key] = n_obs_steps

        # Train / val split
        val_mask = get_val_mask(
            n_episodes=replay_buffer.n_episodes, val_ratio=val_ratio, seed=seed
        )
        train_mask = ~val_mask
        if max_train_episodes is not None:
            train_mask = downsample_mask(
                mask=train_mask, max_n=max_train_episodes, seed=seed
            )

        sampler = SequenceSampler(
            replay_buffer=replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k,
        )

        self.replay_buffer = replay_buffer
        self.sampler = sampler
        self.shape_meta = shape_meta
        self.rgb_keys = rgb_keys
        self.lowdim_keys = lowdim_keys
        self.n_obs_steps = n_obs_steps
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after

    # ------------------------------------------------------------------
    # BaseImageDataset interface
    # ------------------------------------------------------------------

    def get_validation_dataset(self) -> "LeRobotReplayImageDataset":
        val_set = copy.copy(self)
        val_set.sampler = SequenceSampler(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
        )
        val_set.train_mask = ~self.train_mask
        return val_set

    def get_normalizer(self, mode: str = "limits", **kwargs) -> LinearNormalizer:
        normalizer = LinearNormalizer()

        # Action
        action_stat = array_to_stats(self.replay_buffer["action"])
        normalizer["action"] = get_range_normalizer_from_stat(action_stat)

        # Lowdim observations
        for key in self.lowdim_keys:
            stat = array_to_stats(self.replay_buffer[key])
            normalizer[key] = get_range_normalizer_from_stat(stat)

        # RGB observations: map pixel values from [0,1] to [-1,1]
        for key in self.rgb_keys:
            normalizer[key] = get_image_range_normalizer()

        return normalizer

    def get_all_actions(self) -> torch.Tensor:
        return torch.from_numpy(self.replay_buffer["action"])

    def __len__(self) -> int:
        return len(self.sampler)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        data = self.sampler.sample_sequence(idx)

        # Only keep the first n_obs_steps observation frames
        T_slice = slice(self.n_obs_steps)

        obs_dict: dict = {}

        for key in self.rgb_keys:
            # (T, H, W, C) uint8 → (T, C, H, W) float32 in [0, 1]
            obs_dict[key] = (
                np.moveaxis(data[key][T_slice], -1, 1).astype(np.float32) / 255.0
            )
            del data[key]

        for key in self.lowdim_keys:
            obs_dict[key] = data[key][T_slice].astype(np.float32)
            del data[key]

        return {
            "obs": dict_apply(obs_dict, torch.from_numpy),
            "action": torch.from_numpy(data["action"].astype(np.float32)),
        }
