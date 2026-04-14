import json
import importlib.util
import os
from pathlib import Path
from typing import Any

import torch


class LeRobotACTDataset(torch.utils.data.Dataset):
    """LeRobot v3 local dataset adapter for il_lib ACT training."""

    @staticmethod
    def get_all_demo_keys(data_path: str, task_name: Any) -> list[int]:
        info_path = Path(data_path) / "meta" / "info.json"
        if not info_path.exists():
            return [0]
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        total_episodes = int(info.get("total_episodes", 1))
        return list(range(total_episodes))

    def __init__(
        self,
        *args,
        data_path: str,
        demo_keys: list[int],
        seed: int,
        ctx_len: int = 20,
        obs_window_size: int = 1,
        action_key: str = "action",
        state_key: str = "observation.state",
        camera_keys: list[str] | None = None,
        repo_id: str = "local/openfridge",
        **kwargs,
    ):
        del args, seed, kwargs
        self._root = os.path.expanduser(data_path)
        self._demo_keys = list(demo_keys)
        self._ctx_len = int(ctx_len)
        self._obs_window_size = int(obs_window_size)
        self._action_key = action_key
        self._state_key = state_key
        self._camera_keys = camera_keys or [
            "observation.images.first_person_camera",
            "observation.images.left_hand_camera",
            "observation.images.right_hand_camera",
        ]

        # Best-effort patch for LeRobot v3 metadata compatibility.
        self._prepare_local_lerobot_v3(self._root)
        self._patch_lerobot_offline_fallback()
        self._patch_hf_list_feature_alias()

        from lerobot.datasets.lerobot_dataset import LeRobotDataset

        fps = self._read_fps(self._root)
        self._dataset = LeRobotDataset(
            repo_id,
            root=self._root,
            delta_timestamps={action_key: [t / float(fps) for t in range(self._ctx_len)]},
            video_backend="pyav",
        )
        self._sample_indices = self._build_sample_indices()

    @staticmethod
    def _load_openpi_v3_compat():
        # Load compatibility module directly from workspace path to avoid PYTHONPATH dependency.
        workspace_root = Path(__file__).resolve().parents[4]
        compat_path = workspace_root / "baselines" / "openpi" / "src" / "openpi" / "training" / "lerobot_v3_compat.py"
        if not compat_path.exists():
            return None
        spec = importlib.util.spec_from_file_location("openpi_lerobot_v3_compat_local", compat_path)
        if spec is None or spec.loader is None:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    def _prepare_local_lerobot_v3(self, root: str) -> None:
        # Ensure local v3 dataset can be read by lerobot v2.1 code path without downloading from hub.
        compat = self._load_openpi_v3_compat()
        if compat is None:
            self._ensure_local_meta_jsonl(root)
            return
        try:
            compat.apply_lerobot_v3_metadata_patch()
            compat.ensure_v21_meta_from_v30(root)
            compat.set_v3_skip_timestamp_check(True)
        except Exception:
            # Keep a best-effort behavior to avoid blocking non-v3 datasets.
            pass
        self._ensure_local_meta_jsonl(root)

    @staticmethod
    def _ensure_local_meta_jsonl(root: str) -> None:
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
            task_text = "OpenFridge"
            # Prefer task text from tasks.parquet index if available.
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
            episodes_parquet_files = sorted((meta_dir / "episodes").glob("chunk-*/*.parquet"))
            if episodes_parquet_files:
                try:
                    import pandas as pd

                    episodes_df = pd.concat([pd.read_parquet(p) for p in episodes_parquet_files], ignore_index=True)
                    episodes_df = episodes_df.sort_values("episode_index")
                    with episodes_jsonl.open("w", encoding="utf-8") as f:
                        for _, row in episodes_df.iterrows():
                            tasks = row.get("tasks", ["OpenFridge"])
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
                                {
                                    "episode_index": ep_idx,
                                    "tasks": ["OpenFridge"],
                                    "length": 1,
                                }
                            )
                            + "\n"
                        )

        if (not episodes_stats_jsonl.exists()) or episodes_stats_jsonl.stat().st_size == 0:
            # Minimal per-episode stats placeholder to satisfy lerobot v2.1 loader.
            with episodes_stats_jsonl.open("w", encoding="utf-8") as f:
                for ep_idx in range(total_episodes):
                    f.write(json.dumps({"episode_index": ep_idx, "stats": {}}) + "\n")

    @staticmethod
    def _patch_lerobot_offline_fallback() -> None:
        # If lerobot falls back to get_safe_version, force it to stay local and skip HF refs lookup.
        try:
            import lerobot.datasets.utils as _utils
            import lerobot.datasets.lerobot_dataset as _ld
        except Exception:
            return

        if getattr(_utils, "_il_lib_offline_patched", False):
            return

        def _local_safe_version(_repo_id: str, version: str):
            v = str(version)
            return v if v.startswith("v") else f"v{v}"

        _utils.get_safe_version = _local_safe_version
        _ld.get_safe_version = _local_safe_version
        _utils._il_lib_offline_patched = True

    @staticmethod
    def _patch_hf_list_feature_alias() -> None:
        # Some local parquet files may encode datasets features metadata with "_type": "List",
        # while current `datasets` expects "Sequence" / "LargeList".
        try:
            from datasets.features.features import Sequence, _FEATURE_TYPES
        except Exception:
            return
        if "List" not in _FEATURE_TYPES:
            _FEATURE_TYPES["List"] = Sequence

    @staticmethod
    def _read_fps(root: str) -> int:
        info_path = Path(root) / "meta" / "info.json"
        if not info_path.exists():
            return 30
        with info_path.open("r", encoding="utf-8") as f:
            info = json.load(f)
        return int(info.get("fps", 30))

    @staticmethod
    def _to_torch(x: Any) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        return torch.as_tensor(x)

    @staticmethod
    def _to_chw(img: torch.Tensor) -> torch.Tensor:
        # Accept HWC or CHW and return CHW.
        if img.ndim != 3:
            raise ValueError(f"Expected 3D image tensor, got shape {tuple(img.shape)}")
        if img.shape[0] in (1, 3, 4):
            out = img
        else:
            out = img.permute(2, 0, 1)
        if out.dtype != torch.uint8:
            if torch.is_floating_point(out):
                out = (out.clamp(0, 1) * 255.0).to(torch.uint8)
            else:
                out = out.to(torch.uint8)
        return out

    def __len__(self) -> int:
        return len(self._sample_indices)

    def _build_sample_indices(self) -> list[int]:
        # NOTE:
        # LeRobotDataset with `episodes=[...]` can fail for non-contiguous episode ids because
        # internal indexing treats `episode_index` as positional. We load the full local dataset
        # and perform sample-level filtering here instead.
        if not self._demo_keys:
            return list(range(len(self._dataset)))
        demo_set = set(int(x) for x in self._demo_keys)
        episode_indices = self._dataset.hf_dataset["episode_index"]
        sample_indices = [i for i, ep in enumerate(episode_indices) if int(ep) in demo_set]
        if not sample_indices:
            raise ValueError("No samples matched demo_keys in local LeRobot dataset.")
        return sample_indices

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._dataset[self._sample_indices[index]]

        obs: dict[str, torch.Tensor] = {}

        state = self._to_torch(sample[self._state_key]).to(torch.float32)
        if state.ndim == 1:
            state = state.unsqueeze(0)
        else:
            state = state[: self._obs_window_size]
        obs["state"] = state

        for camera_key in self._camera_keys:
            if camera_key not in sample:
                continue
            img = self._to_torch(sample[camera_key])
            if img.ndim == 3:
                img = self._to_chw(img).unsqueeze(0)
            elif img.ndim == 4:
                # Accept TCHW or THWC.
                if img.shape[1] not in (1, 3, 4):
                    img = img.permute(0, 3, 1, 2)
                img = img[: self._obs_window_size]
                img = torch.stack([self._to_chw(frame) for frame in img], dim=0)
            else:
                raise ValueError(f"Unsupported image shape for {camera_key}: {tuple(img.shape)}")
            obs[f"{camera_key}::rgb"] = img

        if not any("::rgb" in k for k in obs):
            raise ValueError("No RGB observations found in LeRobot sample.")

        actions = self._to_torch(sample[self._action_key]).to(torch.float32)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)

        t = min(actions.shape[0], self._ctx_len)
        action_dim = actions.shape[-1]
        padded_actions = torch.zeros((self._ctx_len, action_dim), dtype=torch.float32)
        padded_actions[:t] = actions[:t]

        masks = torch.zeros((self._ctx_len,), dtype=torch.bool)
        masks[:t] = True

        return {
            "obs": obs,
            "actions": {self._action_key: padded_actions},
            "masks": masks,
        }
