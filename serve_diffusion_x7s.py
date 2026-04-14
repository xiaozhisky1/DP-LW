"""
Diffusion Policy inference server for x7s robot.

Compatible with the clients used by serve_x7s.py (OpenPI) and serve_act_x7s.py (ACT).
Accepts pickle-serialized POST requests at /infer with keys:
    images : dict[str, np.ndarray]  -- camera name -> (H, W, C) image
    state  : np.ndarray             -- proprioceptive state vector
    prompt : str (optional, ignored by diffusion policy)

Returns pickle-serialized dict:
    actions : np.ndarray  -- (n_action_steps, action_dim)

Usage:
    python serve_diffusion_x7s.py -c path/to/checkpoint.ckpt
    python serve_diffusion_x7s.py -c path/to/checkpoint.ckpt -d cuda:0 --port 8000
"""

import argparse
import logging
import pickle
import traceback
from collections import deque

import dill
import flask
import hydra
import numpy as np
import torch
import torchvision.transforms.functional as TF
from flask import jsonify, request
from omegaconf import OmegaConf

from diffusion_policy.workspace.base_workspace import BaseWorkspace

# Register custom OmegaConf resolvers used in diffusion_policy configs
OmegaConf.register_new_resolver("eval", eval, replace=True)

app = flask.Flask(__name__)

# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------
_POLICY = None
_DEVICE = "cuda:0"
_N_OBS_STEPS = 2
_IMG_SHAPE = {}        # model_key -> (C, H, W) from shape_meta
_OBS_HISTORY = {}      # model_key -> deque of numpy arrays
_REQUEST_COUNT = 0

# ---------------------------------------------------------------------------
# Camera key mapping: client key -> diffusion-policy shape_meta key
# (covers every naming convention used by the x7s client ecosystem)
# ---------------------------------------------------------------------------
CAMERA_KEY_MAP = {
    # Simulator / OpenPI-style keys
    "base_0_rgb":         "first_person_image",
    "left_wrist_0_rgb":   "left_hand_image",
    "right_wrist_0_rgb":  "right_hand_image",
    # Direct camera names (from x7s client)
    "first_person_camera": "first_person_image",
    "left_hand_camera":    "left_hand_image",
    "right_hand_camera":   "right_hand_image",
    # LeRobot dataset-style keys
    "observation.images.first_person_camera": "first_person_image",
    "observation.images.left_hand_camera":    "left_hand_image",
    "observation.images.right_hand_camera":   "right_hand_image",
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _load_policy(args):
    global _POLICY, _DEVICE, _N_OBS_STEPS, _IMG_SHAPE

    _DEVICE = args.device

    # Load checkpoint (same procedure as eval.py)
    payload = torch.load(open(args.checkpoint, "rb"), pickle_module=dill)
    cfg = payload["cfg"]

    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=None)

    # Pick EMA model when available
    policy = workspace.model
    if args.use_ema and getattr(workspace, "ema_model", None) is not None:
        policy = workspace.ema_model
        print("[DP-Server] Using EMA model")
    else:
        print("[DP-Server] Using base model")

    policy.to(_DEVICE)
    policy.eval()
    _POLICY = policy
    _N_OBS_STEPS = cfg.n_obs_steps

    # Extract expected image shapes from shape_meta
    shape_meta = OmegaConf.to_container(cfg.shape_meta, resolve=True)
    for key, attr in shape_meta["obs"].items():
        if attr.get("type", "low_dim") == "rgb":
            _IMG_SHAPE[key] = tuple(attr["shape"])  # (C, H, W)

    print(f"[DP-Server] Checkpoint : {args.checkpoint}")
    print(f"[DP-Server] Device     : {_DEVICE}")
    print(f"[DP-Server] n_obs_steps: {_N_OBS_STEPS}")
    print(f"[DP-Server] Image keys : {_IMG_SHAPE}")


# ---------------------------------------------------------------------------
# Observation preprocessing
# ---------------------------------------------------------------------------
def _reset_history():
    """Clear the observation history buffer (call between episodes)."""
    global _OBS_HISTORY
    _OBS_HISTORY.clear()


def _prepare_obs(images: dict, state: np.ndarray) -> dict:
    """Convert raw client data into the obs_dict expected by policy.predict_action().

    The policy expects obs_dict[key] of shape (B, To, ...) where To = n_obs_steps.
    We maintain a sliding-window history buffer so that each /infer call only needs
    to supply the *current* frame; the server fills in prior timesteps automatically.
    """
    global _OBS_HISTORY

    current = {}

    # -- images ---------------------------------------------------------------
    for in_key, img_raw in images.items():
        if in_key not in CAMERA_KEY_MAP:
            continue
        model_key = CAMERA_KEY_MAP[in_key]
        if model_key not in _IMG_SHAPE:
            continue

        img_np = np.asarray(img_raw)

        # RGBA -> RGB
        if img_np.ndim == 3 and img_np.shape[-1] == 4:
            img_np = img_np[..., :3]

        # Ensure uint8
        if img_np.dtype != np.uint8:
            if np.issubdtype(img_np.dtype, np.floating):
                img_np = np.clip(img_np, 0.0, 1.0)
                img_np = (img_np * 255.0).astype(np.uint8)
            else:
                img_np = img_np.astype(np.uint8)

        # HWC uint8 -> CHW float32 [0, 1]
        img_tensor = TF.to_tensor(img_np)  # (C, H, W)

        # Resize to the shape the policy was trained on
        target_h, target_w = _IMG_SHAPE[model_key][1], _IMG_SHAPE[model_key][2]
        if img_tensor.shape[1] != target_h or img_tensor.shape[2] != target_w:
            img_tensor = TF.resize(img_tensor, [target_h, target_w], antialias=True)

        current[model_key] = img_tensor.numpy()

    # -- proprioceptive state -------------------------------------------------
    current["agent_pos"] = np.asarray(state, dtype=np.float32)

    # -- build temporal window (To timesteps) ---------------------------------
    obs_dict = {}
    for key, value in current.items():
        if key not in _OBS_HISTORY:
            _OBS_HISTORY[key] = deque(maxlen=_N_OBS_STEPS)

        _OBS_HISTORY[key].append(value)

        # Pad with earliest observation when not enough history yet
        while len(_OBS_HISTORY[key]) < _N_OBS_STEPS:
            _OBS_HISTORY[key].appendleft(_OBS_HISTORY[key][0])

        stacked = np.stack(list(_OBS_HISTORY[key]), axis=0)  # (To, ...)
        obs_dict[key] = torch.from_numpy(stacked).unsqueeze(0).to(_DEVICE)  # (1, To, ...)

    return obs_dict


# ---------------------------------------------------------------------------
# Flask endpoints
# ---------------------------------------------------------------------------
@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


@app.route("/reset", methods=["POST"])
def reset():
    """Reset observation history (call between episodes)."""
    _reset_history()
    return "OK", 200


@app.route("/infer", methods=["POST"])
def infer():
    global _REQUEST_COUNT
    _REQUEST_COUNT += 1

    try:
        if _POLICY is None:
            return jsonify({"error": "policy not loaded"}), 500

        data = pickle.loads(request.data)
        images = data.get("images", {})
        state = np.asarray(data.get("state"), dtype=np.float32)

        # Diagnostic logging for first few requests and periodically
        if _REQUEST_COUNT <= 5 or _REQUEST_COUNT % 50 == 0:
            print(f"\n{'=' * 60}")
            print(f"[Diag] Request #{_REQUEST_COUNT}")
            for cam_name, img in images.items():
                img_np = np.asarray(img)
                print(
                    f"[Diag] Image '{cam_name}': shape={img_np.shape}, "
                    f"dtype={img_np.dtype}, range=[{img_np.min():.3f}, {img_np.max():.3f}]"
                )
            print(f"[Diag] State: shape={state.shape}, range=[{state.min():.4f}, {state.max():.4f}]")

        obs_dict = _prepare_obs(images, state)

        with torch.no_grad():
            result = _POLICY.predict_action(obs_dict)

        # (1, n_action_steps, action_dim) -> (n_action_steps, action_dim)
        action = result["action"].cpu().numpy()[0]

        if _REQUEST_COUNT <= 5 or _REQUEST_COUNT % 50 == 0:
            print(
                f"[Diag] Action: shape={action.shape}, "
                f"range=[{action.min():.4f}, {action.max():.4f}]"
            )
            print(f"{'=' * 60}\n")

        return pickle.dumps({"actions": action})

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Diffusion Policy inference server for x7s robot"
    )
    parser.add_argument(
        "-c", "--checkpoint", required=True, help="Path to checkpoint .ckpt file"
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("-d", "--device", default="cuda:0")
    parser.add_argument(
        "--no-ema",
        action="store_true",
        help="Use base model instead of EMA model",
    )
    parsed = parser.parse_args()

    class _Args:
        checkpoint = parsed.checkpoint
        host = parsed.host
        port = parsed.port
        device = parsed.device
        use_ema = not parsed.no_ema

    args = _Args()
    _load_policy(args)
    print(f"[DP-Server] Serving on http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
