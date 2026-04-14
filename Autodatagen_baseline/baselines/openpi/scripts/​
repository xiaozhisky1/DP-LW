import dataclasses
import logging
import pickle
import traceback

import numpy as np
import tyro
import flask
from flask import request, jsonify

from openpi.policies import policy_config as _policy_config
from openpi.training import config as _config

app = flask.Flask(__name__)
loaded_policy = None


@dataclasses.dataclass
class Args:
    """Arguments for the serve_x7s script."""

    # Training config name.
    config: str = "pi0_x7s"
    # Checkpoint directory.
    checkpoint_dir: str = "checkpoints/pi0_x7s/exp_name/step_20000"
    # Custom prompt for the policy (used as default_prompt during model loading).
    prompt: str =  "subtasks1: Move the robot base to a position near the fridge to allow manipulation of the fridge door.subtasks 2:Extend the arm to the fridge door handle area and grasp it to prepare for opening. subtasks 3: Pull the door open until the fridge is considered open."

    # Host and port for the Flask server.
    host: str = "0.0.0.0"
    port: int = 8000


def load_model(args: Args):
    """Load the x7s policy."""
    global loaded_policy

    print(f"[Server] Loading policy: config={args.config}, checkpoint={args.checkpoint_dir}")

    train_config = _config.get_config(args.config)
    loaded_policy = _policy_config.create_trained_policy(
        train_config,
        args.checkpoint_dir,
        default_prompt=args.prompt,
    )

    # Print the transform chain for debugging
    if hasattr(loaded_policy, "_input_transform"):
        ct = loaded_policy._input_transform
        if hasattr(ct, "transforms"):
            print("[Server] Input transform chain:")
            for i, t in enumerate(ct.transforms):
                print(f"  [{i}] {type(t).__name__}")

    print("[Server] Policy loaded successfully!")


_request_count = 0
_prev_state = None

@app.route("/infer", methods=["POST"])
def infer():
    global _request_count, _prev_state
    _request_count += 1

    try:
        data = pickle.loads(request.data)

        # ========== DIAGNOSTIC: Inspect input data ==========
        images_dict = data["images"]
        state_raw = np.array(data["state"], dtype=np.float32)
        prompt = data["prompt"]

        if _request_count <= 5 or _request_count % 20 == 0:
            print(f"\n{'='*60}")
            print(f"[Diag] Request #{_request_count}")
            print(f"[Diag] Prompt: {prompt[:80]}...")

            # Check images
            for cam_name, img in images_dict.items():
                img_np = np.asarray(img)
                print(f"[Diag] Image '{cam_name}': shape={img_np.shape}, dtype={img_np.dtype}, "
                      f"min={img_np.min():.3f}, max={img_np.max():.3f}, mean={img_np.mean():.3f}")
                # Check if image is all black/white/constant
                if img_np.std() < 1.0:
                    print(f"  [WARN] Image '{cam_name}' appears CONSTANT (std={img_np.std():.4f})!")
                # Check channel count
                if img_np.ndim == 3 and img_np.shape[-1] != 3:
                    print(f"  [WARN] Image '{cam_name}' has {img_np.shape[-1]} channels (expected 3)!")

            # Check state
            print(f"[Diag] State: shape={state_raw.shape}, "
                  f"min={state_raw.min():.4f}, max={state_raw.max():.4f}, mean={state_raw.mean():.4f}")
            print(f"[Diag] State values: {np.array2string(state_raw, precision=3, suppress_small=True)}")

            # Check if state changed from previous request
            if _prev_state is not None:
                state_diff = np.abs(state_raw - _prev_state).max()
                print(f"[Diag] State max change from prev: {state_diff:.6f}")
                if state_diff < 1e-5:
                    print(f"  [WARN] State appears UNCHANGED between requests!")

        _prev_state = state_raw.copy()

        # ========== Build observation dict ==========
        # Map: "images" (client) -> "image" (model internal key)
        # Ensure images are RGB (3 channels), strip alpha if needed
        clean_images = {}
        for cam_name, img in images_dict.items():
            img_np = np.asarray(img)
            if img_np.ndim == 3 and img_np.shape[-1] == 4:
                img_np = img_np[..., :3]  # RGBA -> RGB
            clean_images[cam_name] = img_np

        obs = {
            "image": clean_images,
            "state": state_raw,
            "prompt": prompt,
        }

        action = loaded_policy.infer(obs)

        # ========== DIAGNOSTIC: Inspect output action ==========
        if _request_count <= 5 or _request_count % 20 == 0:
            if "actions" in action:
                act = np.asarray(action["actions"])
                print(f"[Diag] Action output: shape={act.shape}, "
                      f"min={act.min():.4f}, max={act.max():.4f}, mean={act.mean():.4f}")
                # Print first step of the action chunk
                print(f"[Diag] Action[0]: {np.array2string(act[0], precision=3, suppress_small=True)}")
                print(f"[Diag] Action[9]: {np.array2string(act[min(9, len(act)-1)], precision=3, suppress_small=True)}")
            print(f"{'='*60}\n")

        # Convert all numpy/jax arrays to pure Python types for pickle compatibility
        safe_action = _recursive_to_python(action)
        return pickle.dumps(safe_action)

    except Exception as e:
        print("[Server] EXCEPTION:")
        traceback.print_exc()
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/health", methods=["GET"])
def health():
    return "OK", 200


def _recursive_to_python(data):
    """Recursively convert numpy/jax arrays to pure Python types for safe pickling."""
    if isinstance(data, dict):
        return {k: _recursive_to_python(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return [_recursive_to_python(v) for v in data]
    elif hasattr(data, "tolist"):
        return data.tolist()
    elif hasattr(data, "item"):
        return data.item()
    return data


def main(args: Args):
    load_model(args)
    print(f"[Server] Starting Flask server on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=False, debug=False)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main(tyro.cli(Args))
