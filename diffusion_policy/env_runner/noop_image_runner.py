"""
No-operation image runner for real-robot / offline datasets that have no
simulation environment.  Returns an empty log dict so the training loop
runs without errors.  Set ``rollout_every`` to a large value in the
workspace config so this runner is called infrequently.
"""

from typing import Dict

from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.policy.base_image_policy import BaseImagePolicy


class NoopImageRunner(BaseImageRunner):
    """Dummy runner that skips evaluation and returns an empty log."""

    def __init__(self, output_dir: str):
        super().__init__(output_dir)

    def run(self, policy: BaseImagePolicy) -> Dict:
        return {}
