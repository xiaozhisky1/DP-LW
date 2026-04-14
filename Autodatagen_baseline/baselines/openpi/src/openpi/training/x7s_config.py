import dataclasses
from typing import Tuple

import openpi.models.pi0 as _pi0  # 假设使用的是 Pi0 模型
import openpi.training.config as _config
import openpi.transforms as transforms

def get_x7s_config(dataset_name: str = "BowlAndCup") -> _config.TrainConfig:
    """
    针对自定义 X7s LeRobot 数据集的 OpenPi 配置
    """
    
    # 1. 定义动作和状态维度 (来自 info.json)
    ACTION_DIM = 25
    STATE_DIM = 25
    
    # 2. 模型配置 (Pi0)
    # 这里的 horizon 和 dim 需要与你的任务匹配
    model = _pi0.Pi0Config(
        action_dim=ACTION_DIM,
        observation_dim=STATE_DIM,
        max_horizon=10,  # 预测步长，根据任务难度调整
    )

    # 3. 数据转换 (Transforms)
    # 将 LeRobot 的 key 映射到 OpenPi 的 key，并进行 Resize
    data_transforms = transforms.Group(
        inputs=[
            # 图像处理：Resize 到 224x224 并归一化 (Pi0 默认需要 SigLIP 尺寸)
            transforms.ImageTransform(
                # 你的 info.json 中的相机列表
                keys=[
                    "observation.images.eye_in_hand_camera",
                    "observation.images.left_hand_camera",
                    "observation.images.right_hand_camera"
                ],
                # 映射到模型内部名称 (通常需要简写，或者保持原名但需模型支持)
                # 这里我们将它们重命名为通用的 key，方便模型处理
                rename_to=[
                    "images/eye_in_hand",
                    "images/left_hand",
                    "images/right_hand"
                ],
                resize=(224, 224),
                normalize=True, # 归一化到 [0, 1] 或 [-1, 1] 取决于 OpenPi 实现
            ),
            # 状态处理：映射 observation.state -> state
            transforms.Rename(
                mapping={
                    "observation.state": "state",
                    "action": "actions",  # LeRobot 叫 action, OpenPi 通常叫 actions (复数)
                }
            )
        ]
    )

    # 4. 数据集配置
    data = _config.DataConfig(
        # 你的数据集文件夹名称，例如 "BowlAndCup" 或 "BowlAndCup_2026..."
        repo_id=dataset_name, 
        
        # 这里的 transforms 用于 compute_norm_stats 和 train
        data_transforms=data_transforms,
        
        # 如果是本地路径，OpenPi 可能需要知道 root_dir。
        # 如果 OpenPi 使用标准的 lerobot.load_dataset，
        # 你可能需要设置环境变量 LEROBOT_HOME 指向 results/lerobot_dataset 的上级目录
    )

    # 5. 训练配置
    return _config.TrainConfig(
        project_name="x7s_training",
        exp_name=f"pi0_x7s_{dataset_name}",
        model=model,
        data=data,
        batch_size=8,   # 根据显存调整
        num_train_steps=20000,
        save_interval=2000,
        # 确保 assets_dirs 指向能写入 stats 的地方
    )

# 注册配置 (如果 OpenPi 使用注册表模式)
# _config.register_config("x7s_local", get_x7s_config("BowlAndCup"))