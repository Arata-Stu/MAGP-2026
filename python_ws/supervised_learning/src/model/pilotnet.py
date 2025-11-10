import torch
import torch.nn as nn

from src.data.types import ModelMode
from src.utils.weight_init import _init_weights

class ImageNetBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_type = ModelMode.IMAGE
        self.is_rnn = None

class PilotNet(ImageNetBase):
    def __init__(self, num_outputs=2):
        super().__init__()

        # --- 畳み込み層 (特徴抽出) ---
        self.features = nn.Sequential(
            # 1. Conv層: 3ch -> 24ch, Kernel 5x5, Stride 2
            nn.Conv2d(in_channels=3, out_channels=24, kernel_size=5, stride=2),
            nn.ReLU(),
            # 2. Conv層: 24ch -> 36ch, Kernel 5x5, Stride 2
            nn.Conv2d(in_channels=24, out_channels=36, kernel_size=5, stride=2),
            nn.ReLU(),
            # 3. Conv層: 36ch -> 48ch, Kernel 5x5, Stride 2
            nn.Conv2d(in_channels=36, out_channels=48, kernel_size=5, stride=2),
            nn.ReLU(),
            # 4. Conv層: 48ch -> 64ch, Kernel 3x3, Stride 1
            nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            # 5. Conv層: 64ch -> 64ch, Kernel 3x3, Stride 1
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU()
        )

        # --- 全結合層 (操舵と速度の決定) ---
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # ★ 1. FC層: 入力ユニット数を変更
            # (入力サイズは 64ch * 8H * 13W = 6656 となります)
            nn.Linear(in_features=6656, out_features=1164),
            nn.ReLU(),
            # 2. FC層: 1164 -> 100
            nn.Linear(in_features=1164, out_features=100),
            nn.ReLU(),
            # 3. FC層: 100 -> 50
            nn.Linear(in_features=100, out_features=50),
            nn.ReLU(),
            # 4. FC層: 50 -> 10
            nn.Linear(in_features=50, out_features=10),
            nn.ReLU(),
            # 5. 出力層: 10 -> 2 (steer, speed)
            nn.Linear(in_features=10, out_features=num_outputs)
        )

        self.apply(_init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 4:
            x = x[:, -1, :, :, :]  # (B, T, C, H, W) -> (B, C, H, W)

        x = self.features(x)
        output = self.classifier(x)

        if output.shape[1] == 2:
            steer = torch.atan(output[:, 0]) * 2
            speed = output[:, 1] 
            output = torch.stack([steer, speed], dim=1)
        else:
            output = torch.atan(output) * 2
            
        return output