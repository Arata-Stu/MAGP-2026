from __future__ import annotations
from typing import Dict, Any, Optional
import random


class AugmentorBase:
    """
    Augmentorの基本クラス。
    各派生クラスで randomize_augmentation() と apply() を実装する。
    """
    def __init__(self, automatic_randomization: bool = True):
        self.automatic_randomization = automatic_randomization
        self.state: Optional[Dict[str, Any]] = None

    def randomize_augmentation(self):
        """ランダムなaugmentationパラメータをサンプリングする。"""
        raise NotImplementedError

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """実際のaugmentationを適用する。"""
        raise NotImplementedError

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """呼び出し時の共通フロー。"""
        if self.automatic_randomization:
            self.randomize_augmentation()
        assert self.state is not None, "Augmentor state not initialized"
        return self.apply(sample)


class RandomAugmentor(AugmentorBase):
    """
    例: flip や rotate などを後で追加できる雛形クラス。
    ここではダミー実装のみ行う。
    """
    def __init__(self, automatic_randomization: bool = True):
        super().__init__(automatic_randomization)

    def randomize_augmentation(self):
        self.state = {
            "do_flip": random.random() < 0.5,
            "do_rotate": random.random() < 0.3,
            "angle_deg": random.uniform(-10, 10),
        }

    def apply(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        # 実際の処理はここに書く（現時点では何もしない）
        # 例:
        # if self.state["do_flip"]:
        #     sample["rgb_seq"] = [torch.flip(f, dims=[-1]) for f in sample["rgb_seq"]]
        return sample
