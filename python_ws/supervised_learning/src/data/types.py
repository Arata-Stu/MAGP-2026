from enum import auto, Enum
from typing import Dict, List, Any, TypedDict
import torch

class Mode(Enum):
    TRAIN = auto()
    VALIDATION = auto()
    TESTING = auto()

class ModelMode(Enum):
    IMAGE = auto()
    LIDAR = auto()
    FUSION = auto()

class LoaderDataDict(TypedDict, total=False):
    """
    各データローダ（stream/random）で返す1サンプルの構造を型ヒントとして定義。
    全てのキーは任意（total=False）。
    """
    # 入力系列データ
    rgb_seq: List[torch.Tensor]
    # マスク類
    is_first_sample: bool
    is_padded_mask: List[bool]
    meta: Dict[str, Any]
