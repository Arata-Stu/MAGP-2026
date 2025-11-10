from omegaconf import DictConfig
from pathlib import Path
import torch
from .custom_datapipes.datapipe import MapDataPipe
from typing import Optional, Any, Dict, Type
from .types import Mode
from .reader import BaseSequenceReader 

class SequenceBase(MapDataPipe):

    def __init__(self,
                 path: Path,
                 dataset_mode: Mode,
                 dataset_config: DictConfig,
                 reader_cls: Optional[Type[BaseSequenceReader]] = None):
        
        super().__init__()
        assert reader_cls is not None, "reader_cls must be provided"
        
        self.seq_len = dataset_config.sequence_length
        self.keys_to_load = dataset_config.get(
            "keys_to_load",
            ["rgb", "scan", "accel", "steer", "odom"],
        )
        
        self.reader = reader_cls(path, seq_len=self.seq_len, keys_to_load=self.keys_to_load)
        self.length = self.reader.length 
        self._padding_frame: Optional[Dict[str, torch.Tensor]] = None


    def __len__(self) -> int:
        return self.length

    def _get_padding_frame(self) -> Dict[str, torch.Tensor]:
        if self._padding_frame is not None:
            return self._padding_frame

        dummy = self.reader.load_frame(0) 
        pad = {k: torch.zeros_like(v) for k, v in dummy.items() if torch.is_tensor(v) and k in self.keys_to_load}
        self._padding_frame = pad
        return pad

    def _get_frame(self, idx: int) -> Dict[str, Any]:

        frame = self.reader.load_frame(idx)
        padded_frame = {}
        
        for k in self.keys_to_load:
            val = frame.get(k)
            if val is not None:
                padded_frame[k] = val
            else:
                padding_dict = self._get_padding_frame()
                if k in padding_dict:
                    padded_frame[k] = padding_dict[k]
                else:
                    padded_frame[k] = [] 
                    
        return padded_frame

    def __getitem__(self, index: int):
        raise NotImplementedError(
            "SequenceBase is an abstract class. "
            "Use SequenceForStream or SequenceForRandomAccess."
        )