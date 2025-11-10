from pathlib import Path
from typing import Dict, Any, Optional, Type
from omegaconf import DictConfig

from .sequence_base import SequenceBase
from .reader import BaseSequenceReader
from .types import Mode, LoaderDataDict

class SequenceForStream(SequenceBase):
    def __init__(self,
                 path: Path,
                 dataset_mode: Mode,
                 dataset_config: DictConfig,
                 reader_cls: Optional[Type[BaseSequenceReader]] = None):
        
        super().__init__(path, dataset_mode, dataset_config, reader_cls)

        self.start_indices = list(range(0, self.length, self.seq_len))
        self.stop_indices = self.start_indices[1:] + [self.length]
        self.length = len(self.start_indices)

    def __len__(self) -> int:
        return self.length

    def get_fully_padded_sample(self) -> Dict[str, Any]:
        pad = self._get_padding_frame()
        return {f"{k}_seq": [pad.get(k)] * self.seq_len for k in self.keys_to_load}

    def __getitem__(self, index: int) -> LoaderDataDict:
        if not (0 <= index < self.length):
            raise IndexError(f"Index {index} out of bounds for {self.length} chunks")

        start, stop = self.start_indices[index], self.stop_indices[index]
        
        frames = [self._get_frame(i) for i in range(start, stop)]

        pad_len = self.seq_len - len(frames)
        if pad_len > 0:
            pad = self._get_padding_frame()
            frames.extend([pad.copy() for _ in range(pad_len)])

        seq_data: LoaderDataDict = {}
        for k in self.keys_to_load:
            seq_data[f"{k}_seq"] = [f[k] for f in frames]

        seq_data["is_first_sample"] = (index == 0)
        seq_data["is_padded_mask"] = [False] * (self.seq_len - pad_len) + [True] * pad_len

        return seq_data