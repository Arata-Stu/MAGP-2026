from pathlib import Path
from typing import Optional, Type
from omegaconf import DictConfig

from .sequence_base import SequenceBase
from .reader import BaseSequenceReader
from .types import Mode, LoaderDataDict

class SequenceForRandomAccess(SequenceBase):

    def __init__(self,
                 path: Path,
                 dataset_mode: Mode,
                 dataset_config: DictConfig,
                 reader_cls: Optional[Type[BaseSequenceReader]] = None):
        
        super().__init__(path, dataset_mode, dataset_config, reader_cls)

        self.length = max(0, self.length - self.seq_len + 1)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> LoaderDataDict:
        if not (0 <= index < self.length):
            raise IndexError(f"Index {index} out of bounds for length {self.length}")

        start, end = index, index + self.seq_len
        
        frames = [self._get_frame(i) for i in range(start, end)]

        seq_data: LoaderDataDict = {}
        for k in self.keys_to_load:
            seq_data[f"{k}_seq"] = [f[k] for f in frames]

        seq_data["is_first_sample"] = True 
        seq_data["is_padded_mask"] = [False] * self.seq_len 

        return seq_data