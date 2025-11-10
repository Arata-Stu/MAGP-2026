from pathlib import Path
from typing import List, Iterable, Type, Union
from omegaconf import DictConfig
from torch.utils.data import Dataset, ConcatDataset
from tqdm import tqdm

from .types import LoaderDataDict, Mode
from .sequece_for_random import SequenceForRandomAccess
from .augmentor import RandomAugmentor
from .reader import BaseSequenceReader 


class SequenceDataset(Dataset):

    def __init__(self,
                 path: Path,
                 dataset_mode: Mode,
                 dataset_config: DictConfig,
                 reader_cls: Union[Type[BaseSequenceReader], None] = None):
        assert path.is_dir(), f"Invalid path: {path}"

        self.sequence = SequenceForRandomAccess(path, dataset_mode, dataset_config, reader_cls)

        self.spatial_augmentor = None
        if dataset_mode == Mode.TRAIN:
            resolution_hw = tuple(dataset_config.resolution_hw)
            assert len(resolution_hw) == 2

            aug_cfg = dataset_config.get("augmentation", None)
            if aug_cfg is not None:
                self.spatial_augmentor = RandomAugmentor(
                    automatic_randomization=True,   
                )
            else:
                self.spatial_augmentor = RandomAugmentor(automatic_randomization=True)


    def __len__(self):
        return len(self.sequence)

    def __getitem__(self, i: int) -> LoaderDataDict:
        item = self.sequence[i]

        if self.spatial_augmentor is not None:
            item = self.spatial_augmentor(item)
        return item


class CustomConcatDataset(ConcatDataset):

    datasets: List[SequenceDataset]

    def __init__(self, datasets: Iterable[SequenceDataset]):
        super().__init__(datasets=datasets)


def build_random_access_dataset(dataset_mode: Mode,
                                dataset_config: DictConfig,
                                reader_cls: Union[Type[BaseSequenceReader], None] = None) -> CustomConcatDataset:

    root = Path(dataset_config.path)
    assert root.is_dir(), f"Invalid dataset root: {root}"

    mode_dir = {
        Mode.TRAIN: "train",
        Mode.VALIDATION: "val",
        Mode.TESTING: "test",
    }[dataset_mode]

    split_path = root / mode_dir
    assert split_path.is_dir(), f"Missing directory: {split_path}"

    seqs = [
        SequenceDataset(p, dataset_mode, dataset_config, reader_cls)
        for p in tqdm(sorted(split_path.iterdir()), desc=f"Loading {mode_dir} sequences")
        if p.is_dir()
    ]

    print(f"✅ Found {len(seqs)} sequences for {mode_dir} mode.")
    return CustomConcatDataset(seqs)
