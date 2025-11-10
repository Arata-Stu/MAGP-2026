from pathlib import Path
from typing import List, Union, Type
from omegaconf import DictConfig
from functools import partialmethod
from .custom_datapipes.datapipe import IterDataPipe
from .types import Mode
from .sequence_for_stream import SequenceForStream
from .stream_concat_datapipe import ConcatStreamingDataPipe
from .stream_sharded_datapipe import ShardedStreamingDataPipe
from .augmentor import RandomAugmentor
from .reader import BaseSequenceReader 


class RandAugmentIterDataPipeKitti(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe,
                 dataset_config: DictConfig):
        super().__init__()
        self.source_dp = source_dp

        resolution_hw = tuple(dataset_config.resolution_hw)
        if dataset_config.get("downsample", False):
            resolution_hw = tuple(x // 2 for x in resolution_hw)

        aug_cfg = dataset_config.get("augmentation", None)
        if aug_cfg is not None:
            self.augmentor = RandomAugmentor(
                automatic_randomization=False,
            )
        else:
            self.augmentor = RandomAugmentor(automatic_randomization=False)

    def __iter__(self):
        self.augmentor.randomize_augmentation()
        for sample in self.source_dp:
            yield self.augmentor(sample)


def build_stream_dataset(dataset_mode: Mode,
                         dataset_config: DictConfig,
                         reader_cls: Union[Type[BaseSequenceReader], None] = None) -> Union[ConcatStreamingDataPipe, ShardedStreamingDataPipe]:
    dataset_root = Path(dataset_config.path)
    mode_dir = {
        Mode.TRAIN: "train",
        Mode.VALIDATION: "val",
        Mode.TESTING: "test"
    }[dataset_mode]
    split_path = dataset_root / mode_dir
    seqs: List[SequenceForStream] = [
        SequenceForStream(p, dataset_mode, dataset_config, reader_cls)
        for p in sorted(split_path.iterdir()) if p.is_dir()
    ]
    print(f"✅ Found {len(seqs)} sequences under [{mode_dir}]")

    if dataset_mode == Mode.TRAIN:
        augmentation_datapipe_type = partialclass(RandAugmentIterDataPipeKitti, dataset_config=dataset_config)
        return ConcatStreamingDataPipe(
            datapipe_list=seqs,
            batch_size=dataset_config.stream.train.batch_size,
            num_workers=dataset_config.stream.train.num_workers,
            augmentation_pipeline=augmentation_datapipe_type,
            print_seed_debug=False
        )
    else:
        fill_sample = seqs[0].get_fully_padded_sample()
        return ShardedStreamingDataPipe(
            datapipe_list=seqs,
            batch_size=dataset_config.stream.valid.batch_size,
            num_workers=dataset_config.stream.valid.num_workers,
            fill_value=fill_sample
        )


def partialclass(cls, *args, **kwargs):
    class NewCls(cls):
        __init__ = partialmethod(cls.__init__, *args, **kwargs)
    return NewCls