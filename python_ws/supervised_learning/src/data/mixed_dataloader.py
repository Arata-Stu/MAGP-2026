import torch
from torch.utils.data import DataLoader
from omegaconf import DictConfig

from src.data.types import Mode
from src.data.random_dataset import build_random_access_dataset
from src.data.stream_dataset import build_stream_dataset
from src.data.collate import custom_collate_rnd, custom_collate_streaming

def concat_mixed_batches(batch1, batch2):
    assert batch1.keys() == batch2.keys(), f"キー構造が異なります: {batch1.keys()} vs {batch2.keys()}"

    out = {}
    for key in batch1.keys():
        v1, v2 = batch1[key], batch2[key]
        if isinstance(v1, torch.Tensor):
            out[key] = torch.cat((v1, v2))
        elif isinstance(v1, dict):
            out[key] = concat_mixed_batches(v1, v2)
        elif isinstance(v1, list):
            assert len(v1) == len(v2)
            out[key] = [concat_mixed_batches(a, b) if isinstance(a, dict) else torch.cat((a, b)) for a, b in zip(v1, v2)]
        else:
            # worker_id のような数値は stream 側を優先する
            out[key] = v1
    return out


class MixedDataLoader:
    """
    データセットの構築とDataLoaderの管理をまとめて行うクラス。
    Random (Map) と Stream (Iterable) の2つを同時に処理する。
    """
    def __init__(self, 
                 dataset_config: DictConfig,
                 reader_cls: type,
                 dataset_mode: Mode = Mode.TRAIN):
        
        self.dataset_config = dataset_config
        self.reader_cls = reader_cls
        self.dataset_mode = dataset_mode

        print("🔹 [MixedDataLoader] Building Random Access Dataset...")
        self.rnd_dataset = build_random_access_dataset(
            dataset_mode=self.dataset_mode,
            dataset_config=self.dataset_config,
            reader_cls=self.reader_cls,
        )

        print("🔹 [MixedDataLoader] Building Stream Dataset...")
        # Streamデータセット構築時にバッチサイズとワーカー数を渡す
        self.stream_dataset = build_stream_dataset(
            dataset_mode=self.dataset_mode,
            dataset_config=self.dataset_config,
            reader_cls=self.reader_cls,
        )
        
        self.random_loader = DataLoader(
            self.rnd_dataset,
            batch_size=self.dataset_config.random.train.batch_size,
            shuffle=(self.dataset_mode == Mode.TRAIN),
            collate_fn=custom_collate_rnd,
            num_workers=self.dataset_config.random.train.num_workers,
            persistent_workers=self.dataset_config.random.train.num_workers > 0,
            pin_memory=False, 
        )
        
        self.stream_loader = DataLoader(
            self.stream_dataset,
            batch_size=None,  
            collate_fn=custom_collate_streaming,
            num_workers=self.stream_dataset.num_workers,
            persistent_workers=self.stream_dataset.num_workers > 0,
        )
        
        print(f"\n✅ MixedDataLoader initialized.")
        print(f"   - Random Access Dataset size: {len(self.rnd_dataset)} samples")
        print(f"   - Stream Dataset ready for iteration.")

    def __iter__(self):
        self.random_iter = iter(self.random_loader)
        self.stream_iter = iter(self.stream_loader)
        return self

    def __next__(self):
        try:
            random_batch = next(self.random_iter)
        except StopIteration:
            raise StopIteration

        try:
            stream_batch = next(self.stream_iter)
        except StopIteration:
            print("Warning: Stream iterator re-initialized within an epoch.")
            self.stream_iter = iter(self.stream_loader)
            stream_batch = next(self.stream_iter)

        return {
            "random": random_batch,
            "stream": stream_batch
        }

    def __len__(self):
        return len(self.random_loader)