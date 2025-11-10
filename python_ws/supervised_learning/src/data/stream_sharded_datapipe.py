from .custom_datapipes.iter import Concater, IterableWrapper, ZipperLongest
from .custom_datapipes.datapipe import IterDataPipe
import torch
from typing import List

class ShardedStreamingDataPipe(IterDataPipe):

    def __init__(self, seq_list: List, batch_size: int, num_workers: int, fill_value=None):
        super().__init__()
        self.seq_list = seq_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.fill_value = fill_value

        assert len(self.seq_list) > 0, "Empty sequence list is not allowed."
        assert self.batch_size > 0, "Batch size must be positive."
        assert len(self.seq_list) >= self.batch_size, (
            f"Each worker must have at least {self.batch_size} sequences, "
            f"but got only {len(self.seq_list)}. "
            "Otherwise, dynamic batching or empty Concater streams will occur. "
            "Decrease the number of workers or reduce batch_size."
        )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        wid = worker_info.id if worker_info else 0
        nworkers = worker_info.num_workers if worker_info else 1

        local_seqs = [s for i, s in enumerate(self.seq_list) if i % nworkers == wid]

        # 長い順に並べ替え
        local_seqs = sorted(local_seqs, key=lambda s: len(s), reverse=True)
        zipped_streams = []
        for i in range(self.batch_size):
            part = [local_seqs[j] for j in range(i, len(local_seqs), self.batch_size)]
            zipped_streams.append(Concater(*(p.to_iter_datapipe() for p in part)))
        zipped = ZipperLongest(*zipped_streams, fill_value=self.fill_value)
        return iter(zipped.zip(IterableWrapper([wid]).cycle(count=None)))
