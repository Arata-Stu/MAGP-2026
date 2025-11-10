import torch as th
import torch.distributed as dist
from torch.utils.data import get_worker_info
from .custom_datapipes.iter import Zipper, Concater, IterableWrapper
from .custom_datapipes.datapipe import MapDataPipe, IterDataPipe
from typing import Any, Iterator, List, Optional, Type


class DummyIterDataPipe(IterDataPipe):
    def __init__(self, source_dp: IterDataPipe):
        super().__init__()
        assert isinstance(source_dp, IterDataPipe)
        self.source_dp = source_dp

    def __iter__(self):
        yield from self.source_dp


class ConcatStreamingDataPipe(IterDataPipe):

    def __init__(self,
                 datapipe_list: List[MapDataPipe],
                 batch_size: int,
                 num_workers: int,
                 augmentation_pipeline: Optional[Type[IterDataPipe]] = None,
                 print_seed_debug: bool = False):
        super().__init__()
        assert batch_size > 0

        self.datapipe_list = datapipe_list
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.print_seed_debug = print_seed_debug
        self.augmentation_dp = augmentation_pipeline or DummyIterDataPipe

    @staticmethod
    def random_torch_shuffle_list(data: List[Any]) -> Iterator[Any]:
        assert isinstance(data, list)
        return (data[i] for i in th.randperm(len(data)).tolist())

    def _get_zipped_streams(self, datapipe_list: List[MapDataPipe], batch_size: int):
        streams = Zipper(*(
            Concater(*(self.augmentation_dp(x.to_iter_datapipe())
                        for x in self.random_torch_shuffle_list(datapipe_list)))
            for _ in range(batch_size)
        ))
        return streams

    def _print_seed_debug_info(self):
        worker_info = get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        worker_torch_seed = worker_info.seed
        local_num_workers = 1 if worker_info is None else worker_info.num_workers
        global_rank = dist.get_rank() if (dist.is_available() and dist.is_initialized()) else 0
        global_worker_id = global_rank * local_num_workers + local_worker_id
        rnd_number = th.randn(1)
        print(f'{worker_torch_seed=}, {global_worker_id=}, {global_rank=}, {local_worker_id=}, {rnd_number=}',
              flush=True)

    def _get_zipped_streams_with_worker_id(self):
        worker_info = get_worker_info()
        local_worker_id = 0 if worker_info is None else worker_info.id
        worker_id_stream = IterableWrapper([local_worker_id]).cycle(count=None)
        zipped_stream = self._get_zipped_streams(self.datapipe_list, self.batch_size)
        return zipped_stream.zip(worker_id_stream)

    def __iter__(self):
        if self.print_seed_debug:
            self._print_seed_debug_info()
        return iter(self._get_zipped_streams_with_worker_id())
