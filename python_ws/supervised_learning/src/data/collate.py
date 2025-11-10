from typing import Any, Dict, List
import torch

def collate_sequence_batch(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    assert isinstance(batch, (list, tuple)) and len(batch) > 0
    assert isinstance(batch[0], dict), f"Expected dict, got {type(batch[0])}"

    result: Dict[str, Any] = {}

    for key in batch[0].keys():
        values = [b[key] for b in batch]

        if key in ("is_padded_mask", "pad_mask"):
            flat = []
            for seq_list in values:
                if isinstance(seq_list, list):
                    for v in seq_list:
                        if not torch.is_tensor(v):
                            v = torch.tensor(v)
                        flat.append(v.view(-1))
                else:
                    v = seq_list
                    if not torch.is_tensor(v):
                        v = torch.tensor(v)
                    flat.append(v.view(-1))
            stacked = torch.stack(flat).view(len(values), -1)
            result[key] = stacked.bool()
            continue

        first_val = values[0]

        if isinstance(first_val, list) and len(first_val) > 0 and torch.is_tensor(first_val[0]):
            result[key] = [torch.stack(v, dim=0) for v in zip(*values)]

        elif isinstance(first_val, list) and len(first_val) > 0 and isinstance(first_val[0], list):
            result[key] = [list(v) for v in zip(*values)]

        elif torch.is_tensor(first_val):
            result[key] = torch.stack(values, dim=0)

        else:
            try:
                result[key] = torch.tensor(values)
            except (TypeError, ValueError):
                result[key] = values

    return result


def custom_collate_streaming(batch: Any) -> Dict[str, Any]:
    samples, worker_info = batch

    assert isinstance(samples, (list, tuple)), "Expected samples to be a list/tuple"
    assert isinstance(worker_info, int), "Expected worker_info to be an int"

    if isinstance(worker_info, dict):
        worker_id = worker_info.get("worker_id", -1)
    elif isinstance(worker_info, int):
        worker_id = worker_info
    else:
        raise AssertionError(f"Unexpected worker_info type: {type(worker_info)}")

    if isinstance(samples, dict):
        samples = [samples]

    assert isinstance(samples, (list, tuple)) and len(samples) > 0, "Empty samples in batch"
    assert isinstance(samples[0], dict), f"Expected dict samples, got {type(samples[0])}"

    return {
        "data": collate_sequence_batch(samples),
        "worker_id": worker_id,
    }


def custom_collate_rnd(batch: Any) -> Dict[str, Any]:
    samples = batch
    worker_info = torch.utils.data.get_worker_info()
    local_worker_id = 0 if worker_info is None else worker_info.id

    return {
        "data": collate_sequence_batch(samples),
        "worker_id": local_worker_id,
    }