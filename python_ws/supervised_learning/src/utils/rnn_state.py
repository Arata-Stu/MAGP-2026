import torch
from typing import List, Optional, Union, Tuple, Dict



LstmState = Optional[Tuple[torch.Tensor, torch.Tensor]]
LstmStates = List[LstmState]

class RNNStates:
    def __init__(self):
        self.states = {}

    def _has_states(self):
        return len(self.states) > 0

    @classmethod
    def recursive_detach(cls, inp: Union[torch.Tensor, List, Tuple, Dict]):
        if isinstance(inp, torch.Tensor):
            return inp.detach()
        if isinstance(inp, list):
            return [cls.recursive_detach(x) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_detach(x) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_detach(v) for k, v in inp.items()}
        raise NotImplementedError

    @classmethod
    def recursive_reset(cls,
                        inp: Union[torch.Tensor, List, Tuple, Dict],
                        indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if isinstance(inp, torch.Tensor):
            assert inp.requires_grad is False, 'Not assumed here but should be the case.'
            if indices_or_bool_tensor is None:
                inp[:] = 0
            else:
                assert len(indices_or_bool_tensor) > 0
                inp[indices_or_bool_tensor] = 0
            return inp
        if isinstance(inp, list):
            return [cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp]
        if isinstance(inp, tuple):
            return tuple(cls.recursive_reset(x, indices_or_bool_tensor=indices_or_bool_tensor) for x in inp)
        if isinstance(inp, dict):
            return {k: cls.recursive_reset(v, indices_or_bool_tensor=indices_or_bool_tensor) for k, v in inp.items()}
        raise NotImplementedError

    def save_states_and_detach(self, worker_id: int, states: LstmStates) -> None:
        self.states[worker_id] = self.recursive_detach(states)

    def get_states(self, worker_id: int) -> Optional[LstmStates]:
        if not self._has_states():
            return None
        if worker_id not in self.states:
            return None
        return self.states[worker_id]

    def reset(self, worker_id: int, indices_or_bool_tensor: Optional[Union[List[int], torch.Tensor]] = None):
        if not self._has_states():
            return
        if worker_id in self.states:
            self.states[worker_id] = self.recursive_reset(
                self.states[worker_id], indices_or_bool_tensor=indices_or_bool_tensor)