from typing import Iterator, Sized, Tuple, TypeVar, Any, Optional, Set, List

from .._decorator import functional_datapipe
from ..datapipe import IterDataPipe

__all__ = [
    "ConcaterIterDataPipe",
    "ZipperIterDataPipe",
]

T_co = TypeVar('T_co', covariant=True)


@functional_datapipe('concat')
class ConcaterIterDataPipe(IterDataPipe):
    r"""
    Concatenates multiple Iterable DataPipes (functional name: ``concat``). The resulting DataPipe will
    yield all the elements from the first input DataPipe, before yielding from the subsequent ones.

    Args:
        datapipes: Iterable DataPipes being concatenated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> import random
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1 = IterableWrapper(range(3))
        >>> dp2 = IterableWrapper(range(5))
        >>> list(dp1.concat(dp2))
        [0, 1, 2, 0, 1, 2, 3, 4]
    """
    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if len(datapipes) == 0:
            raise ValueError("Expected at least one DataPipe, but got nothing")
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("Expected all inputs to be `IterDataPipe`")
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator:
        for dp in self.datapipes:
            for data in dp:
                yield data

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return sum(len(dp) for dp in self.datapipes)
        else:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))

@functional_datapipe('zip')
class ZipperIterDataPipe(IterDataPipe[Tuple[T_co]]):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip``).
    The output is stopped as soon as the shortest input DataPipe is exhausted.

    Args:
        *datapipes: Iterable DataPipes being aggregated

    Example:
        >>> # xdoctest: +REQUIRES(module:torchdata)
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(5)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (3, 13, 23), (4, 14, 24)]
    """
    datapipes: Tuple[IterDataPipe]

    def __init__(self, *datapipes: IterDataPipe):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` "
                            "for `ZipIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]

    def __iter__(self) -> Iterator[Tuple[T_co]]:
        iterators = [iter(datapipe) for datapipe in self.datapipes]
        yield from zip(*iterators)

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return min(len(dp) for dp in self.datapipes)
        else:
            raise TypeError("{} instance doesn't have valid length".format(type(self).__name__))


@functional_datapipe("zip_longest")
class ZipperLongestIterDataPipe(IterDataPipe):
    r"""
    Aggregates elements into a tuple from each of the input DataPipes (functional name: ``zip_longest``).
    The output is stopped until all input DataPipes are exhausted. If any input DataPipe is exhausted,
    missing values are filled-in with `fill_value` (default value is None).

    Args:
        *datapipes: Iterable DataPipes being aggregated
        *fill_value: Value that user input to fill in the missing values from DataPipe. Default value is None.

    Example:
        >>> from torchdata.datapipes.iter import IterableWrapper
        >>> dp1, dp2, dp3 = IterableWrapper(range(3)), IterableWrapper(range(10, 15)), IterableWrapper(range(20, 25))
        >>> list(dp1.zip_longest(dp2, dp3))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (None, 13, 23), (None, 14, 24)]
        >>> list(dp1.zip_longest(dp2, dp3, -1))
        [(0, 10, 20), (1, 11, 21), (2, 12, 22), (-1, 13, 23), (-1, 14, 24)]
    """
    datapipes: Tuple[IterDataPipe]
    length: Optional[int]
    fill_value: Any

    def __init__(
        self,
        *datapipes: IterDataPipe,
        fill_value: Any = None,
    ):
        if not all(isinstance(dp, IterDataPipe) for dp in datapipes):
            raise TypeError("All inputs are required to be `IterDataPipe` " "for `ZipperLongestIterDataPipe`.")
        super().__init__()
        self.datapipes = datapipes  # type: ignore[assignment]
        self.fill_value = fill_value

    def __iter__(self) -> Iterator[Tuple]:
        iterators = [iter(x) for x in self.datapipes]
        finished: Set[int] = set()
        while len(finished) < len(iterators):
            values: List[Any] = []
            for i in range(len(iterators)):
                value = self.fill_value
                if i not in finished:
                    try:
                        value = next(iterators[i])
                    except StopIteration:
                        finished.add(i)
                        if len(finished) == len(iterators):
                            return
                values.append(value)
            yield tuple(values)

    def __len__(self) -> int:
        if all(isinstance(dp, Sized) for dp in self.datapipes):
            return max(len(dp) for dp in self.datapipes)
        else:
            raise TypeError(f"{type(self).__name__} instance doesn't have valid length")
