import torch
from torch.utils.data import Sampler

from typing import Iterable, Iterator, List, Sized, Union
import random


class LengthBucketRandomSampler(Sampler[int]):
    data_source: Sized
    bucket_size: int

    def __init__(
        self,
        data_source: Sized,
        bucket_size: int,
        len_fn,
        generator=None,
    ) -> None:
        super().__init__(self)
        self.data_source = data_source
        self.bucket_size = bucket_size
        self.len_fn = len_fn
        self.generator = generator
        self.len_idx = sorted(
            [(self.len_fn(self.data_source[idx]), idx) for idx in range(len(self.data_source))]
        )

    # @property
    # def num_samples(self) -> int:
    #     # dataset size might change at runtime
    #     return len(self.data_source)

    def __iter__(self) -> Iterator[int]:
        # print(f"LengthBucketRandomSampler.__iter__()")
        if self.generator is None:
            seed = int(torch.empty((), dtype=torch.int64).random_().item())
            generator = torch.Generator()
            generator.manual_seed(seed)
        else:
            generator = self.generator

        print(f"Bucketizing {len(self.data_source)} samples")
        pos = 0
        while pos < len(self.data_source):
            pos2 = min(pos + self.bucket_size, len(self.data_source))
            bucket = self.len_idx[pos:pos2]
            random.shuffle(bucket)
            # print(f"Min {min([x[0] for x in bucket])} Max {max([x[0] for x in bucket])}")
            yield from [x[1] for x in bucket]
            pos = pos2

    def __len__(self) -> int:
        # return self.num_samples
        return len(self.data_source)


class RandomBatchSampler(Sampler[List[int]]):
    def __init__(
        self,
        sampler: Union[Sampler[int], Iterable[int]],
        batch_size: int,
        drop_last: bool,
    ) -> None:
        # Since collections.abc.Iterable does not check for `__getitem__`, which
        # is one way for an object to be an iterable, we don't do an `isinstance`
        # check here.
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError(
                "batch_size should be a positive integer value, "
                "but got batch_size={}".format(batch_size)
            )
        if not isinstance(drop_last, bool):
            raise ValueError(
                "drop_last should be a boolean value, but got " "drop_last={}".format(drop_last)
            )
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    def __iter__(self) -> Iterator[List[int]]:
        idx_list = list(self.sampler)
        batch_list = []
        pos = 0
        while pos + self.batch_size < len(idx_list):
            pos2 = pos + self.batch_size
            batch_list.append(idx_list[pos:pos2])
            pos = pos2
        if not self.drop_last:
            batch_list.append(idx_list[pos:])

        print(f"Shuffling {len(batch_list)} batches")
        random.shuffle(batch_list)
        yield from batch_list

    def __len__(self) -> int:
        # Can only be called if self.sampler has __len__ implemented
        # We cannot enforce this condition, so we turn off typechecking for the
        # implementation below.
        # Somewhat related: see NOTE [ Lack of Default `__len__` in Python Abstract Base Classes ]
        if self.drop_last:
            return len(self.sampler) // self.batch_size  # type: ignore[arg-type]
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size  # type: ignore[arg-type]
