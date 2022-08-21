import copy
import datetime
import hashlib
import itertools
import json
import logging
import os
import pathlib
import random
from itertools import islice
from typing import Iterable, List, Optional, Tuple

import numpy as np
import simplejson
import torch

logger = logging.getLogger(__name__)


def grouper(iterable, n, worker_id=0, total_workers=1):
    """
    Split an iterable into a list of n-sized chunks. Each element in the chunk is a tuple of (index_num, element).

    Example:

    >>> list(grouper('ABCDEFG', 3))
    [[(0, 'A'), (1, 'B'), (2, 'C')], [(3, 'D'), (4, 'E'), (5, 'F')], [(6, 'G')]]

    Input --> [dictA, dictB, dictC, dictD, dictE, ...] with total worker=3 and n=2

    Output for worker 1: [(dictA, dictB), (dictG, dictH), ...]
    Output for worker 2: [(dictC, dictD), (dictI, dictJ), ...]
    Output for worker 3: [(dictE, dictF), (dictK, dictL), ...]

    This method also adds an index number to every dict yielded.

    :param iterable: a generator object that yields dicts
    :type iterable: generator
    :param n: the dicts are grouped in n-sized chunks that gets converted to datasets
    :type n: int
    :param worker_id: the worker_id for the PyTorch DataLoader
    :type worker_id: int
    :param total_workers: total number of workers for the PyTorch DataLoader
    :type total_workers: int
    """
    # TODO make me comprehensible :)
    def get_iter_start_pos(gen):
        start_pos = worker_id * n
        for i in gen:
            if start_pos:
                start_pos -= 1
                continue
            yield i

    def filter_elements_per_worker(gen):
        x = n
        y = (total_workers - 1) * n
        for i in gen:
            if x:
                yield i
                x -= 1
            else:
                if y != 1:
                    y -= 1
                    continue
                else:
                    x = n
                    y = (total_workers - 1) * n

    iterable = iter(enumerate(iterable))
    iterable = get_iter_start_pos(iterable)
    if total_workers > 1:
        iterable = filter_elements_per_worker(iterable)

    return iter(lambda: list(islice(iterable, n)), [])


def is_json(x):
    if issubclass(type(x), pathlib.Path):
        return True
    try:
        simplejson.dumps(x)
        return True
    except:
        return False


def flatten_list(nested_list):
    """Flatten an arbitrarily nested list, without recursion (to avoid
    stack overflows). Returns a new list, the original list is unchanged.
    >> list(flatten_list([1, 2, 3, [4], [], [[[[[[[[[5]]]]]]]]]]))
    [1, 2, 3, 4, 5]
    >> list(flatten_list([[1, 2], 3]))
    [1, 2, 3]
    """
    nested_list = copy.deepcopy(nested_list)

    while nested_list:
        sublist = nested_list.pop(0)

        if isinstance(sublist, list):
            nested_list = sublist + nested_list
        else:
            yield sublist


def initialize_device_settings(
    use_cuda: Optional[bool] = None,
    local_rank: int = -1,
    multi_gpu: bool = True,
    devices: Optional[List[torch.device]] = None,
) -> Tuple[List[torch.device], int]:
    """
    Returns a list of available devices.

    :param use_cuda: Whether to make use of CUDA GPUs (if available).
    :param local_rank: Ordinal of device to be used. If -1 and `multi_gpu` is True, all devices will be used.
                       Unused if `devices` is set or `use_cuda` is False.
    :param multi_gpu: Whether to make use of all GPUs (if available).
                      Unused if `devices` is set or `use_cuda` is False.
    :param devices: an explicit list of which GPUs to use. Unused if `use_cuda` is False.
    """
    if use_cuda is False:  # Note that it could be None, in which case we also want to just skip this step.
        devices_to_use = [torch.device("cpu")]
        n_gpu = 0
    elif devices:
        devices_to_use = devices
        n_gpu = sum(1 for device in devices if "cpu" not in device.type)
    elif local_rank == -1:
        if torch.cuda.is_available():
            if multi_gpu:
                devices_to_use = [torch.device(device) for device in range(torch.cuda.device_count())]
                n_gpu = torch.cuda.device_count()
            else:
                devices_to_use = [torch.device("cuda")]
                n_gpu = 1
        else:
            devices_to_use = [torch.device("cpu")]
            n_gpu = 0
    else:
        devices_to_use = [torch.device("cuda", local_rank)]
        torch.cuda.set_device(devices_to_use[0])
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")
    logger.info(f"Using devices: {', '.join([str(device) for device in devices_to_use]).upper()}")
    logger.info(f"Number of GPUs: {n_gpu}")
    return devices_to_use, n_gpu


def convert_date_to_rfc3339(date: str) -> str:
    """
    Converts a date to RFC3339 format, as Weaviate requires dates to be in RFC3339 format including the time and
    timezone.

    If the provided date string does not contain a time and/or timezone, we use 00:00 as default time
    and UTC as default time zone.

    This method cannot be part of WeaviateDocumentStore, as this would result in a circular import between weaviate.py
    and filter_utils.py.
    """
    parsed_datetime = datetime.fromisoformat(date)
    if parsed_datetime.utcoffset() is None:
        converted_date = parsed_datetime.isoformat() + "Z"
    else:
        converted_date = parsed_datetime.isoformat()

    return converted_date


def get_dict_checksum(payload_dict):
    """
    Get MD5 checksum for a dict.
    """
    checksum = hashlib.md5(json.dumps(payload_dict, sort_keys=True).encode("utf-8-sig")).hexdigest()
    return checksum


import pickle

import torch
import torch.distributed as dist

# DDP utils


def all_reduce(tensor, group=None):
    if group is None:
        group = dist.group.WORLD
    return dist.all_reduce(tensor, group=group)


def all_gather_list(data, group=None, max_size=16384):
    """Gathers arbitrary data from all nodes into a list.
    Similar to :func:`~torch.distributed.all_gather` but for arbitrary Python
    data. Note that *data* must be picklable.
    Args:
        data (Any): data from the local worker to be gathered on other workers
        group (optional): group of the collective
    """
    SIZE_STORAGE_BYTES = 4  # int32 to encode the payload size

    enc = pickle.dumps(data)
    enc_size = len(enc)

    if enc_size + SIZE_STORAGE_BYTES > max_size:
        raise ValueError("encoded data exceeds max_size, this can be fixed by increasing buffer size: {}".format(enc_size))

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    buffer_size = max_size * world_size

    if not hasattr(all_gather_list, "_buffer") or all_gather_list._buffer.numel() < buffer_size:
        all_gather_list._buffer = torch.cuda.ByteTensor(buffer_size)
        all_gather_list._cpu_buffer = torch.ByteTensor(max_size).pin_memory()

    buffer = all_gather_list._buffer
    buffer.zero_()
    cpu_buffer = all_gather_list._cpu_buffer

    assert enc_size < 256**SIZE_STORAGE_BYTES, "Encoded object size should be less than {} bytes".format(
        256**SIZE_STORAGE_BYTES
    )

    size_bytes = enc_size.to_bytes(SIZE_STORAGE_BYTES, byteorder="big")

    cpu_buffer[0:SIZE_STORAGE_BYTES] = torch.ByteTensor(list(size_bytes))
    cpu_buffer[SIZE_STORAGE_BYTES : enc_size + SIZE_STORAGE_BYTES] = torch.ByteTensor(list(enc))

    start = rank * max_size
    size = enc_size + SIZE_STORAGE_BYTES
    buffer[start : start + size].copy_(cpu_buffer[:size])

    all_reduce(buffer, group=group)

    try:
        result = []
        for i in range(world_size):
            out_buffer = buffer[i * max_size : (i + 1) * max_size]
            size = int.from_bytes(out_buffer[0:SIZE_STORAGE_BYTES], byteorder="big")
            if size > 0:
                result.append(pickle.loads(bytes(out_buffer[SIZE_STORAGE_BYTES : size + SIZE_STORAGE_BYTES].tolist())))
        return result
    except pickle.UnpicklingError:
        raise Exception(
            "Unable to unpickle data from other workers. all_gather_list requires all "
            "workers to enter the function together, so this error usually indicates "
            "that the workers have fallen out of sync somehow. Workers can fall out of "
            "sync if one of them runs out of memory, or if there are other conditions "
            "in your training script that can cause one worker to finish an epoch "
            "while other workers are still iterating over their portions of the data."
        )


def set_all_seeds(seed, deterministic_cudnn=False):
    """
    Setting multiple seeds to make runs reproducible.

    Important: Enabling `deterministic_cudnn` gives you full reproducibility with CUDA,
    but might slow down your training (see https://pytorch.org/docs/stable/notes/randomness.html#cudnn) !

    :param seed:number to use as seed
    :type seed: int
    :param deterministic_torch: Enable for full reproducibility when using CUDA. Caution: might slow down training.
    :type deterministic_cudnn: bool
    :return: None
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_iterator(data: Iterable, in_notebook: bool = False, show_bar: bool = True):
    if not show_bar:
        return data
    if not in_notebook:
        from tqdm import tqdm

        return tqdm(data)
    else:
        import tqdm.notebook as tq

        return tq.tqdm(data)


def get_batches_from_generator(iterable, n):
    """
    Batch elements of an iterable into fixed-length chunks or blocks.
    """
    it = iter(iterable)
    x = tuple(itertools.islice(it, n))
    while x:
        yield x
        x = tuple(itertools.islice(it, n))
