"""Fetch a handful of MNIST/CIFAR-10 samples straight into memory.

Both loaders in this module pull only as many bytes as needed for the
requested sample count, decode them with the standard library, and hand back
plain numpy arrays -- nothing is ever written to `data/` or any other local
path. This is the "no local dataset copy" alternative to the existing
torchvision/tensorflow_datasets loaders in main/run_cifar10_benchmark.py,
which download and cache the full dataset under `--data-dir`.

Both source hosts (the CVDF GCS mirror for MNIST, the University of Toronto
mirror for CIFAR-10) are the same ones torchvision/tensorflow_datasets use
internally; this module just reads their bytes directly instead of going
through a dataset-management library, and stops reading as soon as it has
enough samples rather than materializing the whole file/archive.
"""

from __future__ import annotations

import gzip
import io
import pickle
import struct
import tarfile

import numpy as np
import requests

MNIST_BASE_URL = "https://storage.googleapis.com/cvdf-datasets/mnist"
CIFAR10_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_MNIST_IMAGES_MAGIC = 2051
_MNIST_LABELS_MAGIC = 2049


def fetch_mnist_arrays(
    count: int, train: bool = True, timeout: float = 60.0
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first `count` (image, label) pairs from MNIST, fetched and
    decoded entirely in memory. Images are uint8 arrays shaped (count, 28, 28).

    Reads stop as soon as `count` samples have been decompressed, so this
    does not necessarily pull the entire (compressed) file over the network.
    """

    prefix = "train" if train else "t10k"
    images = _read_idx_images(f"{MNIST_BASE_URL}/{prefix}-images-idx3-ubyte.gz", count, timeout)
    labels = _read_idx_labels(f"{MNIST_BASE_URL}/{prefix}-labels-idx1-ubyte.gz", count, timeout)
    return images, labels


def _read_idx_images(url: str, count: int, timeout: float) -> np.ndarray:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as gz:
            magic, n_total, rows, cols = struct.unpack(">IIII", gz.read(16))
            if magic != _MNIST_IMAGES_MAGIC:
                raise ValueError(f"unexpected MNIST images magic number {magic} from {url}")
            n_read = min(count, n_total)
            payload = gz.read(n_read * rows * cols)
    return np.frombuffer(payload, dtype=np.uint8).reshape(n_read, rows, cols)


def _read_idx_labels(url: str, count: int, timeout: float) -> np.ndarray:
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with gzip.GzipFile(fileobj=response.raw) as gz:
            magic, n_total = struct.unpack(">II", gz.read(8))
            if magic != _MNIST_LABELS_MAGIC:
                raise ValueError(f"unexpected MNIST labels magic number {magic} from {url}")
            n_read = min(count, n_total)
            payload = gz.read(n_read)
    return np.frombuffer(payload, dtype=np.uint8)


class _ProgressReader(io.RawIOBase):
    """Wraps a raw file-like object, counting bytes read and calling
    `on_progress` every `report_every` bytes. Used so a slow multi-minute
    fetch isn't silent the whole way through. Implements the RawIOBase
    interface (readinto) that io.BufferedReader requires of its wrapped
    stream -- a plain object with only .read() is not enough."""

    def __init__(self, raw, on_progress, report_every: int = 1 << 20) -> None:
        super().__init__()
        self._raw = raw
        self._on_progress = on_progress
        self._report_every = report_every
        self._total = 0
        self._last_reported = 0

    def readable(self) -> bool:
        return True

    def readinto(self, buffer) -> int:
        chunk = self._raw.read(len(buffer))
        n = len(chunk)
        buffer[:n] = chunk
        self._total += n
        if self._on_progress is not None and self._total - self._last_reported >= self._report_every:
            self._last_reported = self._total
            self._on_progress(self._total)
        return n


def fetch_cifar10_arrays(
    count: int, timeout: float = 120.0, verbose: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """Return the first `count` (image, label) pairs from CIFAR-10's train
    split, fetched and decoded entirely in memory. Images are uint8 arrays
    shaped (count, 32, 32, 3).

    CIFAR-10 ships as five 10,000-image pickle batches inside one tarball.
    This streams the tar sequentially and stops as soon as enough batches
    have been read to cover `count`, closing the connection early rather
    than downloading the full ~170MB archive whenever count <= 10,000 * k
    for some k < 5. Set verbose=True to print download/decode progress,
    since this can take several minutes on a slow connection to the
    University of Toronto mirror.
    """

    images_batches: list[np.ndarray] = []
    labels_batches: list[np.ndarray] = []
    collected = 0

    def report(total_bytes: int) -> None:
        print(f"  fetch_cifar10_arrays: {total_bytes / 1e6:.1f} MB read so far...", flush=True)

    with requests.get(CIFAR10_URL, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        if verbose:
            print("fetch_cifar10_arrays: connected, reading tar stream...", flush=True)
        raw = _ProgressReader(response.raw, report if verbose else None)
        # tarfile reads its underlying file object in small (a few KB) chunks;
        # wrapping the raw socket stream in a large BufferedReader coalesces
        # those into far fewer, much larger network reads.
        buffered = io.BufferedReader(raw, buffer_size=1 << 20)
        with tarfile.open(fileobj=buffered, mode="r|gz") as tar:
            for member in tar:
                if collected >= count:
                    break
                name = member.name.rsplit("/", 1)[-1]
                if not name.startswith("data_batch_"):
                    continue
                extracted = tar.extractfile(member)
                if extracted is None:
                    continue
                if verbose:
                    print(f"fetch_cifar10_arrays: found {name}, decoding...", flush=True)
                batch = pickle.load(extracted, encoding="bytes")
                batch_images = (
                    batch[b"data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.uint8)
                )
                batch_labels = np.array(batch[b"labels"], dtype=np.uint8)
                images_batches.append(batch_images)
                labels_batches.append(batch_labels)
                collected += batch_images.shape[0]
                if verbose:
                    print(f"fetch_cifar10_arrays: collected {collected}/{count} images", flush=True)

    images = np.concatenate(images_batches, axis=0)[:count]
    labels = np.concatenate(labels_batches, axis=0)[:count]
    return images, labels
