import gzip
import io
import pickle
import struct
import tarfile
import unittest
from unittest import mock

import numpy as np

from src import streaming_datasets


class _FakeStreamResponse:
    """Minimal stand-in for requests.Response, streaming-mode only."""

    def __init__(self, payload: bytes) -> None:
        self.raw = io.BytesIO(payload)
        self.status_code = 200

    def raise_for_status(self) -> None:
        return None

    def __enter__(self) -> "_FakeStreamResponse":
        return self

    def __exit__(self, *exc_info) -> bool:
        return False


def _make_mnist_images_gz(images: np.ndarray) -> bytes:
    n, rows, cols = images.shape
    header = struct.pack(">IIII", 2051, n, rows, cols)
    return gzip.compress(header + images.astype(np.uint8).tobytes())


def _make_mnist_labels_gz(labels: np.ndarray) -> bytes:
    header = struct.pack(">II", 2049, labels.shape[0])
    return gzip.compress(header + labels.astype(np.uint8).tobytes())


def _make_cifar_batch_member(name: str, images: np.ndarray, labels: np.ndarray) -> bytes:
    flat = images.transpose(0, 3, 1, 2).reshape(images.shape[0], -1)
    payload = pickle.dumps({b"data": flat.astype(np.uint8), b"labels": labels.tolist()})
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        info = tarfile.TarInfo(name=name)
        info.size = len(payload)
        tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


class StreamingDatasetsTests(unittest.TestCase):
    def test_fetch_mnist_arrays_decodes_requested_count(self) -> None:
        rng = np.random.default_rng(0)
        images = rng.integers(0, 256, size=(30, 28, 28), dtype=np.uint8)
        labels = rng.integers(0, 10, size=30, dtype=np.uint8)
        images_gz = _make_mnist_images_gz(images)
        labels_gz = _make_mnist_labels_gz(labels)

        def fake_get(url: str, stream: bool = True, timeout: float = 60.0):
            if "images" in url:
                return _FakeStreamResponse(images_gz)
            return _FakeStreamResponse(labels_gz)

        with mock.patch.object(streaming_datasets.requests, "get", side_effect=fake_get):
            fetched_images, fetched_labels = streaming_datasets.fetch_mnist_arrays(10, train=True)

        self.assertEqual(fetched_images.shape, (10, 28, 28))
        np.testing.assert_array_equal(fetched_images, images[:10])
        np.testing.assert_array_equal(fetched_labels, labels[:10])

    def test_fetch_mnist_arrays_rejects_bad_magic(self) -> None:
        bad_gz = gzip.compress(struct.pack(">IIII", 9999, 5, 28, 28) + bytes(5 * 28 * 28))

        with mock.patch.object(streaming_datasets.requests, "get", return_value=_FakeStreamResponse(bad_gz)):
            with self.assertRaises(ValueError):
                streaming_datasets.fetch_mnist_arrays(5, train=True)

    def test_fetch_cifar10_arrays_decodes_single_batch(self) -> None:
        rng = np.random.default_rng(1)
        images = rng.integers(0, 256, size=(40, 32, 32, 3), dtype=np.uint8)
        labels = rng.integers(0, 10, size=40, dtype=np.uint8)
        tar_bytes = _make_cifar_batch_member("cifar-10-batches-py/data_batch_1", images, labels)

        with mock.patch.object(streaming_datasets.requests, "get", return_value=_FakeStreamResponse(tar_bytes)):
            fetched_images, fetched_labels = streaming_datasets.fetch_cifar10_arrays(15)

        self.assertEqual(fetched_images.shape, (15, 32, 32, 3))
        np.testing.assert_array_equal(fetched_images, images[:15])
        np.testing.assert_array_equal(fetched_labels, labels[:15])

    def test_fetch_cifar10_arrays_stops_after_enough_batches(self) -> None:
        rng = np.random.default_rng(2)
        batch1_images = rng.integers(0, 256, size=(20, 32, 32, 3), dtype=np.uint8)
        batch1_labels = rng.integers(0, 10, size=20, dtype=np.uint8)

        payload1 = pickle.dumps(
            {
                b"data": batch1_images.transpose(0, 3, 1, 2).reshape(20, -1).astype(np.uint8),
                b"labels": batch1_labels.tolist(),
            }
        )
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
            info = tarfile.TarInfo(name="cifar-10-batches-py/data_batch_1")
            info.size = len(payload1)
            tar.addfile(info, io.BytesIO(payload1))

            def _poison(*args, **kwargs):
                raise AssertionError("should not read past the first batch")

            poison_info = tarfile.TarInfo(name="cifar-10-batches-py/data_batch_2")
            poison_info.size = 4
            tar.addfile(poison_info, io.BytesIO(b"\x00\x00\x00\x00"))

        with mock.patch.object(streaming_datasets.requests, "get", return_value=_FakeStreamResponse(buffer.getvalue())):
            fetched_images, fetched_labels = streaming_datasets.fetch_cifar10_arrays(10)

        self.assertEqual(fetched_images.shape, (10, 32, 32, 3))
        np.testing.assert_array_equal(fetched_images, batch1_images[:10])


if __name__ == "__main__":
    unittest.main()
