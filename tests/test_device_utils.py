import argparse
import unittest

from main.device_utils import add_torch_device_argument, resolve_torch_device


class FakeCuda:
    def __init__(self, available: bool, count: int = 1) -> None:
        self._available = available
        self._count = count

    def is_available(self) -> bool:
        return self._available

    def device_count(self) -> int:
        return self._count


class FakeTorch:
    def __init__(self, available: bool, count: int = 1) -> None:
        self.cuda = FakeCuda(available, count)

    def device(self, name: str) -> str:
        return name


class TorchDeviceUtilsTests(unittest.TestCase):
    def test_auto_prefers_cuda_when_available(self) -> None:
        self.assertEqual(resolve_torch_device("auto", FakeTorch(True)), "cuda")

    def test_auto_falls_back_to_cpu(self) -> None:
        self.assertEqual(resolve_torch_device("auto", FakeTorch(False)), "cpu")

    def test_explicit_cuda_requires_available_cuda(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot access CUDA"):
            resolve_torch_device("cuda", FakeTorch(False))

    def test_explicit_cuda_index_is_checked(self) -> None:
        with self.assertRaisesRegex(ValueError, "only 1 CUDA device"):
            resolve_torch_device("cuda:1", FakeTorch(True, count=1))

    def test_parser_gets_device_argument(self) -> None:
        parser = argparse.ArgumentParser()
        add_torch_device_argument(parser)
        args = parser.parse_args(["--device", "cuda:0"])
        self.assertEqual(args.device, "cuda:0")


if __name__ == "__main__":
    unittest.main()
