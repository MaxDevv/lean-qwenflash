"""Inference must not require benchmark-specific kernel cgroup limits."""
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from memory_guard import DeviceMemory, HostMemory


class HostMemoryTests(unittest.TestCase):
    def test_ordinary_cgroup_configurations(self):
        for cap, swap in [('max', 'max'), ('32000000000', '0'), ('8000000000', '1000000')]:
            with self.subTest(cap=cap, swap=swap), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                (root / 'memory.max').write_text(cap)
                (root / 'memory.swap.max').write_text(swap)
                (root / 'memory.current').write_text('123')
                monitor = HostMemory(root, limit=1000)
                try:
                    self.assertEqual(monitor.sample(), 123)
                    self.assertEqual(monitor.peak, 123)
                finally:
                    monitor.close()

    def test_runtime_limit_is_still_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / 'memory.current').write_text('1001')
            result = subprocess.run(
                [sys.executable, '-c',
                 'from memory_guard import HostMemory; import sys,time; '
                 'HostMemory(sys.argv[1],limit=1000); time.sleep(2)', directory],
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True, text=True, timeout=10,
            )
            self.assertEqual(result.returncode, 87, result.stderr)
            self.assertIn('RAM LIMIT EXCEEDED', result.stdout)

    def test_without_a_memory_cgroup_the_guard_is_off(self):
        # e.g. WSL's root cgroup: nothing to measure, so no sampler thread to crash
        with tempfile.TemporaryDirectory() as directory:
            monitor = HostMemory(directory, limit=1000)
            self.assertFalse(monitor.running)
            self.assertEqual(monitor.sample(), 0)
            monitor.close()


class DeviceMemoryTests(unittest.TestCase):
    def limit(self, env):
        nvml = mock.MagicMock()
        nvml.nvmlInit_v2.return_value = nvml.nvmlDeviceGetHandleByIndex_v2.return_value = 0
        with mock.patch('ctypes.CDLL', return_value=nvml), mock.patch('threading.Thread'), \
                mock.patch.dict(os.environ, env, clear=False):
            for key in ('QWEN_VRAM_LIMIT_BYTES', 'QWEN_VRAM_GUARD_BYTES'):
                if key not in env:
                    os.environ.pop(key, None)
            return DeviceMemory().limit

    def test_the_guard_follows_the_vram_limit_unless_set(self):
        self.assertEqual(self.limit({}), 16_000_000_000)
        self.assertEqual(self.limit({'QWEN_VRAM_LIMIT_BYTES': '30000000000'}), 30_000_000_000)
        self.assertEqual(self.limit({'QWEN_VRAM_LIMIT_BYTES': '30000000000', 'QWEN_VRAM_GUARD_BYTES': '20000000000'}),
                         20_000_000_000)


if __name__ == '__main__':
    unittest.main()
