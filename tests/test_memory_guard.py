"""Inference must not require benchmark-specific kernel cgroup limits."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from memory_guard import HostMemory


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


if __name__ == '__main__':
    unittest.main()
