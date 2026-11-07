"""Diagnostic snapshots retain CPU counters when kernel policy hides per-process IO."""
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"tools"))
import r20_profile as profiler


class PartialProcAccessTests(unittest.TestCase):
    def test_denied_io_is_unknown_and_other_counters_remain_available(self):
        read_kv = profiler.read_kv
        def read(path):
            if path.name == "io":
                raise PermissionError(13, "test kernel policy", str(path))
            return read_kv(path)
        # Stable /stat fixture for environments whose /proc uses another PID namespace.
        with patch.object(Path, "read_text", return_value="42 (llama (test)) S 1 2 3 4 5 6 12 8 0 10 4 6 13 14"), \
                patch.object(profiler, "read_kv", side_effect=read):
            before, after = profiler.snap(os.getpid()), profiler.snap(os.getpid())
        delta = profiler.change(before, after)
        self.assertIsNone(delta["read_bytes"])
        self.assertIsNone(delta["write_bytes"])
        self.assertIn("test kernel policy", before["unavailable"]["io"])
        self.assertGreaterEqual(delta["cpu_s_all_threads"], 0)
        self.assertGreaterEqual(delta["major_faults"], 0)

    def test_unrelated_pid_not_profiled(self):
        with patch.object(Path, "read_bytes", return_value=b"/x/llama-server\0--alias\0different\0"), \
                patch.object(profiler, "read_kv", side_effect=AssertionError("unrelated /proc access")):
            snapshot = profiler.snap(12345, "expected")
        self.assertIn("pid_identity", snapshot["unavailable"])
        self.assertIsNone(profiler.change(snapshot, snapshot)["cpu_s_all_threads"])


if __name__ == "__main__":
    unittest.main()
