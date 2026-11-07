import importlib.util
from pathlib import Path
import unittest

spec=importlib.util.spec_from_file_location('auto_memory',Path(__file__).resolve().parents[1]/'tools/auto_memory.py')
m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

class MemoryPlanningTests(unittest.TestCase):
    def test_request_boundary_preserves_native_limit(self):
        self.assertEqual(m.context_for(4097,262144,131072),8192)
        self.assertEqual(m.context_for(262144,262144,131072),262144)
        with self.assertRaises(ValueError):m.context_for(262145,262144,131072)
    def test_missing_request_preserves_profile_default(self):
        self.assertEqual(m.context_for(None,262144,131072),131072)
    def test_cache_respects_reserve_and_profile_cap(self):
        self.assertEqual(m.choose_cache(20992,30000,2048,lambda x:13000+x),14848)
        self.assertEqual(m.choose_cache(20992,32700,2048,lambda x:9000+x),20992)
    def test_no_silent_context_or_precision_fallback(self):
        with self.assertRaises(ValueError):m.choose_cache(20992,30000,2048,lambda x:40000+x)
    def test_decode_retains_large_prefill_state(self):
        decode=dict(model_mib=8000,context_mib=21000,compute_mib=60)
        prefill=dict(model_mib=8000,context_mib=22000,compute_mib=12000)
        self.assertEqual(m.phase_memory(decode,prefill),30060)
        self.assertEqual(m.phase_memory(prefill,decode),42000)

    def test_native_rows_reject_ambiguous_device_mapping(self):
        self.assertEqual(m.memory_rows('CUDA0 8000 21000 60\nHost 300000 32 10\n')['CUDA0']['context_mib'],21000)
        with self.assertRaises(ValueError):m.memory_rows('CUDA0 1 2 3\nCUDA1 1 2 3\nHost 1 2 3\n')
        with self.assertRaises(ValueError):m.memory_rows('CUDA0 1 2 3\nCUDA0 1 2 3\nHost 1 2 3\n')

if __name__=='__main__':unittest.main()
