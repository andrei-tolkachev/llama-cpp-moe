import argparse
from contextlib import redirect_stdout
import io
import os
import shutil
import struct
import sys
from unittest.mock import patch
import tempfile
from pathlib import Path
import unittest

TOOLS = Path(__file__).resolve().parents[1]/"tools"
sys.path.insert(0, str(TOOLS))
import r20_maya_ab as maya


class MayaProtocolTests(unittest.TestCase):
    def test_no_drafter_and_eog_are_kept(self):
        phase = ("prompt: 17 tokens in 70 ms (242.86 tok/s, windows of 64)\n"
                 "decode: 3 tokens in 1000 ms (2.00 tok/s); CPU lane: 10 of 40 experts, streamed to the GPU: 2\n")
        out = maya.parse_maya("7,5,1,\n7,5,1,\n", phase + "pass 2 of 2\n" + phase, 17, 256, False)
        self.assertTrue(out[1]["native_eog"])
        self.assertEqual(out[1]["tokens"], [7,5,1])
        self.assertEqual(out[1]["drafts_offered"], 0)
        self.assertEqual(out[1]["timings"]["predicted_per_second"], 2.0)

    def test_dspark_counts_measured_pass(self):
        template = ("prompt: 23 tokens in 20 ms (1150.00 tok/s, windows of 64)\n"
                    "dspark: 3 rounds, {} drafts offered, {} accepted (1.00 tokens a round)\n"
                    "decode: 3 tokens in 200 ms (10.00 tok/s); CPU lane: 1 of 2 experts, streamed to the GPU: 0\n")
        log = template.format(2,1) + "pass 2 of 2\n" + template.format(4,3)
        r = maya.parse_maya("7,8,9,\n7,8,9,\n", log, 23, 3, True)
        self.assertEqual(r[1]["drafts_offered"],4)
        self.assertEqual(r[1]["drafts_accepted"],3)
        with self.assertRaises(ValueError):
            maya.parse_maya("7,8,9,\n7,8,9,\n", log, 23, 3, False)
        with self.assertRaises(ValueError):
            maya.parse_maya("7,8,9,\n7,8,9,\n", log, 23, 4, True)

    def test_pack_checks_actual_tensor_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            alias, pack = root/"view", root/"pack"
            alias.mkdir()
            pack.mkdir()
            source = root/"real.gguf"
            source.touch()
            (alias/"real.gguf").symlink_to(source)
            (pack/"ds4.txt").write_text(f"format gguf\nmodel_dir {alias}\n")
            draft = root/"dspark.json"
            t = pack/"tensors.txt"
            t.write_text("file 0 real.gguf\nfile 1 ../dspark.bin\nt trunk.weight IQ3_XXS 0 45 20 1024\n"
                         "t mtp.0.test F16 1 12 3 4\n")
            self.assertTrue(maya.validate_pack(pack, alias, [source], draft))
            with self.assertRaises(ValueError):
                maya.validate_pack(pack, alias, [source], None)
            t.write_text(t.read_text().replace("mtp.0.test","trunk.engram"))
            with self.assertRaisesRegex(ValueError, "Trunk tensor"):
                maya.validate_pack(pack, alias, [source], draft)

    def test_llama_draft_requires_dflash_and_markov_metadata(self):
        def gguf_str(text):
            data = text.encode()
            return struct.pack("<Q",len(data)) + data
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "dsp.gguf"
            begin = b"GGUF" + struct.pack("<IQQ",3,1,1) + gguf_str("general.architecture") + struct.pack("<I",8)
            for arch, name, valid in [("dflash","markov_w1.weight",True),("dflash","w.weight",False),
                                      ("deepseek41","markov_w1.weight",False)]:
                path.write_bytes(begin + gguf_str(arch) + gguf_str(name) + struct.pack("<IQIQ",1,1,0,0))
                if valid:
                    self.assertEqual(maya.trained_draft_identity(path)["sha256"],maya.bench.file_hash(path))
                else:
                    with self.assertRaisesRegex(ValueError,"trained DSpark"):
                        maya.trained_draft_identity(path)

    def test_llama_plain_and_draft_runtime_flags_are_separate(self):
        with tempfile.TemporaryDirectory() as tmp:
            out=Path(tmp)
            exe=out/"llama-server"
            shutil.copyfile(TOOLS.parent/"tests/fixtures/r20_fake_server.py", exe)
            exe.chmod(0o700)
            model=out/"model.gguf"
            model.touch()
            draft=out/"dflash.gguf"
            draft.touch()
            args=argparse.Namespace(model=model, out=out, mtp=False, ctx=131072, ngl=18, threads=8,
                                    timeout=8, moe_cache_mib=12288, predict=256)
            with patch.dict(os.environ,R20_TEST_LOCK=str(out/"lock")), redirect_stdout(io.StringIO()), \
                    patch.object(maya.smoke,"gpu_snapshot",return_value={}):
                for role,d in [("plain",None), ("dspark",draft)]:
                    with maya.smoke.server(exe,role,{},args,dspark_gguf=d) as (_,launch):
                        cmd=launch["command"]
                        self.assertEqual(cmd[cmd.index("--spec-type")+1], "draft-dspark" if d else "none")
                        self.assertEqual("--spec-draft-model" in cmd, bool(d))
                        if d:
                            self.assertEqual(cmd[cmd.index("--spec-draft-model")+1],str(draft))
        self.assertEqual(maya.draft_counts({"draft_n":7,"draft_n_accepted":3},True)["drafts_offered"],7)
        self.assertEqual(maya.draft_counts({},True)["drafts_offered"],0)
        with self.assertRaises(ValueError):
            maya.draft_counts({"draft_n":7},False)
        with self.assertRaises(ValueError):
            maya.draft_counts({"draft_n":7,"draft_n_accepted":8},True)

    def test_prefix_and_length_divergence(self):
        self.assertEqual(maya.first_difference([1,2], [1,2,3]),2)
        self.assertEqual(maya.first_difference([1,2], [1,5]),1)
        self.assertIsNone(maya.first_difference([1,2],[1,2]))


if __name__ == "__main__":
    unittest.main()
