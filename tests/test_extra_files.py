"""Prove vendor additions remain bounded, hashed and isolated."""
from pathlib import Path
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import unittest

ROOT=Path(__file__).resolve().parents[1]
TOOL=ROOT/'tools/overlay.py'

def sha(data):return hashlib.sha256(data).hexdigest()

class VendorFiles(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='llama-extra-files-')
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.source=self.root/'upstream'
        self.source.mkdir()
        self.env=dict(os.environ,GIT_CONFIG_NOSYSTEM='1',GIT_CONFIG_GLOBAL=os.devnull,GIT_OPTIONAL_LOCKS='0')
        self.git(self.source,'init','-q')
        self.git(self.source,'config','user.email','test@example.invalid')
        self.git(self.source,'config','user.name','test')
        (self.source/'src').mkdir()
        self.target=self.source/'src/target.cpp'
        self.target.write_text('int foo() { return 1; }\n')
        self.git(self.source,'add','src/target.cpp')
        self.git(self.source,'commit','-qm','base')
        self.base=self.git(self.source,'rev-parse','HEAD').stdout.decode().strip()
        self.bundle=self.root/'bundle'
        self.bundle.mkdir()
        (self.bundle/'patches').mkdir()
        (self.bundle/'modules/engram').mkdir(parents=True)
        (self.bundle/'modules/engram/row_cache.hpp').write_bytes(b'// vendor header\n')
        before=self.target.read_bytes()
        self.target.write_text('int foo() { return 2; }\n')
        self.patch=self.git(self.source,'diff','--','src/target.cpp').stdout
        self.git(self.source,'checkout','--','src/target.cpp')
        (self.bundle/'patches/0001.patch').write_bytes(self.patch)
        self.manifest={
            'schema_version':1,'release_id':'fake','upstream':{'repository':'https://example.invalid','base_commit':self.base},
            'patches':[{'id':'test','file':'0001.patch','sha256':sha(self.patch),'targets':[{'path':'src/target.cpp','before_sha256':sha(before),'after_sha256':sha(b'int foo() { return 2; }\n')}]}],
            'additional_files':[{'source':'modules/engram/row_cache.hpp','destination':'src/llama-experimental/row_cache.hpp','sha256':sha(b'// vendor header\n')}],
        }
        self.save()
    def git(self,cwd,*args):
        r=subprocess.run(['git','-c','core.autocrlf=false',*args],cwd=cwd,env=self.env,capture_output=True)
        if r.returncode:raise AssertionError(r.stderr.decode())
        return r
    def save(self): (self.bundle/'manifest.json').write_text(json.dumps(self.manifest)+'\n')
    def invoke(self,cmd,*args):
        r=subprocess.run([sys.executable,str(TOOL),cmd,'--source',str(self.source),'--manifest',str(self.bundle/'manifest.json'),*args],env=self.env,capture_output=True,text=True)
        return r.returncode,json.loads(r.stdout)
    def test_install_and_receipt(self):
        original=self.target.read_bytes()
        rc,out=self.invoke('inspect')
        self.assertEqual(rc,0)
        self.assertEqual(out['tree_state'],'BASELINE')
        dest=self.root/'candidate'
        rc,out=self.invoke('prepare','--destination',str(dest))
        self.assertEqual((rc,out.get('code')), (0,None))
        self.assertEqual((dest/'src/llama-experimental/row_cache.hpp').read_bytes(),b'// vendor header\n')
        self.assertEqual(out['additional_files'],[{'path':'src/llama-experimental/row_cache.hpp','sha256':sha(b'// vendor header\n')}])
        self.assertEqual(original,self.target.read_bytes())
        rc,inspect=self.invoke('inspect')
        self.assertEqual(inspect['tree_state'],'BASELINE')
        self.assertEqual(rc,0)
        self.assertEqual((dest/'src/target.cpp').read_text(),'int foo() { return 2; }\n')
        baseline=self.root/'baseline'
        rc,b=self.invoke('prepare','--destination',str(baseline),'--baseline-only')
        self.assertEqual(rc,0)
        self.assertFalse((baseline/'src/llama-experimental/row_cache.hpp').exists())
    def test_build_pair_accepts_only_approved_vendor_files(self):
        sys.path.insert(0,str(ROOT/'tools'))
        try:
            import build_pair
            baseline=self.root/'baseline'
            candidate=self.root/'candidate'
            rc,out=self.invoke('prepare','--destination',str(baseline),'--baseline-only')
            self.assertEqual(rc,0)
            rc,out=self.invoke('prepare','--destination',str(candidate))
            self.assertEqual(rc,0)
            rec=build_pair.validate_pair(baseline,candidate,
                self.bundle/'manifest.json', self.bundle/'patches')
            self.assertEqual(rec['status'],'BUILD_ONLY_KNOWN_BASE')
            # A modified declared file must fail, even if the receipt is intact.
            (candidate/'src/llama-experimental/row_cache.hpp').write_text('tampered\n')
            with self.assertRaises(build_pair.BuildPairError):
                build_pair.validate_pair(baseline,candidate,self.bundle/'manifest.json',self.bundle/'patches')
            (candidate/'src/llama-experimental/row_cache.hpp').write_text('// vendor header\n')
            (candidate/'src/llama-experimental/extra.hpp').write_text('unexpected\n')
            with self.assertRaises(build_pair.BuildPairError):
                build_pair.validate_pair(baseline,candidate,self.bundle/'manifest.json',self.bundle/'patches')
        finally:
            sys.path.remove(str(ROOT/'tools'))

    def test_modified_vendored_file_refused_preclone(self):
        (self.bundle/'modules/engram/row_cache.hpp').write_text('// altered\n')
        dest=self.root/'candidate'
        rc,out=self.invoke('prepare','--destination',str(dest))
        self.assertEqual(rc,2)
        self.assertEqual(out['code'],'ASSET_HASH_MISMATCH')
        self.assertFalse(dest.exists())
    def test_symlink_vendor_refused(self):
        asset=self.bundle/'modules/engram/row_cache.hpp'
        asset.unlink()
        asset.symlink_to(self.bundle/'manifest.json')
        dest=self.root/'candidate'
        rc,out=self.invoke('prepare','--destination',str(dest))
        self.assertEqual(rc,2)
        self.assertFalse(dest.exists())
    def test_bad_destination_refused_preclone(self):
        self.manifest['additional_files'][0]['destination']='src/../hacked.hpp'
        self.save()
        rc,out=self.invoke('prepare','--destination',str(self.root/'candidate'))
        self.assertEqual(rc,2)
        self.assertEqual(out['code'],'INVALID_PATH')
        self.assertFalse((self.root/'candidate').exists())
    def test_unreviewed_engram_file_and_renamed_destination_refused(self):
        entry = self.manifest["additional_files"][0]
        for key, value in (("source", "modules/engram/test.hpp"),
                           ("destination", "src/llama-experimental/other.hpp")):
            with self.subTest(key=key):
                original = entry[key]
                entry[key] = value
                self.save()
                rc, result = self.invoke("inspect")
                self.assertEqual((rc, result["code"]), (2, "INVALID_MANIFEST"))
                entry[key] = original

    def test_existing_vendor_target_refused(self):
        extra=self.source/'src/llama-experimental'
        extra.mkdir()
        (extra/'row_cache.hpp').write_bytes(b'upstream native data\n')
        self.git(self.source,'add','--all')
        self.git(self.source,'commit','-qm','native vendor collision')
        new_base=self.git(self.source,'rev-parse','HEAD').stdout.decode().strip()
        rc,out=self.invoke('prepare','--destination',str(self.root/'candidate'),'--ref',new_base)
        self.assertEqual(rc,2)
        self.assertEqual(out['code'],'ASSET_DESTINATION_EXISTS')
        self.assertEqual((extra/'row_cache.hpp').read_bytes(),b'upstream native data\n')

if __name__=='__main__':unittest.main()
