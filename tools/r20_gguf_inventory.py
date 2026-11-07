#!/usr/bin/env python3
"""Read the tensor directory of each GGUF shard. Reads only headers of the specified local set.
Reader adapted from Project Maya v1.1.46 tools/ds4/pack_gguf.py, under the accompanying MIT license.
The numeric type codes are interpreted by the exact pinned GGML constants of the build.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import runpy
import struct

class Gguf:
    """One GGUF shard's metadata and tensor infos (offsets absolute in the file)."""

    def __init__(self, path: Path):
        self.path = path
        with path.open("rb") as f:
            self.f = f
            magic = f.read(4)
            if magic != b"GGUF":
                raise SystemExit("%s: not a GGUF file" % path)
            self.version, nt, nkv = struct.unpack("<IQQ", f.read(20))
            self.kv = {}
            for _ in range(nkv):
                k = self._str()
                self.kv[k] = self._val(struct.unpack("<I", f.read(4))[0])
            self.tensors = []
            for _ in range(nt):
                name = self._str()
                nd = struct.unpack("<I", f.read(4))[0]
                dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
                typ, off = struct.unpack("<IQ", f.read(12))
                self.tensors.append((name, typ, dims, off))
            align = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_start = (pos + align - 1) // align * align
        del self.f


    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmt:
            n = struct.calcsize(fmt[t])
            return struct.unpack("<" + fmt[t], self.f.read(n))[0]
        if t == 8:
            return self._str()
        if t == 9:
            at, n = struct.unpack("<IQ", self.f.read(12))
            if at in fmt:
                n_b = struct.calcsize(fmt[at])
                return list(struct.unpack("<%d%s" % (n, fmt[at]), self.f.read(n * n_b)))
            return [self._val(at) for _ in range(n)]
        raise SystemExit("%s: metadata type %d" % (self.path, t))



def inventory(paths, source):
    c=runpy.run_path(str(source/'gguf-py/gguf/constants.py'))
    types, blocks = c['GGMLQuantizationType'], c['GGML_QUANT_SIZES']
    result={'shards':[], 'layers':{}, 'scope':'logical routed tensor bytes; excludes shared/dense, caching, paging, transport'}
    count=None
    for path in paths:
        g=Gguf(path)
        with path.open('rb') as f:
            header=f.read(g.data_start)
        fingerprint=hashlib.sha256(header).hexdigest()
        result['shards'].append({'name':path.name,'header_bytes':len(header),'header_sha256':fingerprint})
        if 'general.architecture' in g.kv:
            arch=g.kv['general.architecture']
            result['architecture']=arch
            count=g.kv.get(arch+'.expert_used_count',count)
            for key in ('block_count','expert_count','ple.layers','ple.ngram_size','ple.heads_per_ngram'):
                if arch+'.'+key in g.kv: result[key]=g.kv[arch+'.'+key]
        for name,tid,dim,off in g.tensors:
            if name=='per_layer_token_embd.weight':
                result['engram_table']={'shape':dim,'type':types(tid).name,'shard':path.name}
            m=re.fullmatch(r'blk\.(\d+)\.ffn_(gate_up|gate|up|down)_exps\.weight',name)
            if not m:continue
            if len(dim)!=3:raise ValueError(f'unexpected expert geometry {name}: {dim}')
            tp=types(tid); group,sz=blocks[tp]
            if dim[0]%group:raise ValueError('non-integral row blocks')
            single=math.prod(dim[:2])//group*sz
            layer=result['layers'].setdefault(m[1],{})
            if m[2] in layer:raise ValueError('duplicate tensor '+name)
            layer[m[2]]={'type':tp.name,'shape':dim,'single_expert_bytes':single,'shard':path.name,
                         'file_offset':g.data_start+off}
    result['experts_selected_per_layer']=count
    if isinstance(count,int) and count>0:
        result['logical_routed_weight_bytes_per_token']=count*sum(x['single_expert_bytes'] for l in result['layers'].values() for x in l.values())
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,required=True);p.add_argument('--source',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args(); m=re.search(r'-(\d{5})-of-(\d{5})\.gguf$',a.model.name)
    if m and m[1]!='00001':p.error('Specify the first shard')
    paths=([a.model.with_name(a.model.name[:m.start()]+f'-{i:05}-of-{m[2]}.gguf') for i in range(1,int(m[2])+1)] if m else [a.model])
    result=inventory(paths,a.source)
    with a.out.open('x') as f:json.dump(result,f,indent=2);f.write('\n')

if __name__=='__main__':main()
