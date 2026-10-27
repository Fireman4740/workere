#!/usr/bin/env python3
"""Rehydrate checksummed, credential-free M0 artifacts from a persisted run log."""
import argparse
import base64
import gzip
import hashlib
import json
from pathlib import Path


def collect(log, out):
    groups, ends = {}, []
    for line in log.read_text().splitlines():
        if line.startswith('M0_EXPORT_CHUNK '):
            v=json.loads(line.split(' ',1)[1]); groups.setdefault(v['sha256'],{})[v['index']]=v['data']
        elif line.startswith('M0_EXPORT_END '): ends.append(json.loads(line.split(' ',1)[1]))
    if not ends: raise ValueError('no complete artifact export in log')
    for end in ends:
        chunks=groups[end['sha256']]
        if set(chunks)!=set(range(end['chunks'])): raise ValueError('incomplete export')
        data=base64.b64decode(''.join(chunks[i] for i in range(end['chunks'])),validate=True)
        if hashlib.sha256(data).hexdigest()!=end['sha256']: raise ValueError('artifact checksum mismatch')
        files=json.loads(gzip.decompress(data))
        if sorted(files)!=end['files']: raise ValueError('export file manifest mismatch')
        out.mkdir(parents=True,exist_ok=True)
        for name,value in files.items():
            if Path(name).name!=name: raise ValueError('unsafe artifact filename')
            (out/name).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
        (out/'export-provenance.json').write_text(json.dumps({'log_path':str(log),'gzip_sha256':end['sha256'],
              'files':end['files']},indent=2)+'\n')
    print(json.dumps({'output':str(out),'files':end['files'],'checksums_verified':True}))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('log',type=Path);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();collect(a.log,a.out)
