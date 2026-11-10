"""Fetch only the anonymous wwPDB/RCSB files pinned by a public replay plan."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.request import Request, urlopen


def pinned_sources(plan, root, ccd_root):
    sources = {}
    for row in plan['records']:
        pdb, code, origin, name = (row[k] for k in ('pdb_id', 'component_id', 'origin', 'filename'))
        if not re.fullmatch('[a-z0-9]{4}', pdb) or not re.fullmatch('[A-Z0-9]{1,5}', code):
            raise ValueError('Unsafe PDB or CCD identifier')
        if (origin, name) in {('mmcif', f'{pdb}.cif.gz'), ('pdb', f'{pdb}.pdb.gz')}:
            url = f'https://files.rcsb.org/download/{name}'
        elif origin == 'versions' and re.fullmatch(rf'pdb_0000{pdb}_xyz_v\d+-\d+\.cif\.gz', name):
            url = f'https://files-versioned.wwpdb.org/pdb_versioned/data/entries/{pdb[1:3]}/pdb_0000{pdb}/{name}'
        else:
            raise ValueError(f'Unexpected coordinate filename for {pdb}')
        for path, link, digest in ((root/origin/name, url, row['coordinate_sha256']),
                                    (ccd_root/f'{code}.cif', f'https://files.rcsb.org/ligands/download/{code}.cif', row['ccd_sha256'])):
            if not re.fullmatch('[0-9a-f]{64}', digest) or (path in sources and sources[path] != (link, digest)):
                raise ValueError(f'Conflicting or invalid pin for {path}')
            sources[path] = (link, digest)
    return sources


def fetch(item):
    path, (url, digest) = item
    if path.exists():
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f'Existing file differs from pinned bytes: {path}')
        return False
    for attempt in range(3):
        try:
            with urlopen(Request(url, headers={'User-Agent': 'ligand-pose-replay/1.0'}), timeout=30) as handle:
                data = handle.read(80 * 1024 * 1024 + 1)
            break
        except OSError:
            if attempt == 2:
                raise
            time.sleep(1 + attempt)
    if len(data) > 80 * 1024 * 1024 or hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'Public file differs from pinned bytes: {url}')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as f:
        f.write(data)
    return True


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', type=Path, required=True)
    p.add_argument('--source-root', type=Path, required=True)
    p.add_argument('--ccd-dir', type=Path, required=True)
    p.add_argument('--workers', type=int, default=6)
    args = p.parse_args()
    plan = json.loads(args.plan.read_text(encoding='utf-8'))
    if plan['schema'] != 'public-pose-replay-v1' or plan['n'] != len(plan['records']):
        raise ValueError('Invalid plan')
    sources = pinned_sources(plan, args.source_root, args.ccd_dir)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        downloaded = sum(pool.map(fetch, sources.items()))
    print(f'Verified {len(sources)} source files ({downloaded} newly downloaded)')


if __name__ == '__main__':
    main()
