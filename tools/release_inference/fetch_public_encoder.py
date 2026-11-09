"""Fetch only the public encoder ZIP member; verify against the v4.1 checksum."""
import argparse
import hashlib
import io
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

URL = 'https://zenodo.org/records/23080342/files/checkpoints_v4_weights.zip?download=1'
SIZE = 310335742
MEMBER = 'checkpoints/joint_heavy_only/seed42/best_e2e.pt'
MEMBER_SIZE = 25803515
MEMBER_SHA = '596f2068201276fa84c81a6241d9742447880836389606e6e8b3817af5c507be'


class RangeReader(io.RawIOBase):
    def __init__(self):
        self.pos = 0
        self.transferred = 0

    def seekable(self):
        return True

    def readable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, offset, whence=0):
        if whence not in (0, 1, 2):
            raise ValueError('Bad seek mode')
        pos = offset + (0, self.pos, SIZE)[whence]
        if pos < 0:
            raise OSError('Negative archive offset')
        self.pos = pos
        return pos

    def read(self, size=-1):
        left = max(0, SIZE - self.pos)
        size = left if size < 0 else min(left, size)
        if not size:
            return b''
        if self.transferred + size > 40 * 1024 * 1024:
            raise OSError('Refusing to fetch more than 40 MiB of archive data')
        start, stop = self.pos, self.pos + size - 1
        request = Request(f'{URL}&offset={start}-{stop}',
                          headers={'Range': f'bytes={start}-{stop}', 'Accept-Encoding': 'identity'})
        with urlopen(request, timeout=30) as response:
            if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{stop}/{SIZE}':
                raise OSError('Server did not honor the exact range request')
            data = response.read(size + 1)
        if len(data) != size:
            raise OSError('Incomplete ZIP range')
        self.pos += size
        self.transferred += size
        return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    if a.output.exists():
        ap.error('Output already exists')
    with RangeReader() as remote, ZipFile(remote) as z:
        info = z.getinfo(MEMBER)
        if info.file_size != MEMBER_SIZE:
            raise ValueError('Published encoder member size mismatch')
        data = z.read(info)
        print(f'Read {remote.transferred:,} bytes of {SIZE:,}-byte ZIP')
    if len(data) != MEMBER_SIZE or hashlib.sha256(data).hexdigest() != MEMBER_SHA:
        raise ValueError('Published encoder member checksum mismatch')
    with a.output.open('xb') as f:
        f.write(data)
    print(f'Verified {MEMBER}: {MEMBER_SHA}')


if __name__ == '__main__':
    main()
