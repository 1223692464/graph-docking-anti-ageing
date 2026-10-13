"""Verify and losslessly restore gzip-compressed release assets (stdlib only)."""
import argparse
import gzip
import hashlib
import json
import shutil
from pathlib import Path
from repo_paths import ROOT


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def prepare(check_only=False):
    entries = json.loads((ROOT / 'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))
    for entry in entries:
        stored, target = ROOT / entry['stored_path'], ROOT / entry['path']
        if digest(stored) != entry['stored_sha256']:
            raise ValueError('Stored checksum mismatch: ' + entry['stored_path'])
        if check_only:
            continue
        if entry['compression'] == 'gzip':
            if not target.exists():
                temporary = target.with_name(target.name + '.unpacking')
                with gzip.open(stored, 'rb') as incoming, temporary.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing)
                if digest(temporary) != entry['sha256']:
                    temporary.unlink()
                    raise ValueError('Extracted checksum mismatch: ' + entry['path'])
                temporary.replace(target)
        if digest(target) != entry['sha256']:
            raise ValueError('Source checksum mismatch: ' + entry['path'])
    print('PASS: verified {} source assets{}'.format(len(entries), ' (stored files only)' if check_only else ' and restored compressed files'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true')
    prepare(parser.parse_args().check_only)
