#!/usr/bin/env python3
"""Apply the archived Python modifications to the matching installed wheel."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str | None:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def inspect_files(manifest: dict, install_root: Path) -> list[dict]:
    """Validate every input before writing any installed file."""
    result = []
    for entry in manifest['files']:
        relative = Path(entry['path'])
        if relative.is_absolute() or '..' in relative.parts or relative.parts[0] != 'vllm':
            raise ValueError(f'Unsafe manifest path: {relative}')
        source = ROOT / manifest['snapshot'] / relative
        if sha256(source) != entry['sha256']:
            raise ValueError(f'Snapshot damaged or Git LFS objects missing: {source}')
        target = install_root / relative
        current = sha256(target)
        if target.exists() and not target.is_file():
            raise ValueError(f'Not a regular file: {target}')
        if current == entry['sha256']:
            status = 'installed'
        elif current == entry['original_sha256']:
            status = 'pending'
        else:
            raise ValueError(f'Unexpected vLLM source, refusing to overwrite: {target}')
        result.append(dict(entry=entry, source=source, target=target, status=status))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument('--apply', action='store_true', help='Back up and install modifications.')
    action.add_argument('--check', action='store_true', help='Fail unless all modifications are installed.')
    parser.add_argument('--backup-dir', type=Path, help='New, empty backup directory (apply only).')
    args = parser.parse_args()
    manifest = json.loads((ROOT / 'reproducibility/vllm-patch-manifest.json').read_text())
    dist = importlib.metadata.distribution('vllm')
    if dist.version != manifest['vllm_version']:
        raise ValueError(f"Requires vllm=={manifest['vllm_version']}, found {dist.version}")
    install_root = Path(dist.locate_file('')).resolve()
    # Editable/source installs may locate the source elsewhere. Fail safely.
    if not (install_root / 'vllm').is_dir():
        raise ValueError('Expected a wheel installation; use a dedicated virtual environment.')
    items = inspect_files(manifest, install_root)
    pending = [item for item in items if item['status'] == 'pending']
    print(f'Python: {sys.executable}\nvLLM location: {install_root / "vllm"}')
    print(f'{len(items) - len(pending)}/{len(items)} modifications installed')
    if args.check and pending:
        raise ValueError('Run this script with --apply first.')
    if not args.apply or not pending:
        if pending:
            print('Dry run: use --apply to install. Stop running P/D services first.')
        return
    backup = args.backup_dir or ROOT / '.vllm-backups' / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    backup.mkdir(parents=True, exist_ok=False)
    originals = []
    # Back up all pending originals before modifying the installation.
    for item in pending:
        target = item['target']
        existed = target.exists()
        originals.append(dict(path=item['entry']['path'], existed=existed))
        if existed:
            destination = backup / item['entry']['path']
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(target, destination)
    (backup / 'backup-manifest.json').write_text(json.dumps(dict(
        install_root=str(install_root), files=originals), indent=2) + '\n')
    changed = []
    try:
        for item in pending:
            target = item['target']
            target.parent.mkdir(parents=True, exist_ok=True)
            changed.append(item)
            shutil.copyfile(item['source'], target)
            if sha256(target) != item['entry']['sha256']:
                raise OSError(f'Copy verification failed: {target}')
    except BaseException:
        for item in reversed(changed):
            original = backup / item['entry']['path']
            if original.exists():
                shutil.copyfile(original, item['target'])
            else:
                item['target'].unlink(missing_ok=True)
        raise
    # Old timestamp-based bytecode must not override restored source.
    for item in pending:
        for bytecode in (item['target'].parent / '__pycache__').glob(item['target'].stem + '.*.pyc'):
            bytecode.unlink()
    print(f'Installed and verified {len(pending)} files. Originals: {backup}')


if __name__ == '__main__':
    try:
        main()
    except (ValueError, OSError, importlib.metadata.PackageNotFoundError) as error:
        raise SystemExit(str(error))
