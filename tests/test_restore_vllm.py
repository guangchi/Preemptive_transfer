"""Restoration must preserve unfamiliar sources and back up originals."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import restore_vllm as restore


def digest(data):
    return hashlib.sha256(data).hexdigest()


@pytest.fixture
def installation(tmp_path, monkeypatch):
    root, site = tmp_path/'repo', tmp_path/'site'
    (root/'reproducibility').mkdir(parents=True)
    (root/'snapshot/vllm').mkdir(parents=True)
    (site/'vllm').mkdir(parents=True)
    files = []
    for name, original in [('old.py', b'original'), ('new.py', None)]:
        content = b'updated-' + name.encode()
        (root/'snapshot/vllm'/name).write_bytes(content)
        if original is not None:
            (site/'vllm'/name).write_bytes(original)
        files.append(dict(path='vllm/'+name, sha256=digest(content),
                          original_sha256=digest(original) if original else None))
    manifest = dict(vllm_version='0.32.0', snapshot='snapshot', files=files)
    (root/'reproducibility/vllm-patch-manifest.json').write_text(json.dumps(manifest))
    monkeypatch.setattr(restore, 'ROOT', root)
    monkeypatch.setattr(restore.importlib.metadata, 'distribution',
                        lambda name: SimpleNamespace(version='0.32.0', locate_file=lambda name: site))
    monkeypatch.setattr('sys.argv', ['restore_vllm.py', '--apply'])
    return root, site, manifest


def test_apply_backups_and_idempotence(installation):
    root, site, manifest = installation
    restore.main()
    assert all(i['status'] == 'installed' for i in restore.inspect_files(manifest, site))
    backup, = (root/'.vllm-backups').iterdir()
    assert (backup/'vllm/old.py').read_bytes() == b'original'
    assert not (backup/'vllm/new.py').exists()
    restore.main()
    assert len(list((root/'.vllm-backups').iterdir())) == 1


@pytest.mark.parametrize('damage', ['foreign-source', 'snapshot'])
def test_refuse_without_partial_install(installation, damage):
    root, site, manifest = installation
    target = site/'vllm/new.py' if damage == 'foreign-source' else root/'snapshot/vllm/new.py'
    target.write_bytes(b'unknown')
    with pytest.raises(ValueError):
        restore.main()
    assert (site/'vllm/old.py').read_bytes() == b'original'
    assert not (root/'.vllm-backups').exists()


def test_failure_rolls_back(installation, monkeypatch):
    root, site, manifest = installation
    original_copy = restore.shutil.copyfile

    def fail_new_file(src, dst, **kwargs):
        if Path(src) == root/'snapshot/vllm/new.py':
            Path(dst).write_bytes(b'partial')
            raise OSError('simulated disk failure')
        return original_copy(src, dst, **kwargs)

    monkeypatch.setattr(restore.shutil, 'copyfile', fail_new_file)
    with pytest.raises(OSError, match='simulated'):
        restore.main()
    assert (site/'vllm/old.py').read_bytes() == b'original'
    assert not (site/'vllm/new.py').exists()
