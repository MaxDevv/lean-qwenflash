"""download_http.py against a fake Hugging Face: every file checked, unfinished runs finished, bad files refused."""
import hashlib
import io
import json
import os
import runpy
import shutil
import sys
import tempfile
import types
import unittest
import urllib.request
from pathlib import Path
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[1] / 'download_http.py'
FILES = {'config.json': b'{"a": 1}\n', 'model-00001.safetensors': os.urandom(3 * 1024 ** 2 + 123)}
BIG = 'model-00001.safetensors'


class Resp(io.BytesIO):
    def __init__(self, data, url, status=200, headers=None):
        super().__init__(data)
        self.status, self.headers, self.url = status, headers or {}, url
    def geturl(self): return self.url
    def __enter__(self): return self
    def __exit__(self, *a): return False


@unittest.skipUnless(sys.platform.startswith('linux'), 'O_DIRECT and posix_fallocate are Linux-only')
class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)
        try:
            os.close(os.open(self.root / 'probe', os.O_CREAT | os.O_RDWR | os.O_DIRECT))
        except OSError:
            self.skipTest('this filesystem has no O_DIRECT')
        self.serve, self.names = dict(FILES), list(FILES)

    def info(self, repo, revision, files_metadata):
        def sibling(name):
            data = FILES.get(name, b'x')
            lfs = types.SimpleNamespace(sha256=hashlib.sha256(data).hexdigest()) if name == BIG else None
            return types.SimpleNamespace(rfilename=name, size=len(data), lfs=lfs,
                                         blob_id=hashlib.sha1(b'blob %d\0' % len(data) + data).hexdigest())
        return types.SimpleNamespace(sha=revision, siblings=[sibling(n) for n in self.names])

    def urlopen(self, req, timeout=None):
        if req.get_method() == 'HEAD':
            return Resp(b'', req.full_url)
        start, end = map(int, req.headers['Range'].split('=')[1].split('-'))
        data = self.serve[req.full_url.rsplit('/', 1)[1]]
        return Resp(data[start:end + 1], req.full_url, 206, {'Content-Range': f'bytes {start}-{end}/{len(data)}'})

    def download(self, model):
        hub = types.SimpleNamespace(HfApi=lambda: types.SimpleNamespace(model_info=self.info))
        with mock.patch.dict(sys.modules, {'huggingface_hub': hub}), \
                mock.patch.object(urllib.request, 'urlopen', self.urlopen), \
                mock.patch.object(sys, 'argv', ['download_http.py', '--model-dir', str(model)]):
            try:
                runpy.run_path(str(SCRIPT), run_name='__main__')
                return 'ok'
            except SystemExit as e:
                return str(e)

    def test_fresh_download_is_checked_and_noted(self):
        model = self.root / 'model'
        self.assertEqual(self.download(model), 'ok')
        for name, data in FILES.items():
            self.assertEqual((model / name).read_bytes(), data)
        self.assertEqual(sorted(json.loads((model / '.verified.json').read_text())), sorted(FILES))

    def test_a_file_whose_chunks_all_arrived_before_a_stop_is_finished(self):
        model = self.root / 'model'
        model.mkdir()
        (model / (BIG + '.http-part')).write_bytes(FILES[BIG])
        (model / (BIG + '.http-state.json')).write_text('[0]')
        (model / 'config.json').write_bytes(FILES['config.json'])
        self.assertEqual(self.download(model), 'ok')
        self.assertEqual((model / BIG).read_bytes(), FILES[BIG])
        self.assertFalse((model / (BIG + '.http-part')).exists())

    def test_a_corrupt_download_is_deleted_and_stops_the_run(self):
        self.serve[BIG] = os.urandom(len(FILES[BIG]))
        model = self.root / 'model'
        self.assertIn('does not match the pinned upload', self.download(model))
        self.assertFalse((model / BIG).exists() or (model / (BIG + '.http-part')).exists())

    def test_an_existing_file_that_differs_is_refused(self):
        model = self.root / 'model'
        model.mkdir()
        (model / 'config.json').write_bytes(b'tampered')
        self.assertIn('does not match the pinned upload', self.download(model))

    def test_symlinked_files_are_checked_not_refused(self):
        model, cache = self.root / 'model', self.root / 'cache'
        model.mkdir()
        cache.mkdir()
        for name, data in FILES.items():
            (cache / name).write_bytes(data)
            (model / name).symlink_to(cache / name)
        self.assertEqual(self.download(model), 'ok')

    def test_a_name_outside_the_folder_is_refused(self):
        self.names = ['../escape.txt']
        self.assertIn('outside the model folder', self.download(self.root / 'model'))
        self.assertFalse((self.root / 'escape.txt').exists())


if __name__ == '__main__':
    unittest.main()
