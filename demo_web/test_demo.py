"""Explicit test doubles only: these tests never load HailoRT or access USB."""
from __future__ import annotations
import http.client
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
import uuid

import demo_config
import demo_server as web
import launcher


class DemoSmokeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.core = self.root / 'core'
        self.settings = {}
        for name in ('state_dir', 'input_dir', 'result_dir', 'details_dir', 'yolo_dir'):
            folder = self.core / name
            folder.mkdir(parents=True, exist_ok=True)
            self.settings[name] = str(folder)
        self.sample = self.root / 'synthetic.jpg'
        self.sample.write_bytes(b'synthetic unit-test bytes; no image decoder is run')
        self.patches = [patch.object(web, 'prepare_runtime'),
                        patch.object(web, 'load_settings', return_value=self.settings),
                        patch.object(web, 'CORE_ROOT', self.core),
                        patch.object(web, 'RUNTIME', self.root),
                        patch.object(web, 'SAMPLES', {key: self.sample for key in ('military', 'diplomatic', 'foreign')})]
        for item in self.patches:
            item.start()
        self.demo = web.Demo()
        self.demo.pnp = {'present': True, 'status': 'OK'}
        self.demo.core_state = lambda: ({'state': 'running'}, True, 0)

    def tearDown(self):
        self.demo.stop.set()
        for item in reversed(self.patches):
            item.stop()
        self.temp.cleanup()

    def fake_submit(self, source, settings, *, bad_hash=False, backend='HAILO_USB'):
        basename = 'test_' + uuid.uuid4().hex
        image = Path(settings['input_dir']) / (basename + '.jpg')
        image.write_bytes(b'test-only')
        detail = Path(settings['details_dir']) / (basename + 'Result.json')
        txt = Path(settings['result_dir']) / (basename + 'Result.txt')
        txt.write_text('synthetic', encoding='utf-8')
        result = {'status': 'ok', 'basename': basename, 'source_sha256': 'wrong' if bad_hash else 'test-hash',
                  'job_id': basename, 'completed_at': web.now(), 'plates': [{'text': 'TEST123', 'accepted': True}],
                  'cpu_neural_fallback': False, 'neural_stage_runs': [
                      {'stage': stage, 'backend': backend, 'status': 'COMPLETED', 'cpu_neural_fallback': False}
                      for stage in ('yolo', 'pose', 'ocr')], 'timing': {'processing_ms': 10, 'total_ms': 11}}
        detail.write_text(json.dumps(result), encoding='utf-8')
        return {'basename': basename, 'source_sha256': 'test-hash', 'image': str(image),
                'result_json': str(detail), 'result_txt': str(txt)}

    def run_worker(self, submitter):
        with patch.object(web, 'check_ready'), patch.object(web, 'submit', side_effect=submitter) as call:
            thread = threading.Thread(target=self.demo.worker)
            thread.start()
            deadline = time.monotonic() + 3
            while self.demo.pending.unfinished_tasks and time.monotonic() < deadline:
                time.sleep(0.01)
            self.demo.stop.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
            self.assertEqual(self.demo.pending.unfinished_tasks, 0)
            return call.call_count

    def test_same_photo_creates_two_fresh_submissions(self):
        one = self.demo.enqueue('military')['jobs'][0]['id']
        two = self.demo.enqueue('military')['jobs'][0]['id']
        self.assertNotEqual(one, two)
        self.assertEqual(self.run_worker(self.fake_submit), 2)
        self.assertNotEqual(self.demo.jobs[one]['core_job_id'], self.demo.jobs[two]['core_job_id'])
        self.assertEqual(self.demo.jobs[one]['status'], 'completed')
        self.assertEqual(self.demo.jobs[one]['stages'][0]['stage'], 'plate')

    def test_result_hash_mismatch_is_an_error(self):
        key = self.demo.enqueue('military')['jobs'][0]['id']
        self.run_worker(lambda *args: self.fake_submit(*args, bad_hash=True))
        self.assertEqual(self.demo.jobs[key]['status'], 'error')
        self.assertIn('hash does not match', self.demo.jobs[key]['error'])

    def test_non_usb_events_do_not_report_success(self):
        key = self.demo.enqueue('military')['jobs'][0]['id']
        self.run_worker(lambda *args: self.fake_submit(*args, backend='CPU_TEST_DOUBLE'))
        self.assertEqual(self.demo.jobs[key]['status'], 'error')
        self.assertFalse(self.demo.jobs[key]['all_observed_events_usb'])

    def test_missing_sample_and_unready_core_reject_jobs(self):
        self.sample.unlink()
        with self.assertRaises(ValueError):
            self.demo.enqueue('military')
        self.demo.pnp = {'present': False, 'status': 'missing'}
        with self.assertRaises(RuntimeError):
            self.demo.enqueue('military')
        self.assertEqual(len(self.demo.jobs), 0)

    def test_queue_is_bounded(self):
        for _ in range(4):
            self.demo.enqueue('all')
        with self.assertRaises(RuntimeError):
            self.demo.enqueue('military')

    def test_shared_json_read_retries_transient_permission_error(self):
        with patch.object(web, 'read_json_once', side_effect=[PermissionError('sharing'), {'ok': True}]) as read:
            self.assertEqual(web.read_json(self.root / 'x.json'), {'ok': True})
            self.assertEqual(read.call_count, 2)

    def test_http_boundaries_and_recording_limit(self):
        server = web.ThreadingHTTPServer(('127.0.0.1', 0), web.Handler)
        port = server.server_address[1]
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        with patch.object(web, 'DEMO', self.demo), patch.object(web, 'HOSTS', {f'127.0.0.1:{port}'}):
            thread.start()
            def request(method, path, body=None, headers=None):
                conn = http.client.HTTPConnection('127.0.0.1', port, timeout=3)
                try:
                    conn.request(method, path, body=body, headers=headers or {})
                    response = conn.getresponse()
                    data = response.read()
                    return response.status, data
                finally:
                    conn.close()
            try:
                status, page = request('GET', '/')
                self.assertEqual(status, 200)
                self.assertIn(b'Run on UGen', page)
                status, payload = request('GET', '/api/state')
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(payload)['jobs'], [])
                self.assertEqual(request('GET', '/', headers={'Host': 'attacker.example'})[0], 403)
                self.assertEqual(request('POST', '/api/run', '{}', {'Origin': 'https://attacker.example'})[0], 403)
                self.assertEqual(request('GET', '/', headers={'Sec-Fetch-Site': 'cross-site'})[0], 403)
                self.assertEqual(request('GET', '/../../README.md')[0], 404)
                self.sample.unlink()
                self.assertEqual(request('GET', '/sample/military.jpg')[0], 404)
                self.assertEqual(request('POST', '/api/run', '{}', {'Content-Type': 'text/plain'})[0], 415)
                self.assertEqual(request('POST', '/api/recording', b'', {
                    'Content-Type': 'video/webm', 'Content-Length': str(web.MAX_RECORDING + 1)})[0], 413)
                self.assertFalse((self.root / 'recordings').exists())
            finally:
                server.shutdown()
                server.server_close()
                thread.join(2)

    def test_configuration_cannot_escape_sample_or_runtime_root(self):
        with patch.object(demo_config, 'ROOT', self.root), patch.object(demo_config, 'SAMPLE_ROOT', self.root / 'samples'):
            (self.root / 'demo.config.json').write_text(json.dumps({
                'app_dir': str(web.APP), 'samples': {'military': '../escape.jpg',
                'diplomatic': 'd.jpg', 'foreign': 'f.jpg'}}), encoding='utf-8')
            with self.assertRaises(ValueError):
                demo_config.load_config()
        with patch.object(demo_config, 'CORE_ROOT', self.core):
            (self.core / 'config.json').write_text(json.dumps({'input_dir': '../escape'}), encoding='utf-8')
            with self.assertRaises(ValueError):
                demo_config.prepare_runtime()

    def test_launcher_lock_and_stop_are_demo_local(self):
        (self.core / 'config.json').write_text(json.dumps({'state_dir': 'state_dir'}), encoding='utf-8')
        with patch.object(launcher, 'RUNTIME', self.root), patch.object(launcher, 'CORE_ROOT', self.core), \
             patch.object(launcher, 'prepare_runtime'):
            owned_lock = launcher.acquire_lock()
            try:
                with self.assertRaises(RuntimeError):
                    launcher.acquire_lock()
                launcher.request_stop()
                self.assertTrue((self.root / 'STOP_DEMO').is_file())
                self.assertTrue((self.core / 'state_dir/STOP').is_file())
            finally:
                owned_lock.close()
            # Releasing our own handle makes the next startup possible.
            again = launcher.acquire_lock()
            again.close()


if __name__ == '__main__':
    unittest.main(verbosity=2)
