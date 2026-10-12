"""Protocol, recovery and failure tests use explicit fake models; no USB access."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
import threading
from unittest.mock import patch

APP = Path(__file__).resolve().parents[1]/'app'
sys.path.insert(0, str(APP))
from PIL import Image
from backend import recognize_crop, read_photo, InputError
from client import check_ready, submit
from queue_store import InputLock, JobStore
from results import normalize_text, text_result
from service import FileCore
from settings import load_settings, basename_from_flag


def plate(text='軍A12345', confidence=.98, box=(1,2,61,42)):
    return dict(text=text, raw_text=text, ocr_confidence=confidence, accepted=confidence>=.5 and bool(text),
                box=list(box), detector_confidence=.9)


class FakeRecognizer:
    models = [{'name': 'fixture-only', 'hef_sha256': 'not-a-real-model'}]
    def __init__(self, settings, plates=None):
        self.calls = 0
        self.closed = False
        self.plates = [plate()] if plates is None else plates
    def infer(self, bgr, filename):
        self.calls += 1
        return dict(status='ok', plates=self.plates, detections=[], errors=[], warnings=[], timing={})
    def close(self):
        self.closed = True


class FileCoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        (self.root/'config.json').write_text('{}', encoding='utf-8')
        self.settings = load_settings(self.root)
        self.capture = io.StringIO()
        self.redirect = contextlib.redirect_stdout(self.capture)
        self.redirect.__enter__()
        self.cores = []

    def tearDown(self):
        for core in self.cores:
            core.store.close()
        self.redirect.__exit__(None, None, None)
        self.temp.cleanup()

    def core(self, plates=None):
        core = FileCore(self.settings, FakeRecognizer)
        core.recognizer = FakeRecognizer(self.settings, plates)
        core.models = core.recognizer.models
        core.protocol_id = 'fixture-only'
        core.store = JobStore(core.state_dir)
        self.cores.append(core)
        return core

    def photo(self, name='demo.jpg'):
        p = self.root/'input'/name
        Image.new('RGB', (120,70), (120,60,0)).save(p)
        return p

    def flag(self, name='demoComplete.txt'):
        p = self.root/'input'/name
        p.write_bytes(b'contents are deliberately ignored')
        return p

    def result(self, basename='demo'):
        return json.loads((self.root/'output/details'/(basename+'Result.json')).read_text(encoding='utf-8'))

    def test_photo_without_flag_is_ignored_and_subdirectories_not_scanned(self):
        self.photo()
        nested=self.root/'input/nested';nested.mkdir()
        (nested/'nestedComplete.txt').touch()
        c=self.core();c.scan()
        self.assertEqual(c.recognizer.calls,0)

    def test_queue_has_overlapping_workers_and_does_not_mix_answers(self):
        gate=threading.Barrier(2)
        class Overlap(FakeRecognizer):
            def infer(self,bgr,filename):
                gate.wait(timeout=5)  # A whole-image serial lock would make this fail.
                return dict(status='ok',plates=[plate(filename.split('.')[0].upper())],
                            detections=[],errors=[],warnings=[],timing={})
        self.settings.update(max_inflight_images=2,max_queued_jobs=2)
        for i in range(4):
            self.photo(f'job{i}.jpg');self.flag(f'job{i}Complete.txt')
        core=FileCore(self.settings,Overlap,max_jobs=4)
        self.assertEqual(core.run(),0)
        self.assertEqual(core.status['peak_active_images'],2)
        for i in range(4):
            result=self.result(f'job{i}')
            self.assertEqual(result['plates'][0]['text'],f'JOB{i}')
            self.assertIsNotNone(result['timing']['queue_wait_ms'])

    def test_stop_drains_running_but_preserves_waiting_jobs_for_recovery(self):
        gate=threading.Barrier(2)
        stopfile=self.root/'state/STOP'
        class StopDuringInference(FakeRecognizer):
            def infer(self,bgr,filename):
                gate.wait(timeout=5)
                stopfile.touch()
                return super().infer(bgr,filename)
        self.settings.update(max_inflight_images=2,max_queued_jobs=4)
        for i in range(4):
            self.photo(f'job{i}.jpg');self.flag(f'job{i}Complete.txt')
        core=FileCore(self.settings,StopDuringInference)
        self.assertEqual(core.run(),0)
        self.assertEqual(len(list((self.root/'input').glob('*Result.txt'))),2)
        self.assertEqual(len(list((self.root/'input').glob('*.processing'))),2)
        stopfile.unlink()
        recovered=FileCore(self.settings,FakeRecognizer,max_jobs=2)
        self.assertEqual(recovered.run(),0)
        self.assertEqual(len(list((self.root/'input').glob('*Result.txt'))),4)
        self.assertFalse(list((self.root/'input').glob('*.processing')))
        self.assertFalse((self.root/'input/demoResult.txt').exists())

    def test_uppercase_flag_utf8_atomic_completion_and_same_name_resubmit(self):
        source=self.photo('DeMo.JPG');flag=self.flag('DeMoCOMPLETE.TXT');c=self.core();c.scan()
        r=self.result('DeMo')
        self.assertEqual(r['image'],source.name)
        self.assertEqual(r['source_sha256'],hashlib.sha256(source.read_bytes()).hexdigest())
        txt=self.root/'input/DeMoResult.txt'
        self.assertEqual(txt.read_bytes(),'true|軍A12345|0.980000\n'.encode('utf-8'))
        self.assertTrue(r['completed_at']);self.assertGreater(r['timing']['total_ms'],0)
        self.assertFalse(flag.exists());self.assertFalse(flag.with_name(flag.name+'.processing').exists())
        self.assertIn('車號=軍A12345',self.capture.getvalue())
        self.assertTrue(source.exists())
        old_job=r['job_id']
        self.flag('DeMoComplete.txt');c.scan()
        self.assertEqual(c.recognizer.calls,2)
        self.assertNotEqual(old_job,self.result('DeMo')['job_id'])
        rows=c.store.connection.execute('SELECT payload,result_text FROM jobs').fetchall()
        self.assertTrue(all(tuple(row)==(None,None) for row in rows))

    def test_bad_missing_and_duplicate_images_are_completed_errors(self):
        (self.root/'input/bad.jpg').write_bytes(b'not an image')
        self.flag('badComplete.txt')
        self.flag('missingComplete.txt')
        self.photo('duplicate.jpg');self.photo('duplicate.png');self.flag('duplicateComplete.txt')
        c=self.core();c.scan()
        self.assertEqual(c.recognizer.calls,0)
        for name,code in [('bad','image_decode_failed'),('missing','image_not_found'),('duplicate','image_not_unique')]:
            r=self.result(name)
            self.assertEqual(r['status'],'error');self.assertEqual(r['errors'][0]['code'],code)
            self.assertTrue(r['completed_at'])
            self.assertEqual((self.root/'input'/(name+'Result.txt')).read_bytes(),b'false|######|0\n')
        self.assertEqual(self.result('bad')['source_sha256'],hashlib.sha256(b'not an image').hexdigest())

    def test_invalid_basename_is_failed_and_never_a_fake_result(self):
        self.flag('Complete.txt');c=self.core();c.scan()
        self.assertTrue(list((self.root/'input').glob('Complete.txt.failed-*')))
        self.assertFalse((self.root/'input/Result.txt').exists())
        for name in ['.Complete.txt','CONComplete.txt','foo|barComplete.txt','demo.Complete.txt']:
            with self.assertRaises(ValueError):basename_from_flag(name)

    def test_result_policy_low_confidence_placeholder_and_normalization(self):
        values=[plate('ABC1234',.4),plate('軍C21059',.99,(0,0,120,80))]
        self.assertEqual(text_result(values),'false|ABC1234|0.400000\ntrue|軍C21059|0.990000\n')
        self.assertEqual(text_result(values,'largest'),'true|軍C21059|0.990000\n')
        self.assertEqual(text_result([]),'false|######|0\n')
        self.assertEqual(text_result([plate('A|B')]),'false|######|0\n')
        self.assertEqual(normalize_text(' a b-c\t１２ 軍 '),'ABC１２軍')

    def test_empty_high_score_retry_does_not_replace_first_and_retry_error_keeps_answer(self):
        class Ocr:
            def __init__(self,answers):self.answers=iter(answers)
            def recognize(self,pixels):
                answer=next(self.answers)
                if isinstance(answer,Exception):raise answer
                return answer
        first=dict(text='abc-1234',confidence=.8)
        r,w=recognize_crop(Ocr([first,dict(text='',confidence=.99)]),None,None,True,self.settings)
        self.assertEqual(r['text'],'ABC1234');self.assertEqual(r['ocr_source'],'rectified')
        r,w=recognize_crop(Ocr([first,RuntimeError('fixture failure')]),None,None,True,self.settings)
        self.assertTrue(r['accepted']);self.assertFalse(r['ocr_retry']['succeeded']);self.assertEqual(len(w),1)
        r,w=recognize_crop(Ocr([first]),None,None,False,self.settings)
        self.assertFalse(r['ocr_retry']['attempted'])

    def test_publication_failure_recovers_saved_answer_without_inference(self):
        self.photo();self.flag();c=self.core()
        from service import atomic_bytes as real_write
        def fail_txt(path,data):
            if str(path).endswith('Result.txt'):raise PermissionError('fixture write lock')
            return real_write(path,data)
        with patch('service.atomic_bytes',side_effect=fail_txt):c.scan()
        self.assertEqual(c.recognizer.calls,1)
        self.assertFalse((self.root/'input/demoResult.txt').exists())
        self.assertTrue(list((self.root/'input').glob('demoComplete.txt.failed-*')))
        self.assertEqual(c.store.pending()[0]['state'],'publish_failed')
        # A fresh core must publish the journal payload, not run a model again.
        restarted=self.core();restarted.recover()
        self.assertEqual(restarted.recognizer.calls,0)
        r=self.result();self.assertTrue(r['recovered']);self.assertIsNone(r['timing']['total_ms'])
        self.assertTrue(r['completed_at']);self.assertFalse(restarted.store.pending())
        self.assertFalse(list((self.root/'input').glob('*failed-*')))

    def test_interrupted_claim_and_orphan_processing_are_reinferred(self):
        self.photo();flag=self.flag();c=self.core()
        c.store.register('demo',flag.name,self.settings)  # Crash before rename.
        self.photo('orphan.jpg');self.flag('orphanComplete.txt').rename(self.root/'input/orphanComplete.txt.processing')
        restarted=self.core();restarted.recover()
        self.assertEqual(restarted.recognizer.calls,2)
        self.assertTrue(self.result()['recovered']);self.assertTrue(self.result('orphan')['completed_at'])
        self.assertEqual(c.recognizer.calls,0)

    def test_single_owner_stop_and_offline_client(self):
        with InputLock(self.root/'input'):
            with self.assertRaises(RuntimeError):
                with InputLock(self.root/'input'):pass
        with self.assertRaises((FileNotFoundError,RuntimeError)):check_ready(self.settings)
        self.photo();self.flag();c=self.core()
        (self.root/'state/STOP').touch();c.scan()
        self.assertEqual(c.recognizer.calls,0);self.assertTrue((self.root/'input/demoComplete.txt').exists())

    def test_exif_and_limits(self):
        image=Image.new('RGB',(120,70),(0,0,255));exif=Image.Exif();exif[274]=6
        p=self.root/'input/rotated.jpg';image.save(p,exif=exif)
        bgr,digest,metadata=read_photo(p,self.settings)
        self.assertEqual(bgr.shape[:2],(120,70));self.assertTrue(metadata['exif_corrected'])
        self.assertEqual(digest,hashlib.sha256(p.read_bytes()).hexdigest())
        with self.assertRaises(InputError):read_photo(p,{**self.settings,'max_pixels':10})
        with self.assertRaises(InputError):read_photo(p,{**self.settings,'max_file_bytes':10})

    def test_helper_publishes_photo_before_complete_and_keeps_source(self):
        source=self.photo('source.jpg');r=submit(source,self.settings)
        target=Path(r['image']);self.assertEqual(source.read_bytes(),target.read_bytes())
        self.assertEqual(target.with_name(r['basename']+'Complete.txt').read_bytes(),b'')
        self.assertFalse(target.with_name(target.name+'.upload').exists())
        self.assertTrue(source.exists())

    def test_full_loop_loads_once_and_stops_after_delivery(self):
        self.photo();self.flag()
        made=[]
        def factory(settings):
            model=FakeRecognizer(settings);made.append(model);return model
        c=FileCore(self.settings,factory,max_jobs=1)
        self.assertEqual(c.run(),0);self.assertEqual(len(made),1);self.assertTrue(made[0].closed)
        s=json.loads((self.root/'state/status.json').read_text(encoding='utf-8'))
        self.assertEqual(s['state'],'stopped');self.assertTrue(s['models_released'])
        self.assertIn('[INPUT_READY]',self.capture.getvalue())


if __name__ == '__main__':unittest.main()
