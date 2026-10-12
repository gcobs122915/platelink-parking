"""No-device tests for buffer ownership, per-photo audit isolation and error propagation."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
from pathlib import Path
import sys
import tempfile
import threading
from types import SimpleNamespace
import unittest
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import backend  # Adds the validated model primitives to the module path.
from async_usb import AsyncSharedEngine, ConcurrentUsbPipeline
from hailo_shared_engine import ModelSpec


class Bindings:
    def __init__(self,outputs):
        self.outputs=outputs
    def input(self):
        return self
    def set_buffer(self,data):
        self.data=data


class AsyncTests(unittest.TestCase):
    def test_concurrent_buffers_are_owned_by_each_job(self):
        with tempfile.TemporaryDirectory() as folder, ThreadPoolExecutor(2) as device, ThreadPoolExecutor(2) as callers:
            path=Path(folder)/'fixture.hef';path.write_bytes(b'fixture')
            spec=ModelSpec('fixture','yolo',path,hashlib.sha256(b'fixture').hexdigest(),(2,2,3),'uint8',{'out':(2,)},(2,2))
            engine=AsyncSharedEngine([spec])
            gate=threading.Barrier(2)
            class Configured:
                def wait_for_async_ready(self,**kwargs):
                    pass
                def create_bindings(self,output_buffers):
                    return Bindings(output_buffers)
                def run_async(self,bindings,callback):
                    def compute():
                        b=bindings[0]
                        gate.wait(timeout=5)
                        b.outputs['out'][:]=int(b.data[0,0,0])
                        callback(SimpleNamespace(exception=None))
                    future=device.submit(compute)
                    return SimpleNamespace(wait=lambda timeout:future.result(timeout/1000))
            engine._device=object()
            engine.active_requests=engine.max_active_requests=0
            engine._entries={'fixture':{'configured':Configured(),'submit_lock':threading.Lock(),'async_capacity':2}}
            futures=[callers.submit(engine.run,'fixture',np.full((2,2,3),number,np.uint8)) for number in (3,9)]
            values=[f.result() for f in futures]
            np.testing.assert_array_equal(values[0][0]['out'],[3,3])
            np.testing.assert_array_equal(values[1][0]['out'],[9,9])
            self.assertEqual(engine.max_active_requests,2)
            self.assertEqual(engine.active_requests,0)
            self.assertNotEqual(values[0][1]['run_seq'],values[1][1]['run_seq'])
            self.assertTrue(all(v[1]['status']=='COMPLETED' for v in values))

    def test_each_photo_has_its_own_execution_audit(self):
        pipeline=ConcurrentUsbPipeline.__new__(ConcurrentUsbPipeline)
        pipeline.local=threading.local()
        barrier=threading.Barrier(2)
        def worker(value):
            pipeline.neural_stage_runs=[value]
            pipeline._last_accelerated=bool(value)
            barrier.wait(timeout=5)
            return pipeline.neural_stage_runs,pipeline._last_accelerated
        with ThreadPoolExecutor(2) as pool:
            a=pool.submit(worker,0);b=pool.submit(worker,1)
            self.assertEqual(a.result(),([0],False))
            self.assertEqual(b.result(),([1],True))


if __name__=='__main__':
    unittest.main()
