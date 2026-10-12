import sys
from pathlib import Path
import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'app'))
import backend
from pose_ranges import plan_input,preprocess,restore_points
from async_usb import ConcurrentUsbPipeline
from all_usb_pipeline import pose_raw

PROFILES=((64,160),(64,192),(96,128),(96,160),(96,192),(128,192))


class PoseRangeTests(unittest.TestCase):
    def test_missing_64x128_keeps_all_pixels_and_only_pads_right_bottom(self):
        crop=np.arange(64*120*3,dtype=np.uint8).reshape(64,120,3)
        pixels,plan=preprocess(crop,PROFILES)
        self.assertEqual(plan['profile_hw'],[64,160])
        self.assertEqual(plan['mode'],'pad_to_fit')
        self.assertEqual(plan['scale_xy'],[1.,1.])
        np.testing.assert_array_equal(pixels[:,:120],crop[:,:,::-1])
        self.assertTrue((pixels[:,120:]==114).all())

    def test_existing_native_profiles_remain_byte_identical(self):
        for h,w in PROFILES:
            for dh,dw in ((0,0),(1,1),(31,31)):
                crop=np.random.default_rng(7).integers(0,256,(h-dh,w-dw,3),dtype=np.uint8)
                pixels,plan=preprocess(crop,PROFILES)
                self.assertEqual(plan['profile_hw'],[h,w])
                self.assertEqual(plan['mode'],'exact_native')
                np.testing.assert_array_equal(pixels,pose_raw.preprocess(crop,h,w))

    def test_large_crop_is_scaled_uniformly_and_corners_are_restored(self):
        crop=np.zeros((201,403,3),np.uint8)
        pixels,plan=preprocess(crop,PROFILES)
        self.assertEqual(plan['mode'],'downscale_to_fit')
        self.assertEqual(plan['profile_hw'],[96,192])
        expected=np.asarray([[12,15],[390,13],[388,188],[15,191]],np.float32)
        model=expected*np.asarray(plan['scale_xy'],np.float32)
        np.testing.assert_allclose(restore_points(model,plan),expected,atol=.0001)
        self.assertLessEqual(abs(plan['scale_xy'][0]-plan['scale_xy'][1]),1/201)
        self.assertEqual(pixels.shape,(96,192,3))

    def test_boundaries_always_fit_and_never_upscale(self):
        for h in (1,31,32,33,63,64,65,95,96,97,127,128,129,1024,6000):
            for w in (1,31,32,33,127,128,129,159,160,161,191,192,193,4096,8000):
                p=plan_input(h,w,PROFILES)
                ph,pw=p['profile_hw'];rh,rw=p['resized_hw']
                self.assertTrue(1<=rh<=ph and 1<=rw<=pw)
                self.assertTrue(0<p['scale_xy'][0]<=1 and 0<p['scale_xy'][1]<=1)
                self.assertEqual(p['pad_ltrb'][:2],[0,0])
                if any(ph>=h and pw>=w for ph,pw in PROFILES):
                    self.assertEqual(p['resized_hw'],[h,w])

    def test_pipeline_returns_crop_coordinates_instead_of_model_coordinates(self):
        pipe=ConcurrentUsbPipeline.__new__(ConcurrentUsbPipeline)
        pipe.pose_specs={p:SimpleNamespace(key=f'fixture-{p}') for p in PROFILES}
        observed=[]
        pipe._run=lambda spec,pixels:(observed.append((spec,pixels.shape)) or {},{'run_seq':1})
        expected=np.asarray([[15,20],[370,20],[370,170],[15,170]],np.float32)
        plan=plan_input(200,400,PROFILES)
        row=np.concatenate(([0,0,192,96,.9],(expected*np.asarray(plan['scale_xy'])).ravel()))[None].astype(np.float32)
        with patch.object(pose_raw,'decode_heads',return_value=None),patch.object(pose_raw,'nms',return_value=row):
            corners,info=pipe.corners(np.zeros((200,400,3),np.uint8))
        np.testing.assert_allclose(corners,expected,atol=.0001)
        self.assertEqual(info['input_transform']['mode'],'downscale_to_fit')
        self.assertEqual(observed[0][1],(96,192,3))

    def test_invalid_inputs_do_not_enter_the_model(self):
        for h,w in ((0,10),(-1,10),(10,0)):
            with self.assertRaises(ValueError):plan_input(h,w,PROFILES)
        with self.assertRaises(ValueError):plan_input(64,128,[])
        with self.assertRaises(ValueError):preprocess(np.zeros((10,20,3),np.float32),PROFILES)

    def test_tiny_false_positive_is_filtered_before_pose_and_ocr(self):
        # Real O118-2 box from the development crop set; bypassing this gate caused
        # a direct-crop stress-test OCR error, so cover the actual service path.
        class Pipeline:
            engine=SimpleNamespace(device_id='fixture-only',engine_id='fixture-only')
            def detect(self,bgr):
                self.neural_stage_runs.append(dict(stage='yolo',backend='HAILO_USB',status='COMPLETED',cpu_neural_fallback=False))
                return np.asarray([[1211.5552,117.2748,1240.8844,131.0692]]),np.asarray([.364]),np.asarray([0])
            def corners(self,crop):
                raise AssertionError('A rejected tiny box must never enter POSE')
        recognizer=backend.UsbRecognizer.__new__(backend.UsbRecognizer)
        recognizer.pipeline=Pipeline()
        recognizer.vocabulary={'fixture_only':True}
        result=recognizer.infer(np.zeros((720,1280,3),np.uint8),'fixture.jpg')
        self.assertEqual(result['status'],'ok')
        self.assertFalse(result['detections'][0]['accepted_size'])
        self.assertEqual(result['plates'],[])
        self.assertEqual(result['devices']['pose']['completed_runs'],0)
        self.assertEqual(result['devices']['ocr']['completed_runs'],0)


if __name__=='__main__':unittest.main()
