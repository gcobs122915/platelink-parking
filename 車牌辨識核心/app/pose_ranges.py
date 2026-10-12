"""Map variable plate crops into resident POSE profiles and restore coordinates.

Exact native profiles keep their original input bytes. Missing sizes use the
smallest profile that fits. Oversized crops fit with aspect-preserving downscale;
padding is always on the right/bottom and is never fed to OCR as plate content.
"""
from fractions import Fraction
import cv2
import numpy as np

POLICY = 'smallest_fit_downscale_v1'


def plan_input(height, width, profiles):
    if type(height) is not int or type(width) is not int or min(height, width) < 1:
        raise ValueError('POSE crop dimensions must be positive integers')
    profiles = sorted(set(tuple(p) for p in profiles))
    if not profiles or any(len(p)!=2 or any(type(n) is not int or n<32 or n%32 for n in p) for p in profiles):
        raise ValueError('POSE requires positive stride-32 resident profiles')
    # Exact rational comparison avoids choosing different buckets due to float ties.
    choices=[]
    for ph,pw in profiles:
        scale=min(Fraction(1),Fraction(ph,height),Fraction(pw,width))
        choices.append(((-scale,ph*pw,ph,pw),scale,(ph,pw)))
    _,scale,(ph,pw)=min(choices)
    rh=max(1,min(ph,round(height*scale)))
    rw=max(1,min(pw,round(width*scale)))
    native=[(height+31)//32*32,(width+31)//32*32]
    mode='exact_native' if native==[ph,pw] and (rh,rw)==(height,width) else (
        'pad_to_fit' if (rh,rw)==(height,width) else 'downscale_to_fit')
    return dict(policy=POLICY,mode=mode,crop_hw=[height,width],native_pad_hw=native,
        profile_hw=[ph,pw],resized_hw=[rh,rw],scale_xy=[rw/width,rh/height],
        pad_ltrb=[0,0,pw-rw,ph-rh],padding_value=114,scale_up=False,
        coordinate_mapping='crop_xy = (model_xy - pad_left_top) / scale_xy',
        note='Integer resized dimensions can differ by subpixel rounding from the ideal uniform scale')


def preprocess(crop, profiles):
    if not isinstance(crop,np.ndarray) or crop.dtype!=np.uint8 or crop.ndim!=3 or crop.shape[2]!=3:
        raise ValueError('POSE requires a uint8 BGR crop')
    plan=plan_input(int(crop.shape[0]),int(crop.shape[1]),profiles)
    rh,rw=plan['resized_hw']
    pixels=crop if plan['crop_hw']==[rh,rw] else cv2.resize(crop,(rw,rh),interpolation=cv2.INTER_LINEAR)
    left,top,right,bottom=plan['pad_ltrb']
    padded=cv2.copyMakeBorder(pixels,top,bottom,left,right,cv2.BORDER_CONSTANT,value=(114,114,114))
    return np.ascontiguousarray(padded[:,:,::-1]),plan


def restore_points(points, plan):
    points=np.asarray(points,dtype=np.float32)
    if points.shape!=(4,2) or not np.isfinite(points).all():
        raise ValueError('POSE must return four finite ordered corners')
    restored=(points-np.asarray(plan['pad_ltrb'][:2],np.float32))/np.asarray(plan['scale_xy'],np.float32)
    # No clamping/reordering: the existing geometry validator handles bad corners.
    return restored
