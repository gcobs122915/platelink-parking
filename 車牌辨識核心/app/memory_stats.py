"""Read-only Windows process and host RAM counters; never relabel these as NPU RAM."""
import ctypes
from ctypes import wintypes
import os


def snapshot():
    if os.name != 'nt':
        return {'available':False, 'reason':'Windows counters only'}
    class Counters(ctypes.Structure):
        _fields_ = [('cb',wintypes.DWORD),('PageFaultCount',wintypes.DWORD)] + [
            (name,ctypes.c_size_t) for name in ('PeakWorkingSetSize','WorkingSetSize','QuotaPeakPagedPoolUsage',
                'QuotaPagedPoolUsage','QuotaPeakNonPagedPoolUsage','QuotaNonPagedPoolUsage',
                'PagefileUsage','PeakPagefileUsage','PrivateUsage')]
    class Memory(ctypes.Structure):
        _fields_ = [('length',wintypes.DWORD),('load',wintypes.DWORD)] + [
            (name,ctypes.c_ulonglong) for name in ('total','available','total_page','available_page',
                                                  'total_virtual','available_virtual','extended')]
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    psapi = ctypes.WinDLL('psapi', use_last_error=True)
    kernel.GetCurrentProcess.restype = wintypes.HANDLE
    psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,ctypes.POINTER(Counters),wintypes.DWORD]
    psapi.GetProcessMemoryInfo.restype = wintypes.BOOL
    kernel.GlobalMemoryStatusEx.argtypes = [ctypes.POINTER(Memory)]
    kernel.GlobalMemoryStatusEx.restype = wintypes.BOOL
    c, m = Counters(), Memory()
    c.cb, m.length = ctypes.sizeof(c), ctypes.sizeof(m)
    if not psapi.GetProcessMemoryInfo(kernel.GetCurrentProcess(),ctypes.byref(c),c.cb):
        raise ctypes.WinError(ctypes.get_last_error())
    if not kernel.GlobalMemoryStatusEx(ctypes.byref(m)):
        raise ctypes.WinError(ctypes.get_last_error())
    return dict(available=True, scope='Windows host RAM; NOT accelerator memory',
        process_working_set_bytes=c.WorkingSetSize, process_peak_working_set_bytes=c.PeakWorkingSetSize,
        process_private_bytes=c.PrivateUsage, process_peak_commit_bytes=c.PeakPagefileUsage,
        host_total_bytes=m.total, host_available_bytes=m.available, host_used_bytes=m.total-m.available)
