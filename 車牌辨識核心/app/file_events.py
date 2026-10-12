"""Native Windows directory wakeups, with periodic rescans and polling fallback."""
from __future__ import annotations
import ctypes
import os
import time


class DirectoryWakeup:
    def __init__(self, directory, scan_seconds=2., poll_seconds=.25):
        self.scan_seconds, self.poll_seconds = scan_seconds, poll_seconds
        self.handle = None
        self.last_scan = 0.
        self.mode = 'polling'
        if os.name == 'nt':
            try:
                from ctypes import wintypes
                self.api = ctypes.WinDLL('kernel32', use_last_error=True)
                self.api.FindFirstChangeNotificationW.argtypes = [wintypes.LPCWSTR, wintypes.BOOL, wintypes.DWORD]
                self.api.FindFirstChangeNotificationW.restype = wintypes.HANDLE
                for name in ('FindNextChangeNotification', 'FindCloseChangeNotification'):
                    fn = getattr(self.api, name)
                    fn.argtypes = [wintypes.HANDLE]
                    fn.restype = wintypes.BOOL
                self.api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
                self.api.WaitForSingleObject.restype = wintypes.DWORD
                handle = self.api.FindFirstChangeNotificationW(str(directory), False, 0x1 | 0x8 | 0x10)
                if handle and handle != ctypes.c_void_p(-1).value:
                    self.handle = handle
                    self.mode = 'windows_notification_with_rescan'
            except (OSError, AttributeError):
                self.handle = None

    def wait(self, stop):
        while not stop.is_set():
            if self.handle is None:
                stop.wait(self.poll_seconds)
                return
            elapsed = time.monotonic() - self.last_scan
            if elapsed >= self.scan_seconds:
                self.last_scan = time.monotonic()
                return
            result = self.api.WaitForSingleObject(self.handle, max(1, int(min(.25, self.scan_seconds-elapsed)*1000)))
            if result == 0:
                if not self.api.FindNextChangeNotification(self.handle):
                    self.close()
                    self.mode = 'polling'
                self.last_scan = time.monotonic()
                return
            if result != 258:  # WAIT_TIMEOUT
                self.close()
                self.mode = 'polling'

    def close(self):
        if self.handle is not None:
            self.api.FindCloseChangeNotification(self.handle)
            self.handle = None
