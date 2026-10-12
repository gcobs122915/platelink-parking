"""One owner per input directory; durable claim-before-rename and result journal."""
from __future__ import annotations
import os
from pathlib import Path
import sqlite3
import uuid
from results import now

ACTIVE = ('queued', 'running', 'publishing', 'publish_failed')


class InputLock:
    def __init__(self, directory):
        self.path = Path(directory) / '.plate-service.lock'
        self.stream = None

    def __enter__(self):
        self.stream = self.path.open('a+b')
        if self.path.stat().st_size == 0:
            self.stream.write(b'0')
            self.stream.flush()
        self.stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.stream.close()
            self.stream = None
            raise RuntimeError(f'此 input 已有另一個核心執行中：{self.path.parent}') from exc
        return self

    def __exit__(self, *args):
        if self.stream is not None:
            self.stream.seek(0)
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(self.stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.stream.fileno(), fcntl.LOCK_UN)
            self.stream.close()
            self.stream = None


class JobStore:
    def __init__(self, state_dir):
        self.connection = sqlite3.connect(Path(state_dir) / 'jobs.sqlite3')
        self.connection.row_factory = sqlite3.Row
        self.connection.execute('PRAGMA journal_mode=WAL')
        self.connection.execute('PRAGMA synchronous=FULL')
        self.connection.executescript('''
            CREATE TABLE IF NOT EXISTS jobs (
                job_id TEXT PRIMARY KEY, basename TEXT NOT NULL, base_key TEXT NOT NULL,
                flag_name TEXT NOT NULL, marker_name TEXT NOT NULL, state TEXT NOT NULL,
                admitted_at TEXT NOT NULL, completed_at TEXT,
                txt_path TEXT NOT NULL, json_path TEXT NOT NULL,
                payload TEXT, result_text TEXT, error TEXT);
            CREATE UNIQUE INDEX IF NOT EXISTS one_active_basename ON jobs(base_key)
                WHERE state IN ('queued','running','publishing','publish_failed');
        ''')

    def register(self, basename, flag_name, settings, marker_name=None):
        if self.connection.execute("SELECT 1 FROM jobs WHERE base_key=? AND state IN ('queued','running','publishing','publish_failed')", (basename.casefold(),)).fetchone():
            return None
        job_id = uuid.uuid4().hex
        self.connection.execute('''INSERT INTO jobs
            (job_id,basename,base_key,flag_name,marker_name,state,admitted_at,txt_path,json_path)
            VALUES (?,?,?,?,?,'queued',?,?,?)''',
            (job_id, basename, basename.casefold(), flag_name, marker_name or flag_name+'.processing', now(),
             str(Path(settings['result_dir']) / (basename+'Result.txt')),
             str(Path(settings['details_dir']) / (basename+'Result.json'))))
        self.connection.commit()
        return self.get(job_id)

    def get(self, job_id):
        row = self.connection.execute('SELECT * FROM jobs WHERE job_id=?', (job_id,)).fetchone()
        return dict(row) if row else None

    def update(self, job_id, **fields):
        allowed = {'state', 'payload', 'result_text', 'error', 'completed_at', 'marker_name'}
        if not fields or set(fields)-allowed:
            raise ValueError('Invalid journal update')
        self.connection.execute('UPDATE jobs SET '+','.join(k+'=?' for k in fields)+' WHERE job_id=?',
                                (*fields.values(), job_id))
        self.connection.commit()
        return self.get(job_id)

    def pending(self):
        return [dict(r) for r in self.connection.execute(
            "SELECT * FROM jobs WHERE state IN ('queued','running','publishing','publish_failed') ORDER BY admitted_at,job_id")]

    def close(self):
        self.connection.close()
