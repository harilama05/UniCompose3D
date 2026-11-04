"""One credential-free warm GPU process; serialized with all 3D backends."""
from __future__ import annotations
import contextlib
import json
import os
import queue
import subprocess
import threading
import time
import weakref
from pathlib import Path

GPU_LOCK = threading.Lock()
_RESIDENTS = weakref.WeakSet()


def release_residents(except_worker=None):
    for worker in list(_RESIDENTS):
        if worker is except_worker: continue
        if except_worker is not None and ({worker.mode, except_worker.mode} == {"sam", "hunyuan"}):
            worker.park()
        else: worker.close()


@contextlib.contextmanager
def gpu_session(stopping, owner=None):
    while not GPU_LOCK.acquire(timeout=.1):
        if stopping.is_set(): raise RuntimeError("Cancelled while waiting for GPU.")
    try:
        if stopping.is_set(): raise RuntimeError("Cancelled before GPU execution.")
        release_residents(owner)
        yield
    finally: GPU_LOCK.release()


class ResidentWorker:
    def __init__(self, python, mode, repository, weights):
        self.python, self.mode = Path(python), mode
        self.repository, self.weights = Path(repository), Path(weights)
        self.process = None; self.messages = queue.Queue(); self.log = None; self.timer = None
        self.lock = threading.RLock()
        self.runner = Path(__file__).resolve().parents[1] / "tools/resident_worker.py"
        _RESIDENTS.add(self)

    def close(self):
        with self.lock:
            if self.timer: self.timer.cancel(); self.timer = None
            process, self.process = self.process, None
            if process is not None:
                if process.poll() is None:
                    if os.name=="nt" and isinstance(process.pid,int):
                        try:
                            import psutil
                            for child in psutil.Process(process.pid).children(recursive=True):
                                try:child.terminate()
                                except psutil.Error:pass
                        except (ImportError, OSError):pass
                    process.terminate()
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: process.kill(); process.wait(timeout=5)
                for stream in (process.stdin, process.stdout):
                    if stream: stream.close()
            if self.log: self.log.close(); self.log = None

    def park(self):
        """Keep CPU parameters, release VRAM while the other fast model runs."""
        with self.lock:
            if self.process is None or self.process.poll() is not None: return
            try:
                self.process.stdin.write(json.dumps({"_control": "park"}) + "\n")
                self.process.stdin.flush()
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    try: message = self.messages.get(timeout=.1)
                    except queue.Empty: continue
                    if message.get("event") == "result": return
                    if message.get("event") == "error": break
            except (OSError, ValueError): pass
            self.close()

    def _idle(self):
        if GPU_LOCK.acquire(blocking=False):
            try: self.close()
            finally: GPU_LOCK.release()
        else:
            self.timer = threading.Timer(5, self._idle); self.timer.daemon = True; self.timer.start()

    def _start(self, directory):
        from .quality import gpu_environment
        self.messages = queue.Queue()
        self.log = (Path(directory) / (self.mode + "-worker.log")).open("a", encoding="utf-8")
        runner = self.runner
        command = [str(self.python), "-u", str(runner), self.mode, "--repository", str(self.repository), "--weights", str(self.weights)]
        env = gpu_environment()
        try:
            self.process = subprocess.Popen(command,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,text=True,encoding="utf-8",bufsize=1,env=env)
        except Exception:
            self.log.close();self.log=None
            raise
        process, messages = self.process, self.messages
        def read():
            try:
                for line in process.stdout:
                    try: messages.put(json.loads(line))
                    except ValueError: continue  # Upstream progress is never treated as a result.
            except (OSError, ValueError): pass
            finally: messages.put({"event": "error", "error": "The GPU worker exited. See its log."})
        threading.Thread(target=read, daemon=True).start()

    def call(self, request, directory, stopping, progress=None, timeout=600):
        from .clients import ProviderError, safe_error
        with gpu_session(stopping, self), self.lock:
            if self.timer: self.timer.cancel(); self.timer = None
            reused = self.process is not None and self.process.poll() is None
            if not reused:
                self.close(); self._start(directory)
            started = time.monotonic()
            try:
                self.process.stdin.write(json.dumps(request) + "\n"); self.process.stdin.flush()
                while True:
                    if stopping.is_set():
                        self.close(); raise ProviderError("GPU job cancelled.")
                    if time.monotonic() - started > timeout:
                        self.close(); raise ProviderError("The GPU job timed out; existing results are retained.")
                    try: message = self.messages.get(timeout=.1)
                    except queue.Empty: continue
                    if message.get("event") == "progress":
                        if progress: progress(message.get("stage", "gpu"), message.get("timings", {}))
                    elif message.get("event") == "error":
                        self.close(); raise ProviderError(safe_error(message.get("error", "GPU model error.")))
                    elif message.get("event") == "result":
                        result = message["result"]; result["worker_reused"] = reused
                        return result
            except (OSError, ValueError):
                self.close(); raise ProviderError("Lost connection to the GPU worker; retry the job.") from None
            finally:
                if self.process is not None:
                    self.timer = threading.Timer(900 if self.mode=="hunyuan" else 90, self._idle); self.timer.daemon = True; self.timer.start()

