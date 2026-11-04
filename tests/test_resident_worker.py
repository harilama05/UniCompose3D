"""Real process protocol and cancellation; no model/credentials/network."""
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unicompose3d.resident import ResidentWorker, gpu_session, release_residents

FIXTURE = """import json,sys,time,os
for line in sys.stdin:
    request=json.loads(line)
    print(json.dumps({'event':'progress','stage':'sam_mask'}),flush=True)
    if request.get('wait'): time.sleep(10)
    print(json.dumps({'event':'result','result':{'pid':os.getpid()}}),flush=True)
"""


class ResidentTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(); self.root=Path(self.temp.name)
        self.runner=self.root/"fixture.py"; self.runner.write_text(FIXTURE)
        self.workers=[]
    def tearDown(self):
        for worker in self.workers: worker.close()
        self.temp.cleanup()
    def worker(self):
        worker=ResidentWorker(sys.executable,"sam",self.root,self.root); worker.runner=self.runner
        self.workers.append(worker); return worker
    def test_same_process_reused_and_switching_workers_releases_previous_gpu_owner(self):
        stop=threading.Event(); one=self.worker()
        a=one.call({},self.root,stop); b=one.call({},self.root,stop)
        self.assertEqual(a["pid"],b["pid"]); self.assertTrue(b["worker_reused"])
        two=self.worker(); two.call({},self.root,stop)
        self.assertIsNone(one.process); self.assertIsNotNone(two.process)
    def test_switching_sam_and_hunyuan_parks_instead_of_reloading_parameters(self):
        stop=threading.Event(); sam=self.worker(); first=sam.call({},self.root,stop)
        sf=self.worker(); sf.mode="hunyuan"; sf_first=sf.call({},self.root,stop)
        self.assertIsNotNone(sam.process)
        again=sam.call({},self.root,stop)
        self.assertEqual(first["pid"],again["pid"])
        self.assertTrue(again["worker_reused"]); self.assertIsNotNone(sf.process)
        sf_again=sf.call({},self.root,stop)
        self.assertEqual(sf_first["pid"],sf_again["pid"])


    def test_cancel_kills_busy_worker_without_waiting_for_its_model_call(self):
        stop=threading.Event(); worker=self.worker()
        timer=threading.Timer(.2,stop.set); timer.start()
        begin=time.monotonic()
        with self.assertRaises(Exception): worker.call({"wait":True},self.root,stop)
        self.assertLess(time.monotonic()-begin,3); self.assertIsNone(worker.process)
        timer.join()


if __name__=="__main__": unittest.main()
